#ifndef K2_DECODER_FST_CACHE_H_
#define K2_DECODER_FST_CACHE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <functional>
#include <fstream>
#include <filesystem>

#include "k2/csrc/fsa.h"
#include <torch/torch.h>

namespace triton { namespace backend { namespace k2_decoder {

/// Simple thread pool for async FST loading
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 4) : stop_(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });

                        if (stop_ && tasks_.empty()) {
                            return;
                        }

                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();

        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void Enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace(std::move(task));
        }
        condition_.notify_one();
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};

/// Thread-safe LRU cache for per-user FST graphs on GPU
///
/// This cache manages FST graphs in GPU memory with automatic eviction
/// based on least-recently-used policy when memory limits are reached.
///
/// Features:
/// - Thread-safe operations with minimal lock contention
/// - Lazy loading from disk on cache miss
/// - Automatic GPU memory management
/// - Telemetry for cache hits/misses
/// - Fallback to base graph if user FST not found
class FstCache {
public:
    /// Statistics for monitoring cache performance
    struct CacheStats {
        size_t hits = 0;
        size_t misses = 0;
        size_t evictions = 0;
        size_t load_failures = 0;
        size_t current_size = 0;
        size_t max_size = 0;

        double hit_rate() const {
            size_t total = hits + misses;
            return total > 0 ? static_cast<double>(hits) / total : 0.0;
        }
    };

    /// Configuration for FST cache behavior
    struct CacheConfig {
        std::string fst_directory;     // Root directory for user FSTs
        size_t max_cached_fsts = 100;  // Maximum number of FSTs to cache
        int32_t device_id = 0;         // CUDA device ID
        bool enable_telemetry = true;  // Enable hit/miss tracking
    };

    /// Create a new FST cache
    ///
    /// @param config Cache configuration
    explicit FstCache(const CacheConfig& config);

    /// Get or load a user's FST graph on GPU
    ///
    /// This method is thread-safe and handles:
    /// - Cache hit: Return cached FST immediately
    /// - Cache miss: Load from disk, move to GPU, cache, return
    /// - Load failure: Return nullptr (caller should use base graph)
    ///
    /// @param user_id User identifier (used as directory name)
    /// @return Pointer to GPU FST, or nullptr if not found
    std::shared_ptr<k2::Fsa> GetOrLoad(const std::string& user_id);

    /// Preload a user's FST into cache asynchronously
    ///
    /// Useful for warming the cache before requests arrive.
    /// Non-blocking - returns immediately and loads in background.
    ///
    /// @param user_id User identifier to preload
    void PreloadAsync(const std::string& user_id);

    /// Explicitly evict a user's FST from cache
    ///
    /// Useful for testing or manual cache management.
    ///
    /// @param user_id User identifier to evict
    /// @return true if FST was in cache and evicted
    bool Evict(const std::string& user_id);

    /// Clear entire cache
    ///
    /// Removes all cached FSTs. Useful for testing or when
    /// FSTs on disk have been updated.
    void Clear();

    /// Get current cache statistics
    ///
    /// @return Copy of current statistics
    CacheStats GetStats() const;

    /// Get path to user's FST file
    ///
    /// Expected structure: {fst_directory}/{user_id}/G.fst
    ///
    /// @param user_id User identifier
    /// @return Full path to FST file
    std::string GetFstPath(const std::string& user_id) const;

    /// Check if a user FST exists on disk
    ///
    /// @param user_id User identifier
    /// @return true if FST file exists and is readable
    bool FstExists(const std::string& user_id) const;

private:
    /// Cache entry storing FST and access metadata
    struct CacheEntry {
        std::shared_ptr<k2::Fsa> fst_gpu;
        std::list<std::string>::iterator lru_it;
    };

    /// Load FST from disk and move to GPU
    ///
    /// Internal method called on cache miss.
    ///
    /// @param user_id User identifier
    /// @return Loaded FST on GPU, or nullptr on failure
    std::shared_ptr<k2::Fsa> LoadFstFromDisk(const std::string& user_id);

    /// Evict least recently used FST if cache is full
    ///
    /// Called before adding new FST when at capacity.
    void EvictLruIfNeeded();

    /// Update LRU order for a user_id
    ///
    /// Moves the user to the front of the LRU list.
    ///
    /// @param user_id User identifier to mark as recently used
    void TouchEntry(const std::string& user_id);

    // Configuration
    CacheConfig config_;

    // Cache data structures
    std::unordered_map<std::string, CacheEntry> cache_;
    std::list<std::string> lru_list_;  // Front = most recent, back = least recent

    // Track FSTs currently being loaded (prevents duplicate loads)
    std::unordered_set<std::string> loading_;

    // Thread pool for async preloading
    std::unique_ptr<ThreadPool> thread_pool_;

    // Thread safety
    mutable std::mutex mutex_;

    // Statistics
    mutable CacheStats stats_;
};

// Implementation details

inline FstCache::FstCache(const CacheConfig& config)
    : config_(config), thread_pool_(std::make_unique<ThreadPool>(4)) {
    stats_.max_size = config_.max_cached_fsts;

    if (config_.fst_directory.empty()) {
        LOG_MESSAGE(TRITONSERVER_LOG_WARN,
            "FST cache initialized with empty directory - all loads will fail");
    } else {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
            ("FST cache initialized: directory=" + config_.fst_directory +
             ", max_size=" + std::to_string(config_.max_cached_fsts) +
             ", device=" + std::to_string(config_.device_id)).c_str());
    }
}

inline std::shared_ptr<k2::Fsa> FstCache::GetOrLoad(const std::string& user_id) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Check if already cached
    auto it = cache_.find(user_id);
    if (it != cache_.end()) {
        if (config_.enable_telemetry) {
            stats_.hits++;
        }
        TouchEntry(user_id);
        return it->second.fst_gpu;
    }

    // Check if already being loaded by another thread
    if (loading_.count(user_id) > 0) {
        // Another thread is loading this FST, return nullptr
        // (caller will use base graph for this request)
        LOG_MESSAGE(TRITONSERVER_LOG_DEBUG,
            ("FST for user_id=" + user_id + " is being loaded by another thread").c_str());
        return nullptr;
    }

    // Cache miss - need to load
    if (config_.enable_telemetry) {
        stats_.misses++;
    }

    // Mark as loading
    loading_.insert(user_id);

    // Release lock while loading from disk (I/O bound operation)
    lock.unlock();
    auto fst_gpu = LoadFstFromDisk(user_id);
    lock.lock();

    // Done loading
    loading_.erase(user_id);

    if (!fst_gpu) {
        // Load failed - return nullptr, caller will use base graph
        if (config_.enable_telemetry) {
            stats_.load_failures++;
        }
        return nullptr;
    }

    // Evict LRU entry if cache is full
    EvictLruIfNeeded();

    // Add to cache
    lru_list_.push_front(user_id);
    cache_[user_id] = {fst_gpu, lru_list_.begin()};
    stats_.current_size = cache_.size();

    return fst_gpu;
}

inline std::shared_ptr<k2::Fsa> FstCache::LoadFstFromDisk(const std::string& user_id) {
    std::string fst_path = GetFstPath(user_id);

    if (!FstExists(user_id)) {
        LOG_MESSAGE(TRITONSERVER_LOG_WARN,
            ("User FST not found: " + fst_path + " - will use base graph").c_str());
        return nullptr;
    }

    try {
        // Open FST file
        std::ifstream is(fst_path, std::ios::binary);
        if (!is.is_open()) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                ("Failed to open FST file: " + fst_path).c_str());
            return nullptr;
        }

        // Load FST from file (on CPU)
        auto fst_cpu = std::make_shared<k2::Fsa>(k2::Fsa::Read(is));

        // Move to GPU
        auto fst_gpu = std::make_shared<k2::Fsa>(
            fst_cpu->To(torch::kCUDA, config_.device_id));

        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
            ("Loaded user FST for user_id=" + user_id + " to GPU device " +
             std::to_string(config_.device_id)).c_str());

        return fst_gpu;
    } catch (const std::exception& e) {
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
            ("Exception loading FST for user_id=" + user_id + ": " +
             std::string(e.what())).c_str());
        return nullptr;
    }
}

inline void FstCache::EvictLruIfNeeded() {
    // Assumes mutex is already locked
    if (cache_.size() >= config_.max_cached_fsts) {
        // Evict least recently used
        std::string lru_user_id = lru_list_.back();
        lru_list_.pop_back();
        cache_.erase(lru_user_id);

        if (config_.enable_telemetry) {
            stats_.evictions++;
        }

        LOG_MESSAGE(TRITONSERVER_LOG_DEBUG,
            ("Evicted LRU FST for user_id=" + lru_user_id).c_str());
    }
}

inline void FstCache::TouchEntry(const std::string& user_id) {
    // Assumes mutex is already locked
    auto it = cache_.find(user_id);
    if (it != cache_.end()) {
        // Move to front of LRU list
        lru_list_.erase(it->second.lru_it);
        lru_list_.push_front(user_id);
        it->second.lru_it = lru_list_.begin();
    }
}

inline bool FstCache::Evict(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(user_id);
    if (it == cache_.end()) {
        return false;
    }

    lru_list_.erase(it->second.lru_it);
    cache_.erase(it);
    stats_.current_size = cache_.size();

    LOG_MESSAGE(TRITONSERVER_LOG_DEBUG,
        ("Manually evicted FST for user_id=" + user_id).c_str());

    return true;
}

inline void FstCache::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    lru_list_.clear();
    stats_.current_size = 0;

    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "FST cache cleared");
}

inline FstCache::CacheStats FstCache::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

inline std::string FstCache::GetFstPath(const std::string& user_id) const {
    // Expected structure: {fst_directory}/{user_id}/G.fst
    std::filesystem::path fst_path = config_.fst_directory;
    fst_path /= user_id;
    fst_path /= "G.fst";
    return fst_path.string();
}

inline bool FstCache::FstExists(const std::string& user_id) const {
    std::string fst_path = GetFstPath(user_id);
    return std::filesystem::exists(fst_path) &&
           std::filesystem::is_regular_file(fst_path);
}

inline void FstCache::PreloadAsync(const std::string& user_id) {
    // Async preload using thread pool (prevents unbounded thread spawning)
    thread_pool_->Enqueue([this, user_id]() {
        this->GetOrLoad(user_id);
    });
}

}}} // namespace triton::backend::k2_decoder

#endif  // K2_DECODER_FST_CACHE_H_
