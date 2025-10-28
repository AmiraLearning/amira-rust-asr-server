# FST Builder Tools

Tools for building personalized language models and FSTs for the k2_decoder backend.

## Overview

This directory contains tools for creating per-user language models from historical transcripts:

1. **`compile_user_fst.py`** - Compile single ARPA LM to FST
2. **`build_lm_from_transcripts.py`** - Full pipeline: transcripts → ARPA → FST

## Installation

### Dependencies

```bash
# macOS
brew install kenlm

# Ubuntu/Debian
sudo apt-get install build-essential libboost-all-dev libeigen3-dev
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

# Python packages
pip install k2 torch kaldi_native_io
```

### Verify Installation

```bash
# Check KenLM
lmplz --help

# Check k2
python -c "import k2; print(k2.__version__)"
```

---

## Quick Start

### Build LM from User Transcripts

```bash
# Single user
python build_lm_from_transcripts.py \
    --user-id user123 \
    --transcripts-dir /data/transcripts/user123 \
    --output-dir /models/user_fsts \
    --order 3 \
    --min-utterances 50

# Output:
# /models/user_fsts/user123/
#   ├── G.fst           # k2 FST graph
#   ├── lm.arpa         # Intermediate ARPA LM
#   └── metadata.json   # Build metadata
```

### Batch Build for All Users

```bash
# Process all users in directory
python build_lm_from_transcripts.py \
    --batch \
    --transcripts-root /data/transcripts \
    --output-dir /models/user_fsts \
    --order 3 \
    --min-utterances 50

# Directory structure:
# /data/transcripts/
#   ├── user001/
#   │   ├── session_001.txt
#   │   └── session_002.txt
#   ├── user002/
#   │   └── ...
#   └── ...
```

### Load from Database Export

```bash
# Export transcripts from database to JSON
psql -d mydb -c "
  SELECT json_build_object(
    'user_id', user_id,
    'transcripts', json_agg(json_build_object(
      'text', transcript_text,
      'timestamp', created_at
    ))
  )
  FROM transcripts
  WHERE user_id = 'user123'
  GROUP BY user_id
" -t -o user123_transcripts.json

# Build LM from JSON
python build_lm_from_transcripts.py \
    --user-id user123 \
    --from-json user123_transcripts.json \
    --output-dir /models/user_fsts
```

---

## Detailed Usage

### build_lm_from_transcripts.py

Full pipeline for building user language models from raw transcripts.

#### Options

**Input Options:**
- `--user-id USER_ID` - Process single user
- `--batch` - Process all users in transcripts root
- `--transcripts-dir DIR` - Directory with transcript files (for single user)
- `--transcripts-root DIR` - Root directory with user subdirs (for batch)
- `--from-json FILE` - Load transcripts from JSON export

**Output Options:**
- `--output-dir DIR` - Output directory for user FSTs (required)

**Language Model Options:**
- `--order N` - N-gram order (default: 3 for trigrams)
- `--prune "X Y Z"` - Pruning thresholds for each order (default: "0 0 1")
  - Example: "0 0 1" = keep all unigrams/bigrams, prune singleton trigrams
  - Example: "0 1 1" = prune singleton bigrams and trigrams
- `--vocab-size N` - Acoustic model vocabulary size (default: 1030)
- `--min-utterances N` - Minimum utterances required (default: 50)

**Other Options:**
- `--force` - Overwrite existing FSTs
- `--max-users N` - Limit number of users in batch mode (for testing)
- `--kenlm-bin PATH` - Path to KenLM lmplz binary (default: lmplz)
- `--verbose` - Enable debug logging

#### Examples

```bash
# Basic single user
python build_lm_from_transcripts.py \
    --user-id alice \
    --transcripts-dir /data/alice_transcripts \
    --output-dir /models/user_fsts

# Batch with custom settings
python build_lm_from_transcripts.py \
    --batch \
    --transcripts-root /data/all_transcripts \
    --output-dir /models/user_fsts \
    --order 4 \
    --prune "0 1 1 2" \
    --min-utterances 100 \
    --verbose

# Test on 10 users only
python build_lm_from_transcripts.py \
    --batch \
    --transcripts-root /data/transcripts \
    --output-dir /tmp/test_fsts \
    --max-users 10

# Force rebuild
python build_lm_from_transcripts.py \
    --user-id bob \
    --transcripts-dir /data/bob \
    --output-dir /models/user_fsts \
    --force
```

---

## Text Cleaning

The script automatically cleans transcripts:

### Removed:
- Noise tokens: `[noise]`, `[laughter]`, `[music]`, `[inaudible]`
- URLs and email addresses
- Extra punctuation (except: `.`, `?`, `!`, `,`)
- Extra whitespace

### Normalized:
- Converted to lowercase
- Whitespace normalization

### Preserved:
- Punctuation that helps n-gram boundaries (`.`, `?`, `!`, `,`)
- Numbers (configurable - can normalize to `<number>`)
- Apostrophes (for contractions like "don't")

### Custom Cleaning

To add custom cleaning rules, modify `TranscriptProcessor.clean_transcript()`:

```python
# In build_lm_from_transcripts.py
class TranscriptProcessor:
    def clean_transcript(self, text: str) -> str:
        # ... existing cleaning ...

        # Add your custom rules
        text = text.replace('um', '')  # Remove filler words
        text = text.replace('uh', '')

        return text
```

---

## Transcript File Formats

### Plain Text (`.txt`)

One utterance per line:

```
hello this is a test transcript
the weather is nice today
i need to schedule a meeting
```

### JSON (`.json`)

```json
{
  "user_id": "user123",
  "transcripts": [
    {"text": "hello world", "timestamp": "2025-01-01T00:00:00Z"},
    {"text": "goodbye world", "timestamp": "2025-01-01T00:01:00Z"}
  ]
}
```

---

## Language Model Parameters

### N-gram Order

- **Order 2 (bigram):** Fast, lower memory, less context
- **Order 3 (trigram):** Recommended balance (default)
- **Order 4 (4-gram):** More context, higher memory
- **Order 5+:** Diminishing returns, exponential memory growth

### Pruning

KenLM pruning format: `"threshold_1 threshold_2 ... threshold_n"`

- `0` = keep all n-grams of that order
- `1` = remove singletons (n-grams seen once)
- `2` = remove n-grams seen ≤2 times

**Examples:**
```bash
# No pruning (large model, highest accuracy)
--prune "0 0 0"

# Conservative (default): prune only singleton trigrams
--prune "0 0 1"

# Aggressive: prune singleton bigrams and trigrams
--prune "0 1 1"

# Very aggressive: prune rare n-grams
--prune "0 2 2"
```

**Impact:**
- Less pruning = larger FST, more memory, slightly better accuracy
- More pruning = smaller FST, less memory, slightly worse accuracy
- Recommended: Start with `"0 0 1"` and adjust based on memory

---

## Output Structure

```
/models/user_fsts/
├── user001/
│   ├── G.fst           # k2 FST (used by Triton backend)
│   ├── lm.arpa         # Intermediate ARPA LM (can delete if space constrained)
│   └── metadata.json   # Build metadata
├── user002/
│   └── ...
└── user003/
    └── ...
```

### Metadata Format

```json
{
  "user_id": "user123",
  "utterances_count": 523,
  "order": 3,
  "prune": "0 0 1",
  "vocab_size": 1030
}
```

---

## Integration with Triton

### Configure k2_decoder Backend

Update `model-repo/k2_decoder/config.pbtxt`:

```protobuf
parameters: {
  key: "USER_FST_DIR"
  value: { string_value: "/models/user_fsts" }
}

parameters: {
  key: "MAX_CACHED_FSTS"
  value: { string_value: "100" }
}
```

### Send Requests with user_id

```python
import tritonclient.grpc as grpcclient
import numpy as np

client = grpcclient.InferenceServerClient("localhost:8001")

# Prepare inputs
encoder_outputs = np.random.rand(1, 100, 1030).astype(np.float32)
user_id = np.array([["user123"]], dtype=object)

inputs = [
    grpcclient.InferInput("encoder_outputs", encoder_outputs.shape, "FP32"),
    grpcclient.InferInput("user_id", user_id.shape, "BYTES"),
]
inputs[0].set_data_from_numpy(encoder_outputs)
inputs[1].set_data_from_numpy(user_id)

# Run inference with personalized LM
result = client.infer("k2_decoder", inputs)
tokens = result.as_numpy("tokens")
```

---

## Performance Tuning

### Build Speed

**Parallelize batch builds:**
```bash
# Use GNU parallel for faster batch processing
ls /data/transcripts | parallel -j8 \
    python build_lm_from_transcripts.py \
        --user-id {} \
        --transcripts-dir /data/transcripts/{} \
        --output-dir /models/user_fsts
```

**Reduce order for speed:**
```bash
# Order 2 is ~4x faster than order 3
--order 2
```

### FST Size

**Typical sizes:**
- Order 2: 5-15 MB per user
- Order 3: 10-50 MB per user
- Order 4: 30-150 MB per user

**Reduce size:**
1. Increase pruning: `--prune "0 1 1"`
2. Lower order: `--order 2`
3. Filter low-frequency words before training

### Memory Usage

**KenLM memory during training:**
- Roughly 3-5x the size of input corpus
- 100k utterances ≈ 2-4 GB RAM

**GPU memory in Triton:**
- Each cached FST: 10-50 MB
- 100 users cached: 1-5 GB GPU memory
- Adjust `MAX_CACHED_FSTS` based on GPU capacity

---

## Troubleshooting

### KenLM Not Found

```
RuntimeError: KenLM not found at lmplz
```

**Solution:**
```bash
# macOS
brew install kenlm

# Linux
git clone https://github.com/kpu/kenlm.git
cd kenlm && mkdir build && cd build
cmake .. && make && sudo make install

# Verify
which lmplz
```

### Too Few Utterances

```
WARNING: User user123 has only 23 utterances (min: 50), skipping
```

**Solution:**
```bash
# Lower minimum
--min-utterances 20

# Or collect more data for the user
```

### GPU Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Reduce `MAX_CACHED_FSTS` in Triton config
2. Use more aggressive pruning
3. Lower n-gram order

### Import Errors

```
ModuleNotFoundError: No module named 'k2'
```

**Solution:**
```bash
# Install PyTorch first (with CUDA if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install k2
pip install k2

# Verify
python -c "import k2; print(k2.__version__)"
```

---

## Advanced Usage

### Custom Vocabulary

Restrict LM to known vocabulary:

```bash
# Create vocab file (one word per line)
cat > vocab.txt <<EOF
hello
world
goodbye
EOF

# Build LM with closed vocabulary
# (Modify build_lm_from_transcripts.py to pass vocab_path to train_arpa)
```

### Interpolate with Base LM

Mix user LM with general LM:

```bash
# Build base LM from large corpus
python build_lm_from_transcripts.py \
    --user-id base \
    --transcripts-dir /data/general_corpus \
    --output-dir /models/base_lms

# Interpolate in Triton (set lm_weight per request)
# lm_weight=0.0 → use base only
# lm_weight=0.5 → mix 50/50
# lm_weight=1.0 → use user only
```

### Continuous Updates

Update user LMs as new transcripts arrive:

```bash
# Cron job to rebuild daily
0 2 * * * /usr/bin/python /path/to/build_lm_from_transcripts.py \
    --batch \
    --transcripts-root /data/transcripts \
    --output-dir /models/user_fsts \
    --force \
    >> /var/log/lm_builder.log 2>&1
```

---

## See Also

- [k2 Documentation](https://k2-fsa.github.io/k2/)
- [KenLM Documentation](https://kheafield.com/code/kenlm/)
- [Triton Inference Server](https://github.com/triton-inference-server)
- [Personalized ASR Guide](../../docs/Personalized_ASR.md)
- [k2_decoder Audit Report](../../src/triton_backends/k2_decoder/AUDIT_REPORT.md)

---

## Contributing

To add features to the LM builder:

1. Fork the repo
2. Add your changes to `build_lm_from_transcripts.py`
3. Test with sample data
4. Submit PR with:
   - Description of change
   - Example usage
   - Performance impact (if applicable)

## License

MIT License - See main project LICENSE file
