# Model Management

This repository uses external model storage to keep the repository size manageable.

## Model Files

The following model files are tracked with Git LFS:
- `*.onnx` - ONNX model files
- `*.pth` - PyTorch model files  
- `*.pt` - PyTorch model files

## Setting up models

### Option 1: Git LFS (Current)
Models are automatically downloaded with `git lfs pull` after cloning.

### Option 2: External Storage (Recommended for production)
For production deployments, consider moving models to:
- AWS S3 / Google Cloud Storage
- Model registries (MLflow, Weights & Biases)
- Container images with models pre-loaded

### Option 3: Download script
Create a `download_models.sh` script to fetch models from your preferred storage:

```bash
#!/bin/bash
mkdir -p model-repo/encoder/1
mkdir -p model-repo/decoder_joint/1  
mkdir -p model-repo/preprocessor/1

# Download from your model storage
curl -o model-repo/encoder/1/model.onnx "https://your-storage.com/encoder.onnx"
curl -o model-repo/decoder_joint/1/model.onnx "https://your-storage.com/decoder.onnx"
curl -o model-repo/preprocessor/1/model.onnx "https://your-storage.com/preprocessor.onnx"
```

## Model Repository Structure

```
model-repo/
├── encoder/
│   ├── 1/
│   │   └── model.onnx          # ~40MB - Main encoder model
│   └── config.pbtxt
├── decoder_joint/
│   ├── 1/  
│   │   └── model.onnx          # ~34MB - Decoder/joint model
│   └── config.pbtxt
├── preprocessor/
│   ├── 1/
│   │   └── model.onnx          # ~137KB - Audio preprocessor
│   └── config.pbtxt
└── vocab.txt                   # Vocabulary file
```

## Repository Size Impact

- **Before**: 2.3GB in `model-repo/`
- **After**: ~150KB (LFS pointers) or 0KB (external storage)

Models are the largest contributor to repository size after build artifacts and third-party dependencies.