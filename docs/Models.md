# Models

Models live in `model-repo/` and are loaded by Triton. For production, prefer external storage or pre-baked images.

## Layout

```
model-repo/
  encoder/
    1/model.onnx
    config.pbtxt
  decoder_joint/
    1/model.onnx
    config.pbtxt
  preprocessor/
    1/model.onnx
    config.pbtxt
  vocab.txt
```

## Setup Options

- Git LFS for local development
- External storage (S3/GCS) for production
- Container images with models pre-loaded

## Download Script Example

```bash
mkdir -p model-repo/{encoder,decoder_joint,preprocessor}/1
# curl commands to fetch models into those directories
```
