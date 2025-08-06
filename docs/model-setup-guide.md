# 🤖 Model Setup Guide

This guide walks you through setting up all AI models for the pipeline, including the faster-whisper upgrade and custom model integration.

## 📋 Quick Start Checklist

- [ ] Install system dependencies
- [ ] Set up Python environment  
- [ ] Download/convert models
- [ ] Verify model loading
- [ ] Test complete pipeline

## 🚀 Faster-Whisper Setup (Speech Recognition)

### What is Faster-Whisper?

**Faster-Whisper** is a CTranslate2-optimized version of OpenAI's Whisper that provides:
- **4-5x faster inference** than standard Whisper
- **Same accuracy** as original models
- **Lower memory usage** with quantization support
- **GPU-optimized** with float16 precision

### Automatic Model Download

The system will automatically download the faster-whisper model on first use:

```python
# This happens automatically when the model loads
from app.models.whisper_model import WhisperModel
model = WhisperModel()
model.load()  # Downloads faster-whisper large-v3-turbo automatically
```

**Model Location**: `/models/whisper/` (CTranslate2 format)
**Expected Files**:
```
/models/whisper/
├── config.json
├── model.bin              # CTranslate2 format
├── tokenizer.json
├── preprocessor_config.json
└── vocabulary.txt
```

### Manual Model Management

If you want to pre-download or manage models manually:

```bash
# Option 1: Let the system auto-download (recommended)
python -c "from app.models.whisper_model import WhisperModel; WhisperModel().load()"

# Option 2: Download using faster-whisper CLI
python -c "
from faster_whisper import WhisperModel
model = WhisperModel('large-v3-turbo')
print('Model downloaded successfully!')
"
```

## 🎯 Custom Model Integration

### Fine-tuned Whisper Models

If you have fine-tuned Whisper models, you need to convert them to CTranslate2 format:

#### 1. Install Conversion Tools
```bash
pip install transformers[torch]>=4.23
# ct2-transformers-converter is included with faster-whisper
```

#### 2. Convert Your Model
```bash
# Convert HuggingFace model to CTranslate2
ct2-transformers-converter \
  --model /path/to/your/finetuned/whisper \
  --output_dir /app/models/whisper \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization float16
```

#### 3. Verify Conversion
```python
python -c "
from app.models.whisper_model import WhisperModel
model = WhisperModel()
success = model.load()
print('✅ Model loaded successfully!' if success else '❌ Model loading failed')
print(f'Model info: {model.get_model_info()}')
"
```

### Example: Converting a HuggingFace Model

```bash
# Example: Convert a fine-tuned model from HuggingFace Hub
ct2-transformers-converter \
  --model "your-username/whisper-finetuned-swahili" \
  --output_dir ./models/whisper \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization float16

# Verify the conversion
ls -la ./models/whisper/
# Should show: config.json, model.bin, tokenizer.json, etc.
```

## 🔧 Other Model Setup

### 1. Translation Model (Swahili ↔ English)

**Location**: `/models/translation/`
**Format**: HuggingFace Transformers

```python
# Verify translation model
python -c "
from app.models.translator_model import TranslatorModel
model = TranslatorModel()
success = model.load()
print('✅ Translation model ready!' if success else '❌ Translation model failed')
"
```

### 2. NER Model (Named Entity Recognition)

**Location**: `/models/ner/`
**Format**: spaCy model

```bash
# Download spaCy model if needed
python -m spacy download en_core_web_md
```

### 3. Classification Model

**Location**: `/models/classifier/`
**Format**: HuggingFace Transformers

```python
# Verify classifier
python -c "
from app.models.classifier_model import ClassifierModel
model = ClassifierModel()
success = model.load()
print('✅ Classifier ready!' if success else '❌ Classifier failed')
"
```

### 4. Summarization Model

**Location**: `/models/summarization/`
**Format**: HuggingFace Transformers

### 5. QA Model (Quality Assessment)

**Location**: `/models/all_qa_distilbert_v1/`
**Format**: HuggingFace Transformers

## 📊 Model Verification

### Complete System Check

```python
python -c "
from app.models.model_loader import ModelLoader
import logging
logging.basicConfig(level=logging.INFO)

print('🔍 Checking all models...')
loader = ModelLoader()
loader.load_all_models()

print('\\n📊 Model Status:')
for name, status in loader.model_status.items():
    print(f'   {name}: {\"✅ Ready\" if status.loaded else \"❌ Failed\"}')
"
```

### Performance Test

```python
# Test whisper performance
python -c "
import time
import numpy as np
from app.models.whisper_model import WhisperModel

model = WhisperModel()
model.load()

# Create test audio (3 seconds of silence)
audio = np.zeros(16000 * 3, dtype=np.int16).tobytes()

start_time = time.time()
transcript = model.transcribe_pcm_audio(audio, sample_rate=16000, language='en')
processing_time = time.time() - start_time

print(f'🚀 Processing time: {processing_time:.3f} seconds')
print(f'📝 Result: \"{transcript}\"')
"
```

## 🐳 Docker Model Setup

### Model Persistence

When using Docker, ensure models persist across container restarts:

```yaml
# docker-compose.yml
services:
  ai-pipeline:
    volumes:
      - ./models:/app/models  # Persist models
      - model_cache:/root/.cache  # Cache HuggingFace downloads

volumes:
  model_cache:
```

### Pre-build Models in Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download models (optional)
RUN python -c "
from faster_whisper import WhisperModel;
WhisperModel('large-v3-turbo', device='cpu')
print('Whisper model pre-downloaded')
"

# Copy application
COPY . /app
WORKDIR /app
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```bash
# Error: "Unable to open file 'model.bin'"
# Solution: Ensure CTranslate2 format conversion
ct2-transformers-converter --model openai/whisper-large-v3-turbo \
  --output_dir ./models/whisper --quantization float16
```

#### 2. CUDA Out of Memory
```python
# Reduce model precision
# In WhisperModel initialization:
self.compute_type = "int8"  # Instead of "float16"
```

#### 3. Slow Performance
```bash
# Check GPU utilization
nvidia-smi

# Verify faster-whisper is using GPU
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"
```

### Model Size Requirements

| Model | Size | VRAM | Description |
|-------|------|------|-------------|
| Faster-Whisper Large-v3-Turbo | ~1.5GB | ~6GB | Speech recognition |
| Translation Model | ~500MB | ~2GB | Swahili ↔ English |
| NER Model (spaCy) | ~50MB | CPU | Named entity recognition |
| Classifier Model | ~250MB | ~1GB | Case classification |
| Summarization Model | ~500MB | ~2GB | Text summarization |
| **Total** | **~2.8GB** | **~11GB** | Complete pipeline |

### Performance Optimization

#### GPU Settings
```bash
# Optimize CUDA settings
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
```

#### Model Quantization
```python
# Use different quantization for different hardware
# GPU: float16 (fastest)
# CPU: int8 (memory efficient)
model = FasterWhisperModel(
    model_size_or_path="large-v3-turbo",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8"
)
```

## 📚 Additional Resources

### Faster-Whisper Documentation
- [GitHub Repository](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2 Documentation](https://opennmt.net/CTranslate2/)

### Model Conversion Tools
```bash
# Useful commands for model management
pip install ctranslate2>=4.0
pip install faster-whisper>=1.2.0
pip install transformers[torch]>=4.23
```

### Custom Model Training

If you want to fine-tune models:

1. **Whisper Fine-tuning**: Use HuggingFace transformers
2. **Convert to CTranslate2**: Use ct2-transformers-converter
3. **Replace Models**: Copy to `/models/whisper/`
4. **Test Pipeline**: Verify end-to-end functionality

## 🎯 Next Steps

Once models are set up:
1. ✅ **Verify Installation**: Run the complete system check
2. 🚀 **Start Services**: `docker-compose up -d`
3. 🧪 **Test Pipeline**: Upload a sample audio file
4. 📊 **Monitor Performance**: Check processing times and resource usage

For production deployment, see the main [README.md](../README.md) for scaling and optimization guidelines.