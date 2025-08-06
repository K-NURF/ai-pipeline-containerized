# 🚀 Installation Guide

Complete guide to get the AI Pipeline up and running on your system.

## 📋 Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Ubuntu 20.04+, macOS 11+, Windows 10+ | Ubuntu 22.04+ |
| **Python** | 3.11+ | 3.11+ |
| **RAM** | 16GB | 32GB+ |
| **Storage** | 20GB free | 50GB+ free |
| **GPU** | Optional | NVIDIA RTX 3060+ (16GB VRAM) |
| **CPU** | 8 cores | 16+ cores |

### Software Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip
sudo apt install -y build-essential ffmpeg portaudio19-dev
sudo apt install -y docker.io docker-compose-v2

# macOS (with Homebrew)
brew install python@3.11 ffmpeg portaudio
brew install docker docker-compose

# NVIDIA GPU Support (Ubuntu)
sudo apt install nvidia-driver-535 nvidia-container-toolkit
sudo systemctl restart docker
```

## ⚡ Quick Start (Recommended)

### Option 1: Automated Setup

```bash
# Clone the repository
git clone https://github.com/your-org/ai-pipeline-containerized.git
cd ai-pipeline-containerized

# Run automated setup
python setup.py --quick-start
```

This script will:
- ✅ Check system requirements
- ✅ Install Python dependencies
- ✅ Set up all AI models (including faster-whisper)
- ✅ Create configuration files
- ✅ Verify the complete pipeline

### Option 2: Docker Compose (Production)

```bash
# Clone and start services
git clone https://github.com/your-org/ai-pipeline-containerized.git
cd ai-pipeline-containerized

# Copy environment configuration
cp .env.example .env

# Start all services
docker-compose up -d

# Check status
curl http://localhost:8123/health/detailed
```

## 🔧 Manual Installation

### Step 1: Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/ai-pipeline-containerized.git
cd ai-pipeline-containerized

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 2: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install spaCy model for NER
python -m spacy download en_core_web_md

# Verify CUDA (if available)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 3: Model Setup

The system uses **Faster-Whisper** for 4-5x faster speech recognition:

```bash
# Automatic model download (recommended)
python -c "
from app.models.whisper_model import WhisperModel
model = WhisperModel()
print('Loading Faster-Whisper model...')
success = model.load()
print('✅ Success!' if success else '❌ Failed')
"

# Verify all models
python setup.py --verify-only
```

### Step 4: Configuration

```bash
# Create configuration file
cp .env.example .env

# Edit configuration (optional)
nano .env
```

Key configuration options:
```bash
# Performance
MAX_CONCURRENT_GPU_REQUESTS=1
ENABLE_MODEL_LOADING=true

# Redis
REDIS_URL=redis://localhost:6379/0

# Debug mode
DEBUG=false
LOG_LEVEL=INFO
```

### Step 5: Start Services

#### Development Mode
```bash
# Start Redis
redis-server &

# Start Celery worker
celery -A app.celery_app worker --loglevel=info -E &

# Start API server
python -m app.main
```

#### Production Mode
```bash
# Use Docker Compose
docker-compose up -d
```

## 🧪 Verification

### Health Check
```bash
curl http://localhost:8123/health/detailed
```

Expected response:
```json
{
  "status": "healthy",
  "models": {
    "whisper": {"status": "loaded", "framework": "faster-whisper"},
    "translation": {"status": "loaded"},
    "ner": {"status": "loaded"},
    "classifier": {"status": "loaded"}
  },
  "system": {
    "gpu_available": true,
    "memory_usage": "8.2GB / 32GB"
  }
}
```

### Test Audio Processing
```bash
# Create test audio file (optional)
python -c "
import numpy as np
import soundfile as sf
sample_rate = 16000
duration = 5  # seconds
t = np.linspace(0, duration, int(sample_rate * duration))
audio = np.sin(2 * np.pi * 440 * t) * 0.3  # 440Hz sine wave
sf.write('test_audio.wav', audio, sample_rate)
print('Created test_audio.wav')
"

# Test complete pipeline
curl -X POST \
  -F "audio=@test_audio.wav" \
  -F "language=en" \
  -F "include_translation=false" \
  http://localhost:8123/audio/process
```

### Performance Benchmark
```bash
python -c "
import time
import numpy as np
from app.models.whisper_model import WhisperModel

model = WhisperModel()
model.load()

# Test 5-second audio processing
audio = np.zeros(16000 * 5, dtype=np.int16).tobytes()
start = time.time()
result = model.transcribe_pcm_audio(audio, 16000, 'en')
duration = time.time() - start

print(f'🚀 Faster-Whisper Performance:')
print(f'   Processing time: {duration:.3f} seconds')
print(f'   Speed ratio: {5/duration:.1f}x real-time')
print(f'   Result: \"{result}\"')
"
```

## 🐳 Docker Deployment

### Development Setup
```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  ai-pipeline:
    build: .
    ports:
      - "8123:8123"
    volumes:
      - .:/app
      - ./models:/app/models
    environment:
      - DEBUG=true
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
```

### Production Setup
```yaml
# docker-compose.yml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data

  ai-pipeline:
    build: .
    restart: unless-stopped
    ports:
      - "8123:8123"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  celery-worker:
    build: .
    restart: unless-stopped
    command: celery -A app.celery_app worker --loglevel=info -E --pool=solo
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  redis_data:
```

### Build and Deploy
```bash
# Build image
docker build -t ai-pipeline:latest .

# Start production stack
docker-compose up -d

# Scale workers
docker-compose up -d --scale celery-worker=3

# View logs
docker-compose logs -f ai-pipeline
```

## 🔧 Custom Model Integration

### Fine-tuned Whisper Models

If you have custom fine-tuned Whisper models:

```bash
# 1. Convert to CTranslate2 format
pip install ctranslate2>=4.0

ct2-transformers-converter \
  --model /path/to/your/finetuned/whisper \
  --output_dir ./models/whisper \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization float16

# 2. Verify conversion
ls -la ./models/whisper/
# Should show: config.json, model.bin, tokenizer.json, etc.

# 3. Test your model
python -c "
from app.models.whisper_model import WhisperModel
model = WhisperModel()
success = model.load()
print('✅ Custom model loaded!' if success else '❌ Failed')
"
```

### Model Directory Structure
```
models/
├── whisper/                    # Faster-Whisper (CTranslate2 format)
│   ├── config.json
│   ├── model.bin
│   ├── tokenizer.json
│   └── preprocessor_config.json
├── translation/                # Swahili-English translator
├── ner/                       # Named Entity Recognition
├── classifier/                # Case classification
├── summarization/             # Text summarization
└── all_qa_distilbert_v1/      # Quality assessment
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```bash
# Error: "Unable to open file 'model.bin'"
# Solution: Re-download faster-whisper model
rm -rf ./models/whisper/*
python -c "from app.models.whisper_model import WhisperModel; WhisperModel().load()"
```

#### 2. CUDA Out of Memory
```python
# Reduce precision in app/models/whisper_model.py
self.compute_type = "int8"  # Instead of "float16"
```

#### 3. Port Already in Use
```bash
# Check what's using port 8123
sudo lsof -i :8123

# Kill the process or use different port
export PORT=8124
python -m app.main
```

#### 4. Docker Permission Issues
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Fix volume permissions
sudo chown -R $USER:$USER ./models ./logs
```

#### 5. Slow Performance
```bash
# Check GPU usage
nvidia-smi

# Verify faster-whisper is using GPU
python -c "
import torch
from app.models.whisper_model import WhisperModel
model = WhisperModel()
model.load()
info = model.get_model_info()
print(f'Device: {info[\"device\"]}')
print(f'Framework: {info[\"framework\"]}')
"
```

### Performance Optimization

#### GPU Settings
```bash
# Set optimal CUDA settings
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TORCH_DTYPE=float16
```

#### Resource Monitoring
```bash
# Monitor system resources
htop
nvidia-smi -l 1

# Monitor API performance
curl -s http://localhost:8123/health/resources | jq
```

## 📊 Expected Performance

### Faster-Whisper Benchmarks

| Hardware | Model | Processing Speed | Memory Usage |
|----------|--------|------------------|--------------|
| RTX 4090 | large-v3-turbo | 30-50x real-time | ~6GB VRAM |
| RTX 3060 Ti | large-v3-turbo | 15-25x real-time | ~6GB VRAM |
| CPU (16 cores) | large-v3-turbo | 3-5x real-time | ~4GB RAM |

### Complete Pipeline Performance

| Audio Length | GPU Processing | CPU Processing |
|--------------|----------------|----------------|
| 30 seconds | 5-8 seconds | 15-25 seconds |
| 2 minutes | 15-25 seconds | 60-90 seconds |
| 10 minutes | 45-75 seconds | 300-450 seconds |

## 🔒 Security Considerations

### Production Deployment
```bash
# Use environment-specific configurations
cp .env.production .env

# Set secure settings
echo "DEBUG=false" >> .env
echo "SECURE_SSL_REDIRECT=true" >> .env
echo "MAX_FILE_SIZE_MB=100" >> .env
```

### Data Privacy
- ✅ **Offline Processing**: No data sent to external APIs
- ✅ **Local Models**: All AI processing happens locally
- ✅ **Data Retention**: Configure automatic cleanup
- ✅ **PII Protection**: Automatic sensitive data detection

## 📚 Next Steps

Once installation is complete:

1. **📖 Read the Documentation**
   - [Model Setup Guide](docs/model-setup-guide.md)
   - [API Documentation](README.md#-api-documentation)
   - [Streaming Guide](STREAMING_GUIDE.md)

2. **🧪 Run Tests**
   ```bash
   python -m pytest tests/
   ```

3. **🚀 Deploy to Production**
   - Configure reverse proxy (nginx)
   - Set up monitoring and logging
   - Configure auto-scaling

4. **🔧 Customize for Your Use Case**
   - Fine-tune models for your domain
   - Add custom classification categories
   - Integrate with your existing systems

## 🤝 Getting Help

- **📚 Documentation**: Check the [docs/](docs/) directory
- **🐛 Issues**: Report bugs on GitHub Issues  
- **💬 Discussions**: Ask questions in GitHub Discussions
- **📧 Support**: Contact the development team

---

**Ready to transform audio into insights!** 🎉

For more advanced configuration and deployment options, see the main [README.md](README.md).