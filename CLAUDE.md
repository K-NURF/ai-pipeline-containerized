# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Commands

### Development
```bash
# Start development server directly
python -m app.main

# Install dependencies
pip install -r requirements.txt

# Install spaCy model (required for NER)
python -m spacy download en_core_web_md
```

### Docker Development
```bash
# Start complete stack (API + Redis + Celery + Flower)
docker-compose up -d

# Start with optimized configuration
docker-compose -f docker-compose.local-optimized.yml up -d

# View logs
docker-compose logs -f ai-pipeline
docker-compose logs -f celery-worker

# Stop and remove
docker-compose down
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_text_chunker.py -v

# Run with coverage
python -m pytest --cov=app tests/

# Run benchmark tests (marked with @pytest.mark.benchmark)
python -m pytest tests/ -m benchmark -v
```

### Health Checks
```bash
# API health
curl http://localhost:8123/health/detailed

# Worker status
curl http://localhost:8123/audio/workers/status

# Queue status
curl http://localhost:8123/audio/queue/status

# Model status
curl http://localhost:8123/health/models
```

## Architecture Overview

This is a **multi-modal AI pipeline** for processing audio files into structured insights, designed for child protection and social services.

### Core Pipeline Flow
**Audio → Transcription (Whisper) → Translation (Swahili↔English) → NLP Analysis (NER, Classification, Summarization) → Structured Output**

### Service Architecture
- **FastAPI Gateway** (`app/main.py`): REST API with automatic documentation at `/docs`
- **Celery Workers** (`app/celery_app.py`): Distributed task processing with GPU management  
- **Redis**: Message broker and task queue
- **Model Layer**: Separate model handlers in `app/models/`

### Key Components

#### API Routes (`app/api/`)
- `audio_routes.py`: Complete pipeline endpoints (`/audio/process`, `/audio/analyze`)
- `whisper_routes.py`: Speech-to-text transcription
- `translator_routes.py`: Language translation
- `ner_routes.py`: Named entity recognition
- `classifier_route.py`: Case classification
- `summarizer_routes.py`: Text summarization
- `realtime_routes.py`: Real-time transcription WebSocket

#### Core Processing (`app/core/`)
- `audio_pipeline.py`: Main processing orchestration
- `text_chunker.py`: Intelligent text chunking for different models
- `resource_manager.py`: GPU memory and system resource management
- `request_queue.py`: Queue management with priority handling

#### Models (`app/models/`)
- `whisper_model.py`: OpenAI Whisper Large V3 Turbo
- `translator_model.py`: Custom fine-tuned Swahili-English translation
- `ner_model.py`: spaCy-based named entity recognition
- `classifier_model.py`: DistilBERT for case classification
- `summarizer_model.py`: Text summarization model

#### Configuration (`app/config/`)
- `settings.py`: Pydantic settings with environment variable support
- Key settings: `ENABLE_MODEL_LOADING`, `MAX_CONCURRENT_GPU_REQUESTS`, `REDIS_URL`

### Model Loading Strategy
- **API Server Mode** (`ENABLE_MODEL_LOADING=false`): Lightweight FastAPI server, models loaded by workers
- **Worker Mode** (`ENABLE_MODEL_LOADING=true`): Celery workers with GPU access load and run models

### Real-time Features
- WebSocket endpoint for streaming transcription (`app/realtime/`)
- Real-time audio processing with incremental results
- GPU-accelerated inference with resource management

## Development Patterns

### Adding New Models
1. Create model wrapper in `app/models/your_model.py` with `load()` and `process()` methods
2. Register in `app/models/model_loader.py` dependencies dict
3. Add API routes in `app/api/your_model_routes.py`
4. Update main pipeline in `app/core/audio_pipeline.py` if needed

### Text Processing Strategy
Use `IntelligentTextChunker` from `app/core/text_chunker.py` for model-specific text chunking:
- Translation: Larger chunks with overlap for context
- Classification: Medium chunks optimized for classification
- Summarization: Large chunks to preserve narrative
- NER: Smaller chunks since entities are typically local

### Resource Management
The system uses `resource_manager.py` for:
- GPU memory monitoring and cleanup
- Request queuing with concurrency limits
- System resource tracking
- Automatic model cleanup on memory pressure

### Error Handling
- Models have fallback strategies for GPU memory issues
- Queue monitoring with alerts for high load
- Health checks for all system components
- Comprehensive logging in `./logs/`

## Production Deployment

### GPU Configuration
- Requires NVIDIA Container Runtime
- Configure `MAX_CONCURRENT_GPU_REQUESTS=1` for single GPU
- Memory limits: 16GB+ GPU VRAM, 32GB+ system RAM

### Scaling
```bash
# Scale Celery workers
docker-compose up --scale celery-worker=3

# Multiple API instances
docker-compose up --scale ai-pipeline=2
```

### Monitoring
- Flower UI at `http://localhost:5555` for Celery monitoring
- Built-in Prometheus metrics at `/metrics`
- Health endpoints for system status
- Real-time queue monitoring

## Important Notes

- Models are loaded from `./models/` directory with specific subdirectories
- Audio files processed in `./temp/` with automatic cleanup
- Redis used for both Celery broker and application state
- All API endpoints return structured JSON with timing and metadata
- Text chunking is model-aware and preserves semantic boundaries
- Real-time processing supports incremental results via WebSocket