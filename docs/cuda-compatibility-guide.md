# 🖥️ CUDA/cuDNN Compatibility Guide

This guide explains how the AI pipeline handles CUDA/cuDNN compatibility issues and provides troubleshooting steps for optimal performance.

## 🚀 Intelligent Device Selection

The system automatically detects and selects the best available configuration:

### **Automatic Configuration Process**

1. **🔍 CUDA Detection**: Check if CUDA is available
2. **🧪 Compatibility Test**: Test CTranslate2 with small model
3. **⚡ Performance Selection**: Choose optimal device/precision
4. **🔄 Fallback Strategy**: Graceful degradation if issues occur

### **Configuration Hierarchy**

| Priority | Device | Precision | Performance | Use Case |
|----------|--------|-----------|-------------|----------|
| **1st** | CUDA | float16 | 20-50x real-time | GPU with full cuDNN |
| **2nd** | CUDA | int8 | 15-30x real-time | GPU with memory limits |
| **3rd** | CPU | int8 | 3-8x real-time | No GPU or CUDA issues |
| **4th** | CPU | float32 | 2-5x real-time | Maximum compatibility |

## 🛠️ How It Works

### **Smart Compatibility Testing**

```python
# The system performs these tests automatically:

1. Basic CUDA availability check
2. CTranslate2 GPU compatibility test with tiny model
3. Actual inference test with dummy audio
4. Memory and library compatibility verification
```

### **Error Detection & Handling**

**CUDA/cuDNN Errors Detected:**
- `libcudnn_ops.so` missing or incompatible
- `Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor`
- CUDA runtime errors
- CuBLAS library issues

**Memory Errors Handled:**
- GPU out of memory
- Insufficient VRAM for model size
- Memory fragmentation issues

**Fallback Triggers:**
- Library incompatibility → CPU mode
- Memory shortage → Lower precision or CPU
- Driver issues → CPU mode with warnings

## 📊 Performance Expectations

### **Real-World Performance**

| Hardware | Auto-Selected Config | Speed | Memory |
|----------|---------------------|-------|---------|
| RTX 4090 | CUDA/float16 | 40-60x real-time | ~6GB VRAM |
| RTX 3060 Ti | CUDA/float16 | 20-35x real-time | ~6GB VRAM |
| GTX 1660 | CUDA/int8 | 15-25x real-time | ~4GB VRAM |
| CPU (16-core) | CPU/int8 | 5-8x real-time | ~4GB RAM |
| CPU (8-core) | CPU/int8 | 3-5x real-time | ~4GB RAM |

### **Compatibility Matrix**

| System | CUDA | cuDNN | Result |
|--------|------|-------|--------|
| Modern GPU + Drivers | ✅ | ✅ | GPU/float16 (fastest) |
| Older GPU | ✅ | ❌ | CPU/int8 (fallback) |
| No GPU | ❌ | N/A | CPU/int8 (stable) |
| Driver Issues | ⚠️ | ⚠️ | CPU/int8 (safe) |

## 🔧 Manual Configuration

### **Override Automatic Selection**

If you need to force specific settings:

```python
# In app/models/whisper_model.py, modify _check_cuda_compatibility():

def _check_cuda_compatibility(self) -> tuple[str, str]:
    # Force specific configuration
    return "cpu", "int8"  # Always use CPU
    # return "cuda", "float16"  # Force GPU (may fail)
    # return "cuda", "int8"  # Force GPU with int8
```

### **Environment Variables**

```bash
# Force CPU mode
export FORCE_CPU_INFERENCE=1

# Specify CUDA device
export CUDA_VISIBLE_DEVICES=0

# Optimize CPU threads
export OMP_NUM_THREADS=8
```

## 🚨 Troubleshooting Common Issues

### **Issue: "Unable to load libcudnn_ops.so"**

**What it means:** CuDNN library missing or incompatible

**Solution:**
```bash
# Install compatible cuDNN
pip install nvidia-cudnn-cu12  # For CUDA 12.x
# or
pip install nvidia-cudnn-cu11  # For CUDA 11.x

# Verify installation
python -c "import torch; print(torch.backends.cudnn.version())"
```

**Auto-handled:** ✅ System automatically falls back to CPU

### **Issue: "CUDA out of memory"**

**What it means:** GPU doesn't have enough VRAM

**Solution:**
```bash
# Reduce other GPU usage
nvidia-smi  # Check what's using GPU

# The system automatically tries:
# 1. CUDA with int8 (lower memory)
# 2. CPU mode if still failing
```

**Auto-handled:** ✅ System tries int8 then CPU fallback

### **Issue: "Invalid handle" or "symbol not found"**

**What it means:** CUDA/cuDNN version mismatch

**Solution:**
```bash
# Check versions
nvcc --version
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"

# Reinstall with matching versions
pip uninstall ctranslate2 faster-whisper
pip install ctranslate2-cuda12 faster-whisper  # Match your CUDA version
```

**Auto-handled:** ✅ System falls back to CPU mode

### **Issue: Model loading very slow**

**Possible causes:**
1. Downloading model for first time (expected)
2. Compatibility testing with multiple configurations
3. CPU mode selected (still faster than old transformers)

**Check logs for:**
```
🧪 Testing GPU compatibility with small model...
✅ CUDA/cuDNN compatibility test passed!
```

## 📈 Performance Monitoring

### **Check Current Configuration**

```python
python -c "
from app.models.whisper_model import WhisperModel
model = WhisperModel()
model.load()
info = model.get_model_info()
print(f'Device: {info[\"device\"]}')
print(f'Compute Type: {info[\"compute_type\"]}')
print(f'Framework: {info[\"framework\"]}')
"
```

### **Benchmark Performance**

```python
python -c "
import time
import numpy as np
from app.models.whisper_model import WhisperModel

model = WhisperModel()
model.load()

# Test with 5-second audio
audio = np.zeros(16000 * 5, dtype=np.int16).tobytes()
start = time.time()
result = model.transcribe_pcm_audio(audio, 16000, 'en')
duration = time.time() - start

print(f'Configuration: {model.device}/{model.compute_type}')
print(f'Processing time: {duration:.3f}s for 5s audio')
print(f'Speed ratio: {5/duration:.1f}x real-time')
print(f'Expected for your hardware: Excellent!' if 5/duration > 10 else 'Good performance')
"
```

### **Resource Usage**

```bash
# Monitor during processing
nvidia-smi -l 1  # GPU usage
htop             # CPU/RAM usage

# Check VRAM usage
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
else:
    print('CPU mode - no GPU memory used')
"
```

## ✅ Validation

### **System Health Check**

```bash
python -c "
from app.models.whisper_model import WhisperModel
import logging
logging.basicConfig(level=logging.INFO)

print('🔍 Running compatibility validation...')
model = WhisperModel()
success = model.load()

if success:
    info = model.get_model_info()
    print(f'✅ Success: {info[\"device\"]}/{info[\"compute_type\"]}')
    print(f'📊 Performance: {info[\"performance\"]}')
    print(f'🚀 Ready for production!')
else:
    print(f'❌ Failed: {model.error}')
"
```

### **Expected Log Output**

**Successful GPU Setup:**
```
🚀 Loading Faster-Whisper model...
🔍 Testing CUDA/cuDNN compatibility with CTranslate2...
🧪 Testing GPU compatibility with small model...
✅ CUDA/cuDNN compatibility test passed!
⚡ Expected performance: 20-50x real-time (GPU/FP16)
✅ Model loaded successfully with optimal configuration
🚀 Model test successful - ready for inference
✅ Faster-Whisper model loaded successfully on cuda with float16
```

**Successful CPU Fallback:**
```
🚀 Loading Faster-Whisper model...
🔍 Testing CUDA/cuDNN compatibility with CTranslate2...
🧪 Testing GPU compatibility with small model...
⚠️ CUDA/cuDNN compatibility issue detected: Unable to load libcudnn_ops.so...
🔄 Falling back to CPU mode for stability
⚡ Expected performance: 3-8x real-time (CPU/INT8)
✅ Model loaded successfully with CPU fallback
🚀 Model test successful - ready for inference
✅ Faster-Whisper model loaded successfully on cpu with int8
```

## 🎯 Best Practices

### **For Developers**
1. **Trust the auto-selection** - the system chooses optimal settings
2. **Monitor the logs** - they show exactly what's happening
3. **Test after updates** - CUDA/driver updates can affect compatibility
4. **Use the validation script** - verify setup after changes

### **For Production**
1. **Pin dependency versions** - ensure consistent behavior
2. **Use Docker** - containerize for consistent environments
3. **Monitor performance** - set up alerts for degraded performance
4. **Have CPU fallback ready** - always works as backup

### **For Users with GPU Issues**
1. **Don't panic** - CPU mode still provides good performance
2. **Update drivers** - often fixes compatibility issues
3. **Check system requirements** - ensure adequate VRAM
4. **Contact support** - with log output for specific help

## 🔮 Future Improvements

The compatibility system will continue to evolve:

- **More granular GPU detection**
- **Automatic driver optimization suggestions**  
- **Performance tuning recommendations**
- **Better error messages with solutions**

---

**The goal is simple: Maximum performance with zero configuration hassle!** 🚀

The system automatically handles all the complexity so you can focus on your application, not GPU driver compatibility.