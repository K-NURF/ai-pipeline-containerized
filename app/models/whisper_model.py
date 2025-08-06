import torch
import logging
import librosa
import tempfile
import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Generator, Tuple

logger = logging.getLogger(__name__)

class WhisperModel:
    """Faster-Whisper implementation for high-performance speech recognition"""
    
    def __init__(self, model_path: str = None):
        from ..config.settings import settings
        
        self.model_path = model_path or settings.get_model_path("whisper")
        self.fallback_model_id = "openai/whisper-large-v3-turbo"
        self.model = None
        self.device = None
        self.compute_type = None
        self.is_loaded = False
        self.error = None
        
        # Supported language codes for Whisper
        self.supported_languages = {
            "auto": "Auto-detect",
            "en": "English",
            "es": "Spanish", 
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic",
            "hi": "Hindi",
            "sw": "Swahili",
            "am": "Amharic",
            "lg": "Luganda",
            "rw": "Kinyarwanda",
            "so": "Somali",
            "yo": "Yoruba",
            "ig": "Igbo",
            "ha": "Hausa",
            "zu": "Zulu",
            "xh": "Xhosa",
            "af": "Afrikaans",
            "ny": "Chichewa"
        }
        
    def _check_local_model_exists(self) -> bool:
        """Check if local model files exist"""
        if not os.path.exists(self.model_path):
            return False
        
        # Check for essential Whisper model files
        required_files = ["config.json"]
        optional_files = ["model.safetensors", "pytorch_model.bin"]
        
        # Config must exist
        config_path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config_path):
            return False
        
        # At least one model file must exist
        model_file_exists = any(
            os.path.exists(os.path.join(self.model_path, file)) 
            for file in optional_files
        )
        
        if not model_file_exists:
            logger.warning(f"Config found but no model files in {self.model_path}")
            return False
        
        logger.info(f"✅ Local Whisper model files detected in {self.model_path}")
        return True
        
    def _check_cuda_compatibility(self) -> tuple[str, str]:
        """Advanced CUDA/cuDNN compatibility with cuDNN 8.x library detection"""
        
        # Check CTranslate2 version first
        try:
            import ctranslate2
            ct2_version = ctranslate2.__version__
            logger.info(f"🔍 CTranslate2 version: {ct2_version}")
            
            # Version 4.4.0 requires cuDNN 8, check if available
            if ct2_version.startswith("4.4."):
                logger.info("🔍 CTranslate2 4.4.x requires cuDNN 8 libraries")
                logger.info("🔍 Checking for cuDNN 8.x library availability...")
            else:
                logger.warning(f"⚠️ CTranslate2 {ct2_version} may have compatibility issues")
        except Exception as e:
            logger.warning(f"⚠️ Could not check CTranslate2 version: {e}")
        
        # Check if CUDA is available at all
        if not torch.cuda.is_available():
            logger.info("🖥️ CUDA not available, using CPU")
            return "cpu", "int8"
        
        # Check for known problematic conditions
        logger.info("🔍 Checking CUDA/cuDNN environment...")
        
        # Basic CUDA device check
        try:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🖥️ GPU detected: {gpu_name} ({vram_gb:.1f}GB VRAM)")
        except Exception as e:
            logger.warning(f"⚠️ GPU detection failed: {e}")
            return "cpu", "int8"
        
        # Check if we have enough VRAM for the model
        if vram_gb < 4:
            logger.warning(f"⚠️ Insufficient VRAM ({vram_gb:.1f}GB < 4GB required)")
            return "cpu", "int8"
        
        # Advanced cuDNN library detection
        cudnn_8_available = self._check_cudnn_8_libraries()
        
        # Check PyTorch cuDNN availability
        try:
            cudnn_available = torch.backends.cudnn.is_available()
            if not cudnn_available:
                logger.warning("⚠️ PyTorch reports cuDNN not available")
                return "cpu", "int8"
            
            cudnn_version = torch.backends.cudnn.version()
            logger.info(f"🔍 PyTorch cuDNN version: {cudnn_version}")
            
            # Check if we have cuDNN 8.x libraries available
            if cudnn_8_available:
                logger.info("✅ cuDNN 8.x libraries detected - GPU acceleration enabled!")
                logger.info("🚀 CTranslate2 will use cuDNN 8.x for optimal performance")
            else:
                logger.warning("⚠️ cuDNN 8.x libraries not found for CTranslate2")
                logger.info("💡 Install nvidia-cudnn-cu12==8.9.7.29 for GPU acceleration")
                logger.info("🔄 Falling back to CPU mode")
                return "cpu", "int8"
                
        except Exception as e:
            logger.warning(f"⚠️ cuDNN check failed: {e}")
            return "cpu", "int8"
        
        # GPU is available with proper cuDNN 8.x libraries
        # Choose precision based on VRAM
        if vram_gb >= 8:
            logger.info("⚡ High VRAM available - selecting float16 precision")
            logger.info("⚡ Expected performance: 20-50x real-time (GPU/FP16)")
            return "cuda", "float16"
        elif vram_gb >= 6:
            logger.info("⚡ Moderate VRAM available - selecting int8 precision")
            logger.info("⚡ Expected performance: 15-30x real-time (GPU/INT8)")
            return "cuda", "int8"
        else:
            logger.info("⚡ Limited VRAM - using CPU for safety")
            logger.info("⚡ Expected performance: 3-8x real-time (CPU/INT8)")
            return "cpu", "int8"

    def _check_cudnn_8_libraries(self) -> bool:
        """Check if cuDNN 8.x libraries are available for CTranslate2"""
        import os
        import site
        
        try:
            # Check for nvidia-cudnn-cu12 8.x installation
            site_packages = site.getsitepackages()
            for site_pkg in site_packages:
                # Look for nvidia cudnn 8.x package
                cudnn_8_paths = [
                    os.path.join(site_pkg, "nvidia", "cudnn", "lib"),
                    os.path.join(site_pkg, "nvidia_cudnn_cu12.libs"),
                ]
                
                for cudnn_path in cudnn_8_paths:
                    if os.path.exists(cudnn_path):
                        # Look for cuDNN 8.x specific libraries
                        cudnn_8_libs = [
                            "libcudnn_ops_infer.so.8",
                            "libcudnn_cnn_infer.so.8", 
                            "libcudnn.so.8"
                        ]
                        
                        found_libs = []
                        for lib_file in os.listdir(cudnn_path):
                            for required_lib in cudnn_8_libs:
                                if required_lib in lib_file:
                                    found_libs.append(lib_file)
                        
                        if found_libs:
                            logger.info(f"🔍 Found cuDNN 8.x libraries in: {cudnn_path}")
                            logger.info(f"📚 Libraries: {', '.join(found_libs)}")
                            
                            # Set environment variable for CTranslate2 to find these libraries
                            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
                            if cudnn_path not in current_ld_path:
                                new_ld_path = f"{cudnn_path}:{current_ld_path}" if current_ld_path else cudnn_path
                                os.environ["LD_LIBRARY_PATH"] = new_ld_path
                                logger.info(f"🔧 Updated LD_LIBRARY_PATH to include cuDNN 8.x libraries")
                            
                            return True
            
            logger.info("🔍 cuDNN 8.x libraries not found in standard locations")
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Error checking cuDNN 8.x libraries: {e}")
            return False

    def _load_model_with_fallback(self):
        """Load model with comprehensive fallback strategy"""
        from faster_whisper import WhisperModel as FasterWhisperModel
        
        attempts = [
            # Attempt 1: Use determined device/compute_type
            (self.device, self.compute_type, "optimal configuration"),
            # Attempt 2: If CUDA failed, try CPU
            ("cpu", "int8", "CPU fallback"),
            # Attempt 3: Last resort with most compatible settings
            ("cpu", "float32", "maximum compatibility")
        ]
        
        for device, compute_type, description in attempts:
            try:
                logger.info(f"🔄 Attempting {description}: device={device}, compute_type={compute_type}")
                
                # Try local model first
                if self._check_local_model_exists() and device == self.device and compute_type == self.compute_type:
                    try:
                        logger.info(f"🚀 Loading local model from {self.model_path}")
                        model = FasterWhisperModel(
                            model_size_or_path=self.model_path,
                            device=device,
                            compute_type=compute_type,
                            local_files_only=True,
                            cpu_threads=4
                        )
                        
                        # Update our settings if fallback was used
                        self.device = device
                        self.compute_type = compute_type
                        
                        logger.info(f"✅ Local model loaded with {description}")
                        return model
                        
                    except Exception as local_error:
                        logger.warning(f"⚠️ Local model failed: {local_error}")
                        # Continue to try download
                
                # Try downloading from HuggingFace
                logger.info(f"🌐 Downloading model: {self.fallback_model_id}")
                
                # Extract model size
                if "large-v3-turbo" in self.fallback_model_id:
                    model_size = "large-v3-turbo"
                elif "large-v3" in self.fallback_model_id:
                    model_size = "large-v3"
                elif "large-v2" in self.fallback_model_id:
                    model_size = "large-v2"
                else:
                    model_size = "large-v3-turbo"
                
                model = FasterWhisperModel(
                    model_size_or_path=model_size,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=4
                )
                
                # Update our settings if fallback was used
                self.device = device
                self.compute_type = compute_type
                
                logger.info(f"✅ Model loaded successfully with {description}")
                return model
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Log specific error types
                if 'cudnn' in error_msg or 'invalid handle' in error_msg:
                    logger.warning(f"⚠️ CUDA/cuDNN error with {description}: {e}")
                elif 'out of memory' in error_msg:
                    logger.warning(f"⚠️ Memory error with {description}: {e}")
                else:
                    logger.warning(f"⚠️ Failed {description}: {e}")
                
                # Continue to next attempt
                continue
        
        # If all attempts failed
        raise RuntimeError("Failed to load Faster-Whisper model with any configuration")

    def load(self) -> bool:
        """Load Whisper model with faster-whisper backend and intelligent device selection"""
        try:
            logger.info(f"🚀 Loading Faster-Whisper model...")
            
            from faster_whisper import WhisperModel as FasterWhisperModel
            
            # Intelligent device and compute type selection
            self.device, self.compute_type = self._check_cuda_compatibility()
            
            logger.info(f"🚀 Selected device: {self.device}, compute type: {self.compute_type}")
            
            # Performance expectations
            if self.device == "cuda":
                if self.compute_type == "float16":
                    logger.info("⚡ Expected performance: 20-50x real-time (GPU/FP16)")
                else:
                    logger.info("⚡ Expected performance: 15-30x real-time (GPU/INT8)")
            else:
                logger.info("⚡ Expected performance: 3-8x real-time (CPU/INT8)")
            
            # Load the model using the determined device/compute_type
            self.model = self._load_model_with_fallback()
            
            # Test the model with a small dummy transcription
            # This helps catch cuDNN issues early with proper error handling
            try:
                logger.info("🧪 Testing model with dummy audio...")
                # Create a small dummy audio array (1 second of silence at 16kHz)
                dummy_audio = np.zeros(16000, dtype=np.float32)
                segments, _ = self.model.transcribe(dummy_audio, language="en", beam_size=1)
                # Consume the generator to test it works
                list(segments)
                logger.info("✅ Model test successful - GPU inference working")
            except Exception as e:
                error_msg = str(e).lower()
                if 'cudnn' in error_msg or 'invalid handle' in error_msg or 'aborted' in error_msg:
                    logger.error(f"❌ cuDNN error during test: {e}")
                    logger.info("🔄 Falling back to CPU mode due to GPU test failure...")
                    
                    # Reload model in CPU mode
                    try:
                        self.device = "cpu"
                        self.compute_type = "int8"
                        self.model = self._load_model_with_fallback()
                        logger.info("✅ Successfully reloaded model in CPU mode")
                    except Exception as fallback_error:
                        logger.error(f"❌ CPU fallback also failed: {fallback_error}")
                        raise RuntimeError(f"Both GPU and CPU model loading failed: {fallback_error}")
                else:
                    logger.warning(f"⚠️ Model test failed but continuing: {e}")
            
            self.is_loaded = True
            self.error = None
            logger.info(f"✅ Faster-Whisper model loaded successfully on {self.device} with {self.compute_type}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load Faster-Whisper model: {e}"
            logger.error(f"❌ {error_msg}")
            self.error = error_msg
            self.is_loaded = False
            return False
    
    def _validate_language(self, language: Optional[str]) -> Optional[str]:
        """Validate and normalize language code"""
        if not language or language.lower() in ["auto", "none", ""]:
            return None  # Auto-detect
        
        # Normalize language code
        lang_code = language.lower().strip()
        
        # Check if it's a supported language
        if lang_code in self.supported_languages:
            return lang_code
        
        # Try to find by language name
        for code, name in self.supported_languages.items():
            if name.lower() == lang_code:
                return code
        
        # If not found, log warning but continue (Whisper supports many languages)
        logger.warning(f"⚠️ Language '{language}' not in known list, but will attempt transcription")
        return lang_code
    
    def transcribe_audio_file(self, audio_file_path: str, language: Optional[str] = None) -> str:
        """Transcribe audio file to text"""
        if not self.is_loaded:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            logger.info(f"🚀 Transcribing audio file: {Path(audio_file_path).name}")
            
            # Validate and normalize language
            validated_language = self._validate_language(language)
            if validated_language:
                logger.info(f"🚀 Target language: {validated_language} ({self.supported_languages.get(validated_language, 'Unknown')})")
            else:
                logger.info("🚀 Language: Auto-detect")
            
            # Load audio with librosa (handles multiple formats)
            audio_array, sample_rate = librosa.load(audio_file_path, sr=16000, mono=True)
            
            # Calculate audio duration
            duration = len(audio_array) / sample_rate
            logger.info(f"🚀 Audio duration: {duration:.1f} seconds")
            
            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(
                audio_array,
                language=validated_language,
                task="transcribe",  # Explicitly set to transcribe
                beam_size=1,  # Faster inference with beam_size=1
                best_of=1,    # Single candidate for speed
                temperature=0.0,  # Deterministic output
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,  # Faster for short audio
                word_timestamps=False  # Disable for faster processing
            )
            
            logger.info(f"🚀 Detected language: {info.language} (probability: {info.language_probability:.2f})")
            logger.info(f"🚀 Audio duration: {info.duration:.1f}s")
            
            # Collect transcript from segments
            transcript_parts = []
            for segment in segments:
                transcript_parts.append(segment.text.strip())
            
            transcript = " ".join(transcript_parts).strip()
            
            logger.info(f"✅ Transcription completed: {len(transcript)} characters")
            
            return transcript
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def transcribe_audio_bytes(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """Transcribe audio from bytes (for uploaded files)"""
        if not self.is_loaded:
            raise RuntimeError("Whisper model not loaded")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            try:
                temp_file.write(audio_bytes)
                temp_file.flush()
                
                result = self.transcribe_audio_file(temp_file.name, language)
                return result
                
            finally:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
    
    def transcribe_pcm_audio(self, audio_bytes: bytes, sample_rate: int = 16000, language: Optional[str] = None) -> str:
        """Transcribe raw PCM audio bytes (for streaming from TCP) - OPTIMIZED for speed"""
        if not self.is_loaded:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            logger.info(f"🚀 Transcribing PCM audio: {len(audio_bytes)} bytes at {sample_rate}Hz")
            
            # Validate and normalize language
            validated_language = self._validate_language(language)
            
            # Convert raw PCM bytes to numpy array (assuming 16-bit signed integers)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Calculate audio duration
            duration = len(audio_array) / sample_rate
            logger.info(f"🚀 Audio duration: {duration:.1f} seconds")
            
            # OPTIMIZED transcription for real-time streaming
            segments, info = self.model.transcribe(
                audio_array,
                language=validated_language,
                task="transcribe",
                beam_size=1,  # Fastest beam search
                best_of=1,    # Single candidate
                temperature=0.0,  # Deterministic
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,  # Don't condition on previous for speed
                word_timestamps=False,  # Disable timestamps for speed
                prepend_punctuations="\"'([{-",
                append_punctuations="\"'.。,!?:)]}",
                # VAD (Voice Activity Detection) for better silence handling
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Minimum silence duration
                    speech_pad_ms=30  # Padding around speech
                )
            )
            
            # Collect transcript from segments (faster iteration)
            transcript_parts = []
            for segment in segments:
                text = segment.text.strip()
                if text:  # Only add non-empty segments
                    transcript_parts.append(text)
            
            transcript = " ".join(transcript_parts).strip()
            
            logger.info(f"✅ PCM transcription completed: {len(transcript)} characters")
            
            return transcript
            
        except Exception as e:
            logger.error(f"❌ PCM transcription failed: {e}")
            raise RuntimeError(f"PCM transcription failed: {str(e)}")
    
    def transcribe_streaming(self, audio_bytes: bytes, language: Optional[str] = None) -> Generator[Tuple[str, float], None, None]:
        """Generator for streaming transcription with progress updates"""
        if not self.is_loaded:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            # For faster-whisper, we'll simulate streaming by processing the full audio
            # and yielding intermediate results
            validated_language = self._validate_language(language)
            
            # Convert audio bytes to numpy array
            if isinstance(audio_bytes, bytes):
                # Assume this is a WAV file or similar - use librosa to decode
                with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
                    temp_file.write(audio_bytes)
                    temp_file.flush()
                    audio_array, _ = librosa.load(temp_file.name, sr=16000, mono=True)
            else:
                audio_array = audio_bytes  # Assume it's already a numpy array
            
            # Process with word-level timestamps for streaming effect
            segments, info = self.model.transcribe(
                audio_array,
                language=validated_language,
                task="transcribe",
                beam_size=1,
                word_timestamps=True,  # Enable for streaming simulation
                condition_on_previous_text=False
            )
            
            # Simulate streaming by yielding progressive transcript
            full_transcript = ""
            total_duration = info.duration if hasattr(info, 'duration') else len(audio_array) / 16000
            
            for i, segment in enumerate(segments):
                segment_text = segment.text.strip()
                if segment_text:
                    full_transcript += (" " if full_transcript else "") + segment_text
                    
                    # Calculate progress as percentage
                    segment_end = getattr(segment, 'end', (i + 1) * 5.0)  # Estimate if no timestamp
                    progress = min(100.0, (segment_end / total_duration) * 100.0)
                    
                    yield full_transcript.strip(), progress
            
            # Final yield with 100% progress
            if full_transcript:
                yield full_transcript.strip(), 100.0
                
        except Exception as e:
            logger.error(f"❌ Streaming transcription failed: {e}")
            yield "", 0.0  # Return empty result on error
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names"""
        return self.supported_languages.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": "whisper",
            "model_path": self.model_path,
            "fallback_model_id": self.fallback_model_id,
            "model_type": "speech-to-text",
            "framework": "faster-whisper",
            "backend": "CTranslate2",
            "device": str(self.device) if self.device else None,
            "compute_type": str(self.compute_type) if self.compute_type else None,
            "is_loaded": self.is_loaded,
            "error": self.error,
            "supported_formats": ["wav", "mp3", "flac", "m4a", "ogg"],
            "max_audio_length": "unlimited (efficient processing)",
            "sample_rate": "16kHz",
            "task": "transcribe",
            "languages": "multilingual (99+ languages)",
            "version": "large-v3-turbo",
            "long_form_support": True,
            "streaming_optimized": True,
            "quantization": "float16 (GPU) / int8 (CPU)",
            "performance": "4-5x faster than transformers",
            "supported_language_codes": list(self.supported_languages.keys()),
            "local_model_available": self._check_local_model_exists()
        }
    
    def is_ready(self) -> bool:
        """Check if model is ready for inference"""
        return self.is_loaded and self.model is not None

# Global instance following your pattern
whisper_model = WhisperModel()