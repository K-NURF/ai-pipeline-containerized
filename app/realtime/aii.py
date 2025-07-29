# app/realtime/aii.py - ADD ASTERISK SUPPORT
import torch
import numpy as np
import io
import tempfile
import os

def load_model():
    """Load Whisper model"""
    try:
        import whisper
        model = whisper.load_model("small")
        print("✅ Loaded Whisper small model")
        return model, None, {}, {}
    except ImportError:
        return None, None, {}, {}

def transcribe(model, tokenizer, transcribe_options, decode_options, audio_bytes, mime_type="audio/webm"):
    """Browser WebM transcription (existing function)"""
    if model is None:
        return "Model not loaded"
    
    try:
        print(f"📊 Processing {len(audio_bytes)} bytes of {mime_type}")
        
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            result = model.transcribe(
                tmp_path,
                language="en",
                temperature=0.0,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                condition_on_previous_text=False
            )
            
            text = result["text"].strip()
            print(f"📝 File-based transcription: '{text}'")
            return text if text else "No speech detected"
            
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return f"Error: {str(e)}"

def transcribe_asterisk_audio(model, tokenizer, transcribe_options, decode_options, audio_bytes):
    """Asterisk raw PCM audio transcription (NEW FUNCTION)"""
    if model is None:
        return "Model not loaded"
    
    try:
        print(f"📞 Processing Asterisk PCM audio: {len(audio_bytes)} bytes")
        
        # Convert raw PCM bytes to numpy array (like original implementation)
        if isinstance(audio_bytes, bytearray):
            # Ensure buffer size is compatible with int16
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes[:-1]
                
            # Convert SLIN (signed linear) to float32
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_array = audio_bytes
            
        # Calculate duration
        duration = len(audio_array) / 16000  # Assuming 16kHz sample rate
        print(f"🎵 Asterisk audio duration: {duration:.2f} seconds")
        
        if duration < 0.5:
            return ""  # Too short, return empty (Asterisk mode)
        
        # Normalize audio levels
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array)) * 0.9
            
        # Use Whisper with Asterisk-optimized settings
        result = model.transcribe(
            audio_array, 
            language="en",
            temperature=0.0,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            condition_on_previous_text=False,
            beam_size=1,  # Faster for real-time
            best_of=1     # Faster for real-time
        )
        
        text = result["text"].strip()
        print(f"📝 Asterisk transcription: '{text}'")
        
        return text if text else ""
        
    except Exception as e:
        print(f"❌ Asterisk transcription error: {e}")
        return f"Error: {str(e)}"