# app/realtime/aii.py - FIXED WHISPER PARAMETERS
import torch
import numpy as np
import io
import tempfile
import os

def load_model():
    """Load Whisper model"""
    try:
        import whisper
        print("🔄 Loading Whisper large-v3-turbo model...")
        model = whisper.load_model("large-v3-turbo")
        print("✅ Loaded Whisper large-v3-turbo model successfully")
        return model, None, {}, {}
    except ImportError as e:
        print(f"❌ Whisper import error: {e}")
        return None, None, {}, {}
    except Exception as e:
        print(f"❌ Whisper model loading error: {e}")
        return None, None, {}, {}

def transcribe(model, tokenizer, transcribe_options, decode_options, audio_bytes, mime_type="audio/webm"):
    """Fixed WebM transcription with correct parameters"""
    if model is None:
        return "Model not loaded"
    
    try:
        print(f"📊 Processing {len(audio_bytes)} bytes of {mime_type}")
        
        # Use temporary file approach
        print("🔄 Using temporary file method for WebM...")
        
        # Determine file extension
        if "webm" in mime_type.lower():
            file_extension = ".webm"
        elif "wav" in mime_type.lower():
            file_extension = ".wav"
        elif "mp4" in mime_type.lower():
            file_extension = ".mp4"
        else:
            file_extension = ".webm"  # Default
        
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            print(f"📁 Wrote {len(audio_bytes)} bytes to {tmp_path}")
            
            # Use simplified Whisper parameters (no patience without beam_size)
            result = model.transcribe(
                tmp_path,
                language="sw",  # Changed to Swahili
                temperature=0.0,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                condition_on_previous_text=False
                # Removed: patience=1.0  <- This was causing the error
            )
            
            text = result["text"].strip()
            print(f"📝 File-based transcription: '{text}'")
            
            return text if text else "No speech detected"
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return f"Error: {str(e)}"