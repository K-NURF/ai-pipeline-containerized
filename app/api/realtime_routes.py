from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import base64
import time
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/realtime", tags=["realtime"])

@router.get("/status")
async def realtime_status():
    """Check if real-time transcription is ready"""
    # Check if Asterisk server model is loaded
    try:
        from app.realtime.asterisk_server import get_model_status
        asterisk_ready = get_model_status()
        
        # Also check pipeline models if available
        from app.models.model_loader import model_loader
        pipeline_ready = model_loader.models.get("whisper") is not None
        
        return {
            "asterisk_model_loaded": asterisk_ready,
            "pipeline_model_loaded": pipeline_ready,
            "status": "ready" if asterisk_ready else "loading",
            "recommended_source": "asterisk_model" if asterisk_ready else "waiting"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

@router.websocket("/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """Browser-compatible WebSocket endpoint using Asterisk server's model"""
    await websocket.accept()
    logger.info("🔗 WebSocket connection accepted (Browser mode)")
    
    # Try to use Asterisk server's model first
    try:
        from app.realtime.asterisk_server import get_shared_model
        model, tokenizer, transcribe_options, decode_options = get_shared_model()
        
        if model is None:
            await websocket.send_text(json.dumps({
                'type': 'error',
                'message': 'No model available. Asterisk server not ready.'
            }))
            return
            
        logger.info("✅ Using Asterisk server's Whisper model for browser transcription")
        
    except Exception as e:
        logger.error(f"Failed to get Asterisk model: {e}")
        await websocket.send_text(json.dumps({
            'type': 'error',
            'message': f'Model loading error: {str(e)}'
        }))
        return
    
    try:
        await websocket.send_text(json.dumps({
            'type': 'status',
            'message': f'Connected - Using Asterisk server model'
        }))
        
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                
                if data['type'] == 'audio_segment':
                    segment_number = data.get('segment_number', 'unknown')
                    mime_type = data.get('mime_type', 'audio/webm')
                    logger.info(f"📊 Processing segment #{segment_number} with Asterisk model")
                    
                    try:
                        audio_bytes = base64.b64decode(data['audio'])
                        
                        # Use Asterisk server's transcription function
                        result = await asyncio.to_thread(
                            process_browser_audio_with_asterisk_model,
                            model,
                            audio_bytes,
                            mime_type
                        )
                        
                        logger.info(f"📝 Segment #{segment_number}: {result[:50]}...")
                        
                        await websocket.send_text(json.dumps({
                            'type': 'transcription',
                            'text': result,
                            'timestamp': time.time(),
                            'segment_number': segment_number
                        }))
                        
                    except Exception as e:
                        logger.error(f"Transcription error: {e}")
                        await websocket.send_text(json.dumps({
                            'type': 'error',
                            'message': f'Transcription error: {str(e)}',
                            'segment_number': segment_number
                        }))
                        
            except json.JSONDecodeError as e:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': f'JSON error: {str(e)}'
                }))
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket client disconnected (Browser mode)")

@router.websocket("/asterisk")
async def websocket_asterisk(websocket: WebSocket):
    """Asterisk-compatible WebSocket endpoint using pipeline models"""
    await websocket.accept()
    logger.info("🔗 WebSocket connection accepted (Asterisk mode)")
    
    # Use the pipeline's model loader
    from app.models.model_loader import model_loader
    
    # Wait for models to be loaded
    max_wait = 30
    wait_time = 0
    while not model_loader.models.get("whisper") and wait_time < max_wait:
        await asyncio.sleep(1)
        wait_time += 1
    
    whisper_model = model_loader.models.get("whisper")
    if not whisper_model:
        logger.error("❌ Whisper model not available for Asterisk")
        await websocket.close()
        return
    
    logger.info("✅ Asterisk using pipeline's Whisper model")
    
    buffer = [bytearray(), bytearray()]  # [uid_buffer, audio_buffer]
    b = 0  # 0 = reading UID, 1 = reading audio
    offset = 0
    
    try:
        while True:
            try:
                # Try to receive binary data (Asterisk sends raw bytes)
                data = await websocket.receive_bytes()
                
                logger.debug(f"📦 Received {len(data)} raw bytes from Asterisk")
                
                # Process raw audio data (like original implementation)
                for index, byte in enumerate(data):
                    if byte == 13:  # Carriage return - end of UID
                        logger.info(f"📞 Asterisk client UID: {buffer[0].decode()}")
                        b = 1  # Switch to audio mode
                        buffer[1].clear()  # Reset audio buffer
                        offset = 0
                        continue
                    
                    buffer[b].append(byte)
                
                # Process audio when we have enough (every 5 seconds = 160000 bytes)
                if b == 1 and len(buffer[1]) >= 160000:
                    bn = len(buffer[1])
                    
                    # Handle excess bytes for next iteration
                    bm = (bn - offset) - 160000
                    if bm > 0:
                        excess_data = buffer[1][-bm:]
                        buffer[1] = buffer[1][:-bm]
                    
                    try:
                        logger.info(f"🎙️ Processing Asterisk audio: {len(buffer[1])} bytes")
                        
                        # Use pipeline model with resource management
                        from app.core.resource_manager import resource_manager
                        
                        request_id = f"asterisk_{time.time()}"
                        if await resource_manager.acquire_gpu(request_id):
                            try:
                                result = await asyncio.to_thread(
                                    process_asterisk_audio,
                                    whisper_model,
                                    buffer[1]
                                )
                            finally:
                                resource_manager.release_gpu(request_id)
                        else:
                            result = "GPU busy"
                        
                        logger.info(f"📝 Asterisk transcription: {result}")
                        
                        # Send result back to Asterisk (simple text, not JSON)
                        if result and result.strip():
                            await websocket.send_text(result)
                        
                    except Exception as e:
                        logger.error(f"❌ Asterisk transcription error: {e}")
                        await websocket.send_text(f"ERROR: {str(e)}")
                    
                    # Manage sliding window
                    offset += 160000
                    N_SAMPLES_BYTES = 480000  # 30 seconds
                    if bn >= N_SAMPLES_BYTES:
                        buffer[1].clear()
                        offset = 0
                    
                    # Add back excess bytes
                    if bm > 0:
                        buffer[1].extend(excess_data)
                        
            except Exception as inner_e:
                logger.error(f"❌ Error processing Asterisk data: {inner_e}")
                break
                
    except WebSocketDisconnect:
        logger.info("🔌 Asterisk client disconnected")
    except Exception as e:
        logger.error(f"❌ Asterisk WebSocket error: {e}")

def process_pipeline_audio(whisper_model, audio_bytes, mime_type="audio/webm"):
    """Process browser audio using the pipeline's Whisper model"""
    import tempfile
    import os
    
    try:
        logger.info(f"📊 Processing {len(audio_bytes)} bytes of {mime_type} with pipeline model")
        
        # Save to temp file (same approach as working browser version)
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Check what methods the pipeline model has
            if hasattr(whisper_model, 'transcribe_audio_file'):
                # Use the pipeline's file method
                result = whisper_model.transcribe_audio_file(tmp_path, language="en")
            elif hasattr(whisper_model, 'transcribe_audio_bytes'):
                # Use the pipeline's bytes method
                result = whisper_model.transcribe_audio_bytes(audio_bytes, language="en")
            elif hasattr(whisper_model, 'transcribe'):
                # Use raw Whisper API
                result = whisper_model.transcribe(tmp_path, language="en", temperature=0.0)
                result = result.get("text", "").strip()
            else:
                # Fallback
                result = "Pipeline model interface unknown"
            
            logger.info(f"📝 Pipeline transcription: '{result}'")
            return result if result else "No speech detected"
            
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"❌ Pipeline audio processing error: {e}")
        return f"Error: {str(e)}"

def process_asterisk_audio(whisper_model, audio_bytes):
    """Process Asterisk raw PCM audio using pipeline model"""
    import numpy as np
    
    try:
        logger.info(f"📞 Processing Asterisk PCM audio: {len(audio_bytes)} bytes with pipeline model")
        
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
        logger.info(f"🎵 Asterisk audio duration: {duration:.2f} seconds")
        
        if duration < 0.5:
            return ""  # Too short, return empty (Asterisk mode)
        
        # Normalize audio levels
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array)) * 0.9
            
        # Use pipeline model
        if hasattr(whisper_model, 'transcribe_audio_array'):
            result = whisper_model.transcribe_audio_array(audio_array, language="en")
        elif hasattr(whisper_model, 'transcribe'):
            # Use raw Whisper API with numpy array
            result = whisper_model.transcribe(audio_array, language="en", temperature=0.0)
            result = result.get("text", "").strip()
        else:
            result = "Pipeline model incompatible with raw audio"
        
        logger.info(f"📝 Asterisk pipeline transcription: '{result}'")
        return result if result else ""
        
    except Exception as e:
        logger.error(f"❌ Asterisk pipeline audio error: {e}")
        return f"Error: {str(e)}"

def process_browser_audio_with_asterisk_model(model, audio_bytes, mime_type="audio/webm"):
    """Process browser audio using the Asterisk server's model"""
    import tempfile
    import os
    
    try:
        logger.info(f"📊 Processing {len(audio_bytes)} bytes of {mime_type} with Asterisk model")
        
        # Save to temp file (same approach as working browser version)
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Use the same transcription approach as working before
            result = model.transcribe(
                tmp_path,
                language="en",
                temperature=0.0,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                condition_on_previous_text=False
            )
            
            text = result.get("text", "").strip()
            logger.info(f"📝 Asterisk model transcription: '{text}'")
            return text if text else "No speech detected"
            
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"❌ Asterisk model processing error: {e}")
        return f"Error: {str(e)}"