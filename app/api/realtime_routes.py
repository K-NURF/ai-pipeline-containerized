from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import base64
import time
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/realtime", tags=["realtime"])

# Global model loading
model = None
tokenizer = None
transcribe_options = None
decode_options = None
model_loading_attempted = False

def init_realtime_model():
    global model, tokenizer, transcribe_options, decode_options, model_loading_attempted
    
    if model_loading_attempted:
        return
        
    model_loading_attempted = True
    
    try:
        logger.info("🎙️ Loading real-time transcription model...")
        from ..realtime.aii import load_model
        model, tokenizer, transcribe_options, decode_options = load_model()
        logger.info("✅ Real-time model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load real-time model: {e}")
        model = "dummy"
        return False

@router.get("/status")
async def realtime_status():
    """Check if real-time transcription is ready"""
    init_realtime_model()
    return {
        "model_loaded": model is not None,
        "model_type": "whisper" if model != "dummy" else "dummy",
        "status": "ready" if model else "error"
    }

@router.websocket("/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔗 WebSocket connection accepted")
    
    # Initialize model if needed
    success = init_realtime_model()
    
    try:
        await websocket.send_text(json.dumps({
            'type': 'status',
            'message': f'Connected and ready for complete audio segments'
        }))
        
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                
                # Handle complete audio segments (not chunks)
                if data['type'] == 'audio_segment':
                    segment_number = data.get('segment_number', 'unknown')
                    mime_type = data.get('mime_type', 'audio/webm')
                    logger.info(f"📊 Received {mime_type} audio segment #{segment_number}")
                    
                    try:
                        # Decode base64 audio data
                        audio_bytes = base64.b64decode(data['audio'])
                        
                        # Process with MIME type info
                        from ..realtime.aii import transcribe
                        result = transcribe(model, tokenizer, transcribe_options, decode_options, audio_bytes, mime_type)
                        
                        logger.info(f"📝 Segment #{segment_number} result: {result[:50]}...")
                        
                        # Send result back
                        await websocket.send_text(json.dumps({
                            'type': 'transcription',
                            'text': result,
                            'timestamp': time.time(),
                            'segment_number': segment_number
                        }))
                        
                    except Exception as e:
                        logger.error(f"Transcription error for segment #{segment_number}: {e}")
                        await websocket.send_text(json.dumps({
                            'type': 'error',
                            'message': f'Transcription error: {str(e)}',
                            'segment_number': segment_number
                        }))
                
                # Still support the old chunk method for backwards compatibility        
                elif data['type'] == 'audio_chunk':
                    await websocket.send_text(json.dumps({
                        'type': 'error',
                        'message': 'Please use audio_segment instead of audio_chunk for better results'
                    }))
                        
            except json.JSONDecodeError as e:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': f'JSON error: {str(e)}'
                }))
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket client disconnected")