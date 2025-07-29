# app/realtime/asterisk_server.py - EXACT CLONE OF ORIGINAL
import socket
import threading
import time
import numpy as np

# Import your models
def start_asterisk_server(host='0.0.0.0', port=8300):
    """Start raw socket server exactly like original"""
    
    # Load models once
    from .aii import load_model, transcribe
    model, tokenizer, transcribe_options, decode_options = load_model()
    print("✅ Asterisk server: Models loaded")
    
    def handle_client(conn, addr):
        print(f"[Asterisk] Connection from {addr}")
        buffer = [bytearray(), bytearray()]
        b = 1
        offset = 0
        N_SAMPLES_BYTES = 480000  # 30 seconds
        
        try:
            while True:
                data = conn.recv(640)  # 20ms SLIN - EXACT SAME AS ORIGINAL
                
                if not data:
                    print(f"[Asterisk] Connection closed by {addr}")
                    break

                if b == 1:
                    buffer[1].extend(data)
                    bn = len(buffer[1]) 
                    if (bn - offset) >= 160000:  # every 5 seconds - EXACT SAME
                        bm = (bn - offset) - 160000
                        if bm > 0:  # truncate excess bytes  
                            data = buffer[1][-bm:]
                            buffer[1][:] = buffer[1][:-bm]

                        ts0 = time.time()
                        
                        # Use original transcribe function with raw bytes
                        out = transcribe_raw_asterisk(model, tokenizer, transcribe_options, decode_options, buffer[1])
                        
                        ts1 = time.time()
                        diff = round(ts1 - ts0, 2)
                        print(f"[Asterisk] {diff:<6} | {bn//32000:<3} {bn} | {out}")

                        # Send result back to Asterisk (if needed)
                        if out and out.strip():
                            try:
                                conn.send(f"{out}\n".encode())
                            except:
                                pass

                        # Sliding window management - EXACT SAME
                        offset += 160000
                        if bn >= N_SAMPLES_BYTES:
                            buffer[1].clear()
                            offset = 0
                        
                        if bm > 0:
                            buffer[1].extend(data)
                continue

                # Handle UID parsing - EXACT SAME
                for index, byte in enumerate(data):
                    if byte == 13:
                        print(f"[Asterisk] uid={buffer[0]}")
                        b = 1
                        continue
                    buffer[b].append(byte)
                    
        except Exception as e:
            print(f"[Asterisk] Error: {e}")
        finally:
            conn.close()

    # Start server - EXACT SAME AS ORIGINAL
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen()
    print(f"[Asterisk] Raw socket server listening on {host}:{port}")

    while True:
        conn, addr = server_sock.accept()
        print(f"[Asterisk] Accepted connection from {addr}")
        p = threading.Thread(target=handle_client, args=(conn, addr)) 
        p.start()

def transcribe_raw_asterisk(model, tokenizer, transcribe_options, decode_options, audio_bytes):
    """Process raw SLIN audio exactly like original"""
    try:
        # Convert raw bytes to audio array - EXACT SAME AS ORIGINAL
        if isinstance(audio_bytes, bytearray):
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes[:-1]
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_array = audio_bytes
            
        # Use Whisper
        result = model.transcribe(
            audio_array,
            language="en", 
            temperature=0.0,
            no_speech_threshold=0.6
        )
        
        return result["text"].strip()
        
    except Exception as e:
        print(f"[Asterisk] Transcription error: {e}")
        return ""