#!/usr/bin/env python3
"""
Simple HTTPS server to serve the realtime transcription test page.
Microphone access requires HTTPS, so this creates a self-signed certificate server.
"""
import http.server
import ssl
import socketserver
import os
from pathlib import Path

class HTTPSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow cross-origin requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.end_headers()

def main():
    # Server configuration
    PORT = 8443
    CERT_FILE = "cert.pem"
    KEY_FILE = "key.pem"
    
    # Check if certificate files exist
    if not os.path.exists(CERT_FILE) or not os.path.exists(KEY_FILE):
        print("❌ SSL certificate files not found!")
        print("Please run this command first:")
        print('openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"')
        return
    
    # Create HTTPS server
    with socketserver.TCPServer(("", PORT), HTTPSHandler) as httpd:
        # Wrap socket with SSL
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(CERT_FILE, KEY_FILE)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        
        print(f"🔒 HTTPS Server starting on port {PORT}")
        print(f"📱 Open your browser to: https://localhost:{PORT}/test_realtime_remote.html")
        print("⚠️  You'll need to accept the self-signed certificate warning")
        print("🎙️  The microphone will work because of HTTPS!")
        print("\nPress Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    main()