#!/usr/bin/env python3
"""
GPU QuickCap - AI Video Captioning Server
Run this script to start the FastAPI server
"""

import uvicorn
import os

if __name__ == "__main__":
    # Try different ports if 8000 is in use
    ports_to_try = [8080, 8000, 3000, 5000, 8888]
    
    print("🚀 Starting GPU QuickCap Server...")
    print("⚡ GPU acceleration will be used if available")
    print("-" * 50)
    
    # Ensure required directories exist (directories will be created by app.py based on environment)
    # Static directories
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    # Local directories for development
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("captions", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Try to find an available port
    import socket
    for port in ports_to_try:
        try:
            # Test if port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
            
            print(f"📱 Web interface will be available at: http://localhost:{port}")
            print(f"🤖 API docs will be available at: http://localhost:{port}/docs")
            print("-" * 50)
            
            uvicorn.run(
                "app:app",
                host="0.0.0.0",
                port=port,
                reload=True,
                log_level="info"
            )
            break
        except OSError:
            continue
    else:
        print("❌ Could not find an available port. Please close other applications or run manually:")
        print("   uvicorn app:app --host 0.0.0.0 --port 8080 --reload")