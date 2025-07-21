"""
GPU QuickCap Configuration
Configure FFmpeg binary path and other settings
"""

import os

# FFmpeg Configuration
# You can set the FFmpeg binary path in several ways:

# Method 1: Set environment variable (recommended for production)
# set FFMPEG_BINARY=C:\ffmpeg\bin\ffmpeg.exe  (Windows)
# export FFMPEG_BINARY=/usr/local/bin/ffmpeg  (Linux/Mac)

# Method 2: Modify the FFMPEG_BINARY_PATH below
FFMPEG_BINARY_PATH = "/usr/bin/ffmpeg"  # Set to your FFmpeg path, e.g., "C:/ffmpeg/bin/ffmpeg.exe"

# Method 3: Place ffmpeg.exe in the project directory
# Just copy ffmpeg.exe to the same folder as app.py

# Common FFmpeg installation paths
COMMON_FFMPEG_PATHS = [
    # Windows paths
    "C:/ffmpeg/bin/ffmpeg.exe",
    "C:/Program Files/ffmpeg/bin/ffmpeg.exe",
    "C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe",
    
    # macOS paths
    "/usr/local/bin/ffmpeg",
    "/opt/homebrew/bin/ffmpeg",
    "/usr/bin/ffmpeg",
    
    # Linux paths (prioritize /usr/bin/ffmpeg)
    "/usr/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/snap/bin/ffmpeg",
    
    # Portable/local paths
    "./ffmpeg.exe",
    "./ffmpeg",
    "./bin/ffmpeg.exe",
    "./bin/ffmpeg"
]

def get_ffmpeg_binary():
    """Get the FFmpeg binary path from configuration"""
    # Priority order:
    # 1. Environment variable
    # 2. Configured path in this file
    # 3. System PATH
    # 4. Common installation paths
    
    # Check environment variable first
    env_path = os.getenv("FFMPEG_BINARY")
    if env_path:
        return env_path
    
    # Check configured path
    if FFMPEG_BINARY_PATH:
        return FFMPEG_BINARY_PATH
    
    # Default to full path
    return "/usr/bin/ffmpeg"

# Video Processing Settings
VIDEO_SETTINGS = {
    "width": 1080,
    "height": 1920,
    "bitrate": "5M",
    "preset": "fast",
    "audio_codec": "aac"
}

# Caption Settings
CAPTION_SETTINGS = {
    "max_width_percent": 0.8,  # 80% of video width
    "y_position": 0.7,         # 70% from top
    "words_per_phrase": 6,
    "line_spacing": 30,
    "max_lines": 2             # Maximum number of lines per caption
}

# Logging Settings
LOGGING_SETTINGS = {
    "level": "INFO",
    "file": "gpu_quickcap.log",
    "format": "%(asctime)s - %(levelname)s - %(message)s"
}