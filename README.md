# GPU QuickCap Backend

AI-powered video captioning backend with GPU acceleration using OpenAI Whisper.

## Features

- FastAPI web server
- OpenAI Whisper integration for transcription
- Multiple video caption templates
- Profanity filtering
- Speaker tracking
- B-roll processing
- R2 cloud storage integration
- Production-ready with gunicorn

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy environment variables:
```bash
cp .env.example .env
```

3. Edit .env file with your configuration

4. Run the server:
```bash
python app.py
```

For production deployment, use gunicorn:
```bash
gunicorn -c gunicorn.conf.py app:app
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg
- Required Python packages (see requirements.txt)
