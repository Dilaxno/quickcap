import multiprocessing

# Set the start method to 'spawn' for CUDA compatibility with multiprocessing
# This must be done at the very beginning of the application, before any other
# modules that might use CUDA (like torch) are imported.
try:
    multiprocessing.set_start_method("spawn", force=True)
    print("--- Multiprocessing start method set to 'spawn' for CUDA ---")
except RuntimeError:
    # A RuntimeError is raised if the start method has already been set.
    # This can happen in some environments (e.g., during reloads) and is safe to ignore.
    pass

import os
import shutil
import uuid
import whisper
import subprocess
import logging
import time
import torch
import glob
import re
import requests
import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.formparsers import MultiPartParser
import asyncio
import json

# Increase the maximum request body size for file uploads to 1 GB
MultiPartParser.max_size = 1 * 1024 * 1024 * 1024
from PIL import Image, ImageDraw, ImageFont

# Import speaker tracking module
from speaker_tracking import SpeakerTracker

# Import profanity filtering modules
from profanity_list import contains_profanity, censor_profanity, PROFANITY_WORDS
from profanity_processor import ProfanityProcessor
from librosa_beep_processor import LibrosaBeepProcessor

# Import emoji processor
from emoji_processor import EmojiProcessor

# Import B-roll processor
from broll_processor import get_broll_processor

# Import R2 storage service
from r2_storage import r2_storage

# Import title generator (using Phi-3 Mini model) with lazy loading
# This prevents the model from loading at server startup
phi3_ready_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".phi3_model_ready")
phi3_available = os.path.exists(phi3_ready_marker)

if phi3_available:
    print("✅ Phi-3 model marker file found, model will be loaded on demand")
else:
    print("⚠️ Phi-3 model marker file not found, will use simple title generator")
    print("   Run setup_phi3_model.bat to set up the model if needed")

# Define a lazy loading function for the title generator
def get_title_generator():
    """Lazy loading function for title generator"""
    if phi3_available:
        try:
            from phi3_title_generator import generate_titles
            return generate_titles
        except Exception as e:
            print(f"⚠️ Error importing Phi-3 Mini title generator: {e}")
    
    # Fallback to simple title generator
    from simple_title_generator import generate_simple_titles
    return generate_simple_titles

# This will be called only when needed, not at startup
generate_titles = get_title_generator()

# Background R2 upload function
async def upload_to_r2_background(output_path: str, video_id: str, video_metadata: dict, template: str, total_time: float):
    """Upload video to R2 storage in the background after user receives the video"""
    try:
        print(f"\n🚀 Starting background R2 upload for video {video_id}...")
        logger.info(f"Starting background R2 upload for video {video_id}")
        r2_upload_start = time.time()
        
        # Upload to R2
        r2_public_url = r2_storage.upload_video(
            local_file_path=output_path,
            video_id=video_id,
            content_type="video/mp4",
            metadata=video_metadata
        )
        
        r2_upload_time = time.time() - r2_upload_start
        
        if r2_public_url:
            print(f"[OK] Background R2 upload completed in {r2_upload_time:.2f}s")
            print(f"🌐 R2 URL: {r2_public_url}")
            logger.info(f"Background R2 upload completed in {r2_upload_time:.2f}s: {r2_public_url}")
            
            # Send WebSocket notification for successful upload
            try:
                video_notification = {
                    "type": "video_uploaded_to_r2",
                    "video_id": video_id,
                    "title": f"Video {video_id}",
                    "url": r2_public_url,
                    "template": template,
                    "processing_time": total_time,
                    "upload_time": r2_upload_time,
                    "timestamp": time.time()
                }
                
                await connection_manager.broadcast_message(json.dumps(video_notification))
                print(f"[WebSocket] Sent R2 upload notification for {video_id}")
                logger.info(f"WebSocket R2 upload notification sent for video {video_id}")
                
            except Exception as ws_error:
                print(f"[WebSocket] Error sending R2 upload notification: {ws_error}")
                logger.error(f"WebSocket R2 upload notification error: {ws_error}")
            
            print(f"\n{'='*60}")
            print(f"☁️  BACKGROUND R2 UPLOAD SUMMARY:")
            print(f"   ⏱️  Upload time: {r2_upload_time:.2f}s")
            print(f"   🌐 Public URL: {r2_public_url}")
            print(f"   📹 Video ID: {video_id}")
            print(f"{'='*60}")
            
        else:
            print(f"[ERROR] Background R2 upload failed for video {video_id}")
            logger.error(f"Background R2 upload failed for video {video_id}")
            
            # Send failure notification
            try:
                failure_notification = {
                    "type": "video_upload_failed",
                    "video_id": video_id,
                    "error": "R2 upload failed",
                    "timestamp": time.time()
                }
                
                await connection_manager.broadcast_message(json.dumps(failure_notification))
                print(f"[WebSocket] Sent R2 upload failure notification for {video_id}")
                
            except Exception as ws_error:
                print(f"[WebSocket] Error sending failure notification: {ws_error}")
                logger.error(f"WebSocket failure notification error: {ws_error}")
                
    except Exception as e:
        r2_upload_time = time.time() - r2_upload_start if 'r2_upload_start' in locals() else 0
        print(f"[ERROR] Background R2 upload error for video {video_id}: {e}")
        logger.error(f"Background R2 upload error for video {video_id}: {e}")
        
        # Send error notification
        try:
            error_notification = {
                "type": "video_upload_error",
                "video_id": video_id,
                "error": str(e),
                "timestamp": time.time()
            }
            
            await connection_manager.broadcast_message(json.dumps(error_notification))
            print(f"[WebSocket] Sent R2 upload error notification for {video_id}")
            
        except Exception as ws_error:
            print(f"[WebSocket] Error sending error notification: {ws_error}")
            logger.error(f"WebSocket error notification error: {ws_error}")

# Translation service removed
# from translation_service import translation_service

# Import configuration
try:
    from config import get_ffmpeg_binary, COMMON_FFMPEG_PATHS, VIDEO_SETTINGS, CAPTION_SETTINGS
except ImportError:
    # Fallback if config.py is not available
    def get_ffmpeg_binary():
        return os.getenv("FFMPEG_BINARY", "/usr/bin/ffmpeg")
    COMMON_FFMPEG_PATHS = ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"]
    VIDEO_SETTINGS = {"width": 1080, "height": 1920, "bitrate": "5M", "preset": "fast", "audio_codec": "aac"}
    CAPTION_SETTINGS = {"max_width_percent": 0.8, "y_position": 0.7, "words_per_phrase": 6, "line_spacing": 30, "max_lines": 2}

# Configure comprehensive logging for production
import sys
from contextlib import redirect_stdout, redirect_stderr

def setup_logging():
    """Setup comprehensive logging that captures all output"""
    
    # Determine if we're in production
    is_production = os.getenv("ENVIRONMENT") == "production" or os.getenv("NODE_ENV") == "production"
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    if is_production:
        # Production logging - write to files
        log_dir = "/var/log/quickcap"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "quickcap.log")
        error_log_file = os.path.join(log_dir, "quickcap_error.log")
        
        # File handler for all logs
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler for errors and warnings
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # Console handler for critical issues (will be captured by systemd)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(console_handler)
        
        logging.info("Production logging configured - logs writing to {}".format(log_file))
        
    else:
        # Development logging - console and local file
        # Use UTF-8 encoding for file handler to support emoji
        file_handler = logging.FileHandler('gpu_quickcap.log', encoding='utf-8')
        file_handler.setFormatter(detailed_formatter)
        
        # Use StreamHandler for console output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(detailed_formatter)
        
        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        logging.info("Development logging configured")

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="GPU QuickCap", description="AI-powered video captioning with GPU acceleration")

# Add a simple API status endpoint for health checks
@app.get("/api/status")
async def api_status():
    """Simple API status endpoint for health checks"""
    # Add CORS headers explicitly for this endpoint
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
    }
    return JSONResponse(
        content={
            "status": "ok",
            "service": "GPU QuickCap API",
            "version": "1.0.0",
            "timestamp": time.time()
        },
        headers=headers
    )

@app.options("/api/status")
async def api_status_options():
    """Handle OPTIONS requests for the API status endpoint"""
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }
    return JSONResponse(content={}, headers=headers)

@app.get("/api-test")
async def api_test():
    """Serve the API test page"""
    return FileResponse("static/api-test.html")

@app.options("/upload")
async def upload_options():
    """Handle OPTIONS requests for the upload endpoint"""
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Max-Age": "86400"  # 24 hours
    }
    return JSONResponse(content={}, headers=headers)

# Import the WhisperPreloader for direct model access
from whisper_preloader_new import whisper_preloader

# --- Persistent Transcription Worker ---
# Global queues and process handle for the transcription worker
task_queue = None
result_queue = None
transcription_process = None
worker_initialized = False
model_ready = False  # Flag to track if the model is fully loaded and ready

@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event handler.
    Initializes and starts the persistent transcription worker process.
    Model will be loaded on-demand when first transcription is requested.
    """
    global task_queue, result_queue, transcription_process, worker_initialized, model_ready
    
    # Start the worker process
    model_name = os.getenv("WHISPER_MODEL", "small.en")
    
    ctx = multiprocessing.get_context('spawn')
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    
    logging.info("--- Starting transcription worker process ---")
    transcription_process = ctx.Process(target=transcription_worker, args=(task_queue, result_queue, model_name))
    transcription_process.start()
    logging.info("--- Transcription worker process started ---")
    
    # Set initial model status
    worker_initialized = True
    model_ready = False  # Will be set to True when first transcription is requested

@app.on_event("shutdown")
async def shutdown_event():
    """
    FastAPI shutdown event handler.
    Cleans up and stops the transcription worker process.
    """
    global task_queue, transcription_process
    if task_queue and transcription_process and transcription_process.is_alive():
        logging.info("--- Stopping transcription worker process ---")
        task_queue.put(None)  # Send sentinel to stop worker
        transcription_process.join(timeout=5) # Wait for the process to terminate
        if transcription_process.is_alive():
            logging.warning("--- Transcription worker process did not terminate gracefully. Terminating. ---")
            transcription_process.terminate()
        else:
            logging.info("--- Transcription worker process stopped ---")

async def restart_transcription_worker():
    """
    Restart the transcription worker process if it has died or become unresponsive.
    """
    global task_queue, result_queue, transcription_process
    
    try:
        # Clean up the old process if it exists
        if transcription_process and transcription_process.is_alive():
            logger.info("Terminating existing transcription worker...")
            task_queue.put(None)  # Send sentinel to stop worker
            transcription_process.join(timeout=3)
            if transcription_process.is_alive():
                transcription_process.terminate()
                transcription_process.join(timeout=2)
        
        # Create new queues and process
        model_name = os.getenv("WHISPER_MODEL", "small.en")
        ctx = multiprocessing.get_context('spawn')
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        
        logger.info("Starting new transcription worker process...")
        transcription_process = ctx.Process(target=transcription_worker, args=(task_queue, result_queue, model_name))
        transcription_process.start()
        
        # Give the process a moment to start
        await asyncio.sleep(1)
        
        if transcription_process.is_alive():
            logger.info("✅ Transcription worker restarted successfully")
        else:
            logger.error("❌ Failed to restart transcription worker")
            
    except Exception as e:
        logger.error(f"Error restarting transcription worker: {e}")

async def prewarm_transcription_worker():
    """
    Pre-warms the transcription worker by sending a simple signal to ensure it's running.
    This avoids the tensor shape mismatch issues with audio processing.
    """
    try:
        logging.info("Pre-warming transcription worker with simple signal...")
        
        # Instead of sending an audio file, send a special signal
        # Use a tuple with None as the path to indicate this is just a warmup check
        task = (None, "en", False)  # (path=None signals this is just a warmup check)
        task_queue.put(task)
        
        # Wait for the result with timeout
        try:
            # Use asyncio to implement timeout for queue.get()
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, result_queue.get),
                timeout=60.0  # 1 minute timeout for pre-warming
            )
            
            # Check if the worker returned an error
            if isinstance(result, dict) and 'error' in result:
                logging.error(f"Pre-warming failed: {result['error']}")
                return False
            elif isinstance(result, dict) and result.get('status') == 'ready':
                logging.info("✅ Transcription worker pre-warmed successfully!")
                return True
            else:
                logging.warning("Unexpected response from worker during pre-warming")
                return False
                
        except asyncio.TimeoutError:
            logging.error("Pre-warming timed out after 60 seconds")
            return False
        except Exception as e:
            logging.error(f"Error during pre-warming: {e}")
            return False
            
    except Exception as e:
        logging.error(f"Error pre-warming transcription worker: {e}")
        return False

# Content Security Policy Middleware
class CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add CSP headers to all responses
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://static.cloudflareinsights.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob: https://*; "
            "media-src 'self' blob: https://content.googleapis.com https://api.quickcap.pro; "
            "frame-src 'self' https://apis.google.com https://www.google.com; "
            "connect-src 'self' http://localhost:* https://* ws://localhost:* wss://*; "
            "worker-src 'self' blob:;"
        )
        
        return response

# Add CORS middleware to allow requests from the frontend
# Define allowed origins for CORS
origins = [
    "http://localhost",
    "http://localhost:8080",  # Backend port
    "http://localhost:5173",  # Default for Vite
    "http://localhost:5174",  # Additional Vite port
    "http://localhost:3000",  # Default for Create React App
    "https://app.quickcap.pro", # Production frontend
    "https://quickcap.pro",
    "https://*.quickcap.pro",  # Allow all subdomains
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Transcription-File", "X-Video-ID", "X-Sequence-Number", "X-R2-URL", "X-R2-Upload-Time", "X-Storage-Type"],
)

# Add CSP middleware
app.add_middleware(CSPMiddleware)

# R2 Storage API Endpoints
from fastapi import APIRouter

storage_router = APIRouter(prefix="/api/storage", tags=["storage"])

@storage_router.get("/status")
async def storage_status():
    """Check R2 storage service status"""
    return {
        "enabled": r2_storage.enabled,
        "bucket": r2_storage.bucket_name if r2_storage.enabled else None,
        "service": "Cloudflare R2"
    }

@storage_router.get("/videos")
async def list_stored_videos():
    """List all videos stored in R2"""
    if not r2_storage.enabled:
        raise HTTPException(status_code=503, detail="R2 storage is not enabled")
    
    try:
        videos = r2_storage.list_videos()
        return {"videos": videos, "count": len(videos)}
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail="Failed to list videos")

@storage_router.get("/videos/{video_id}")
async def get_video_info(video_id: str):
    """Get information about a specific video"""
    if not r2_storage.enabled:
        raise HTTPException(status_code=503, detail="R2 storage is not enabled")
    
    try:
        # Try different possible object keys
        possible_keys = [
            f"videos/{video_id}.mp4",
            f"videos/{video_id}.mov",
            f"videos/{video_id}.avi",
            f"videos/{video_id}"
        ]
        
        for key in possible_keys:
            info = r2_storage.get_video_info(key)
            if info:
                return {"video_id": video_id, "info": info}
        
        raise HTTPException(status_code=404, detail="Video not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get video information")

@storage_router.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a video from R2 storage"""
    if not r2_storage.enabled:
        raise HTTPException(status_code=503, detail="R2 storage is not enabled")
    
    try:
        # Try different possible object keys
        possible_keys = [
            f"videos/{video_id}.mp4",
            f"videos/{video_id}.mov", 
            f"videos/{video_id}.avi",
            f"videos/{video_id}"
        ]
        
        deleted = False
        for key in possible_keys:
            if r2_storage.delete_video(key):
                deleted = True
                break
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return {"message": "Video deleted successfully", "video_id": video_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting video: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete video")

@storage_router.get("/videos/{video_id}/url")
async def get_video_url(video_id: str, expires_in: int = 3600):
    """Get a presigned URL for accessing a video"""
    if not r2_storage.enabled:
        raise HTTPException(status_code=503, detail="R2 storage is not enabled")
    
    try:
        # Try different possible object keys
        possible_keys = [
            f"videos/{video_id}.mp4",
            f"videos/{video_id}.mov",
            f"videos/{video_id}.avi", 
            f"videos/{video_id}"
        ]
        
        for key in possible_keys:
            info = r2_storage.get_video_info(key)
            if info:
                presigned_url = r2_storage.generate_presigned_url(key, expires_in)
                public_url = info.get('public_url')
                
                return {
                    "video_id": video_id,
                    "public_url": public_url,
                    "presigned_url": presigned_url,
                    "expires_in": expires_in
                }
        
        raise HTTPException(status_code=404, detail="Video not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video URL: {e}")
        raise HTTPException(status_code=500, detail="Failed to get video URL")

@storage_router.post("/videos/{video_id}/view")
async def record_video_view(video_id: str, view_data: dict = None):
    """Record a video view for analytics"""
    if not r2_storage.enabled:
        raise HTTPException(status_code=503, detail="R2 storage is not enabled")
    
    try:
        # Check if video exists first
        possible_keys = [
            f"videos/{video_id}.mp4",
            f"videos/{video_id}.mov",
            f"videos/{video_id}.avi",
            f"videos/{video_id}"
        ]
        
        video_exists = False
        for key in possible_keys:
            info = r2_storage.get_video_info(key)
            if info:
                video_exists = True
                break
        
        if not video_exists:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # For now, we'll just return a success response with mock analytics
        # In a full implementation, you would store view data in a database
        logger.info(f"Video view recorded for {video_id}")
        
        return {
            "success": True,
            "video_id": video_id,
            "views": 1,  # Mock view count - in real implementation, increment from database
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording video view: {e}")
        raise HTTPException(status_code=500, detail="Failed to record video view")

@storage_router.post("/videos/cleanup-expired")
async def cleanup_expired_videos(cleanup_request: dict = None):
    """Clean up videos older than the specified age"""
    if not r2_storage.enabled:
        raise HTTPException(status_code=503, detail="R2 storage is not enabled")
    
    try:
        # Get max age from request, default to 5 days (432000 seconds)
        max_age = 432000
        if cleanup_request and 'maxAge' in cleanup_request:
            max_age = cleanup_request['maxAge']
        
        result = r2_storage.cleanup_expired_videos(max_age)
        
        if result['success']:
            logger.info(f"Cleanup completed: {result['deletedCount']} videos deleted")
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Cleanup failed'))
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup expired videos")

# Include the storage router in the main app
app.include_router(storage_router)

# B-roll API Endpoints
broll_router = APIRouter(prefix="/api/broll", tags=["broll"])

@broll_router.post("/upload/{video_id}")
async def upload_broll_image(
    video_id: str, 
    phrase_id: str = Form(...),
    image: UploadFile = File(...),
):
    """Upload a B-roll image for a specific phrase in a video"""
    try:
        # Get the B-roll processor
        broll_processor = get_broll_processor()
        
        # Read the image data
        image_data = await image.read()
        
        # Save the image
        image_path = broll_processor.save_broll_image(image_data, video_id, phrase_id)
        
        # Get existing B-roll data
        broll_data = broll_processor.get_broll_data(video_id)
        
        # Update the B-roll data with the new image
        if "images" not in broll_data:
            broll_data["images"] = {}
        
        broll_data["images"][phrase_id] = image_path
        
        # Save the updated B-roll data
        broll_processor.save_broll_data(video_id, broll_data)
        
        return {
            "success": True,
            "video_id": video_id,
            "phrase_id": phrase_id,
            "image_path": image_path
        }
    except Exception as e:
        logger.error(f"Error uploading B-roll image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload B-roll image: {str(e)}")

@broll_router.get("/{video_id}")
async def get_broll_data(video_id: str):
    """Get B-roll data for a video"""
    try:
        broll_processor = get_broll_processor()
        broll_data = broll_processor.get_broll_data(video_id)
        
        return {
            "success": True,
            "video_id": video_id,
            "broll_data": broll_data
        }
    except Exception as e:
        logger.error(f"Error getting B-roll data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get B-roll data: {str(e)}")

@broll_router.delete("/{video_id}/{phrase_id}")
async def delete_broll_image(video_id: str, phrase_id: str):
    """Delete a B-roll image for a specific phrase"""
    try:
        broll_processor = get_broll_processor()
        
        # Get existing B-roll data
        broll_data = broll_processor.get_broll_data(video_id)
        
        # Check if the image exists
        if "images" not in broll_data or phrase_id not in broll_data["images"]:
            raise HTTPException(status_code=404, detail="B-roll image not found")
        
        # Get the image path
        image_path = broll_data["images"][phrase_id]
        
        # Delete the image file if it exists
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Remove the image from the B-roll data
        del broll_data["images"][phrase_id]
        
        # Save the updated B-roll data
        broll_processor.save_broll_data(video_id, broll_data)
        
        return {
            "success": True,
            "video_id": video_id,
            "phrase_id": phrase_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting B-roll image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete B-roll image: {str(e)}")

@broll_router.post("/process/{video_id}")
async def process_video_with_broll(
    video_id: str,
    timestamps: list
):
    """Process a video with B-roll images"""
    try:
        # Get the B-roll processor
        broll_processor = get_broll_processor()
        
        # Get B-roll data for this video
        broll_data = broll_processor.get_broll_data(video_id)
        
        # Check if there are any B-roll images
        if not broll_data.get("images"):
            raise HTTPException(status_code=400, detail="No B-roll images found for this video")
        
        # Find the input video path
        input_video_path = os.path.join("outputs", f"{video_id}.mp4")
        if not os.path.exists(input_video_path):
            input_video_path = os.path.join("outputs", f"{video_id}_final.mp4")
            if not os.path.exists(input_video_path):
                raise HTTPException(status_code=404, detail="Video not found")
        
        # Define the output path for the B-roll video
        output_video_path = os.path.join("outputs", f"{video_id}_broll.mp4")
        
        # Process the video with B-roll
        success = broll_processor.apply_broll_to_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            broll_data=broll_data,
            timestamps=timestamps
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process video with B-roll")
        
        # Return the path to the processed video
        return {
            "success": True,
            "video_id": video_id,
            "output_path": output_video_path,
            "download_url": f"/api/videos/{video_id}/download?type=broll"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video with B-roll: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process video with B-roll: {str(e)}")
# Include the B-roll router in the main app
app.include_router(broll_router)

# Translation API Endpoints removed
# The translation service and its endpoints have been completely removed

# WebSocket Connection Manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        print(f"[WebSocket] User {user_id} connected")
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            print(f"[WebSocket] User {user_id} disconnected")
    
    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(message)
                print(f"[WebSocket] Sent message to user {user_id}: {message}")
            except Exception as e:
                print(f"[WebSocket] Error sending message to user {user_id}: {e}")
                self.disconnect(user_id)
    
    async def broadcast_message(self, message: str):
        for user_id, connection in self.active_connections.copy().items():
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"[WebSocket] Error broadcasting to user {user_id}: {e}")
                self.disconnect(user_id)

# Global connection manager instance
connection_manager = ConnectionManager()

# WebSocket endpoint for real-time video updates
@app.websocket("/ws/videos/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await connection_manager.connect(websocket, user_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(user_id)
    except Exception as e:
        print(f"[WebSocket] Error in connection for user {user_id}: {e}")
        connection_manager.disconnect(user_id)

templates = Jinja2Templates(directory="templates")

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Note: outputs directory mount will be configured after directory setup

# Handle Chrome DevTools request
@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools():
    # Return an empty JSON response
    return JSONResponse(content={})

# Create a favicon.ico route
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

# Add a model status endpoint to check if the Whisper model is loaded
@app.get("/api/model/status")
async def model_status():
    """Check the status of the Whisper model loading"""
    try:
        # Check if the worker process is alive and ready
        worker_ready = transcription_process is not None and transcription_process.is_alive()
        
        # Try to get status from the status file first
        try:
            from model_status_fix import get_model_status
            file_status = get_model_status()
            model_loaded = file_status.get("model_loaded", False)
            loading_in_progress = file_status.get("loading_in_progress", False)
        except Exception as file_e:
            logger.warning(f"Could not read model status file: {file_e}")
            model_loaded = whisper_preloader.model_loaded or worker_ready
            loading_in_progress = False  # Default to False since loading_in_progress attribute doesn't exist
        
        # Create a simplified status response that doesn't require accessing CUDA
        # This avoids the CUDA reinitialization error in forked subprocesses
        status = {
            "model_name": whisper_preloader.model_name,
            "model_loaded": model_loaded,  # Use status from file if available
            "loading_in_progress": loading_in_progress,
            "cuda_available": whisper_preloader.cuda_available,
            "worker_ready": worker_ready,
            "server_time": time.time()
        }
        
        # If worker is ready, the model is definitely loaded
        if worker_ready:
            status["model_loaded"] = True
            
            # Update the status file to reflect this
            try:
                from model_status_fix import update_model_status
                update_model_status(loaded=True, loading=False)
            except Exception as update_e:
                logger.warning(f"Could not update model status file: {update_e}")
        
        # If model is not loaded and not loading, trigger loading
        if not status["model_loaded"] and not loading_in_progress:
            # Start loading in background thread
            import threading
            preload_thread = threading.Thread(target=whisper_preloader.load_model)
            preload_thread.daemon = True
            preload_thread.start()
            status["loading_started"] = True
        
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error in model status endpoint: {e}")
        return JSONResponse(content={"error": str(e), "message": "Error checking model status"}, status_code=500)

# Add a simple test endpoint to verify API accessibility
@app.get("/api/whisper-test")
async def whisper_test():
    """Simple test endpoint to verify API accessibility"""
    return JSONResponse(content={"status": "ok", "message": "Whisper API is accessible"})

# Add a model preload endpoint to force model loading
@app.post("/api/model/preload")
async def preload_model():
    """Force preloading of the Whisper model"""
    # Start loading in background thread
    import threading
    preload_thread = threading.Thread(target=lambda: whisper_preloader.load_model(force=True))
    preload_thread.daemon = True
    preload_thread.start()
    
    return JSONResponse(content={"message": "Model preloading started", "status": whisper_preloader.get_status()})

def transcription_worker(task_queue, result_queue, model_name):
    """
    A persistent worker process that loads the Whisper model on-demand and processes
    transcription tasks from a queue.
    """
    import whisper
    import torch
    import logging
    import time
    from whisper_preloader_new import WhisperPreloader

    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"[Worker] Process started. Initializing WhisperPreloader...")
        # Create a worker-specific instance of the preloader
        worker_preloader = WhisperPreloader()
        
        # Model will be loaded on first transcription request
        logger.info("[Worker] WhisperPreloader initialized - model will be loaded on first request")

        while True:
            task = task_queue.get()
            
            if task is None:
                logger.info("[Worker] Received shutdown signal")
                break
            
            video_path, language, verbose = task
            
            # Handle special pre-warming signal
            if video_path is None:
                logger.info("[Worker] Received warmup signal")
                result_queue.put({"status": "ready"})
                continue
            
            try:
                result = worker_preloader.transcribe(
                    video_path,
                    language=language,
                    verbose=verbose
                )
                result_queue.put(result)
            except Exception as e:
                logger.error(f"[Worker] Transcription error: {e}")
                result_queue.put({"error": str(e)})
            task = task_queue.get()
            
            if task is None:
                logger.info("[Worker] Sentinel received. Shutting down.")
                break
            
            video_path, language, verbose = task
            
            # Handle special pre-warming signal
            if video_path is None:
                logger.info("[Worker] Received pre-warming signal, sending ready status")
                result_queue.put({"status": "ready", "message": "Worker is initialized and ready"})
                continue
                
            try:
                logger.info(f"[Worker] Starting transcription for: {video_path}")
                transcribe_start = time.time()
                
                # Use the preloader to handle transcription
                result = worker_preloader.transcribe(
                    video_path, 
                    language=language, 
                    verbose=verbose
                )
                
                transcribe_time = time.time() - transcribe_start
                logger.info(f"[Worker] Transcription finished in {transcribe_time:.2f}s")
                result_queue.put(result)
            except Exception as e:
                logger.error(f"[Worker] An error occurred during transcription: {e}", exc_info=True)
                result_queue.put({"error": str(e)})
    except Exception as e:
        logger.error(f"[Worker] Failed to initialize or run worker: {e}", exc_info=True)
        result_queue.put({"error": f"Worker initialization failed: {str(e)}"})

# Check if CUDA is available
if torch.cuda.is_available():
    logger.info(f"[LAUNCH] GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("⚠️  Running on CPU (GPU not available)")

# Configure directories based on environment
IS_PRODUCTION = os.getenv("ENVIRONMENT") == "production" or os.getenv("NODE_ENV") == "production"

if IS_PRODUCTION:
    # Use /tmp for temporary files in production
    UPLOAD_DIR = "/tmp/uploads"
    CAPTION_DIR = "/tmp/captions"
    OUTPUT_DIR = "/tmp/outputs"
    logger.info("[PROD] Production mode: Using /tmp for temporary files")
else:
    # Use local directories for development
    UPLOAD_DIR = "uploads"
    CAPTION_DIR = "captions"
    OUTPUT_DIR = "outputs"
    logger.info("[DEV] Development mode: Using local directories")

def get_next_sequence_number():
    """
    Get the next sequence number for output files by checking existing files
    and finding the highest number used so far.
    Returns a formatted string like "01", "02", etc.
    """
    # Look for files with pattern "Quickcap output XX" in the output directory
    existing_files = glob.glob(os.path.join(OUTPUT_DIR, "Quickcap output *.*"))
    
    # Extract sequence numbers from filenames
    sequence_numbers = []
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        match = re.search(r'Quickcap output (\d+)', filename)
        if match:
            sequence_numbers.append(int(match.group(1)))
    
    # Get the next sequence number (max + 1, or 1 if no files exist)
    next_number = 1
    if sequence_numbers:
        next_number = max(sequence_numbers) + 1
    
    # Format as two-digit string (01, 02, etc.)
    return f"{next_number:02d}"

def generate_animation_filter(animation_type, animation_speed, start_time, end_time, x_pos="(W-w)/2", y_pos="H*0.7"):
    """
    Generate FFmpeg filter for caption animations
    
    Args:
        animation_type: Type of animation (typewriter, bounce, slide_in, etc.)
        animation_speed: Speed multiplier for animation (0.5-2.0)
        start_time: Start time for the animation
        end_time: End time for the animation
        x_pos: X position expression
        y_pos: Y position expression
    
    Returns:
        Tuple of (overlay_filter, additional_filters)
    """
    duration = end_time - start_time
    base_duration = max(0.3, min(duration * 0.3, 1.0))  # Base animation duration (0.3s to 1s)
    anim_duration = base_duration / animation_speed
    
    logger.info(f"[FFMPEG] Generating animation: {animation_type} (speed: {animation_speed}x, duration: {anim_duration:.2f}s)")
    
    if animation_type == "none":
        overlay_filter = f"overlay=enable='between(t,{start_time},{end_time})':x={x_pos}:y={y_pos}"
        logger.debug(f"No animation: {overlay_filter}")
        return overlay_filter, ""
    
    elif animation_type == "bounce":
        # Bounce effect - text bounces down from above (simplified)
        bounce_height = 50
        # Use a simpler bounce that works better in FFmpeg
        bounce_y = f"if(lt(t-{start_time},{anim_duration}),{y_pos}-{bounce_height}+{bounce_height}*(t-{start_time})/{anim_duration},{y_pos})"
        overlay_filter = f"overlay=enable='between(t,{start_time},{end_time})':x={x_pos}:y='{bounce_y}'"
        logger.debug(f"Bounce animation: {overlay_filter}")
        return overlay_filter, ""
    
    elif animation_type == "slide_in":
        # Slide in from left - start off-screen and slide to center
        slide_x = f"if(lt(t-{start_time},{anim_duration}),-w+(w+{x_pos})*(t-{start_time})/{anim_duration},{x_pos})"
        overlay_filter = f"overlay=enable='between(t,{start_time},{end_time})':x='{slide_x}':y={y_pos}"
        logger.debug(f"Slide in animation: {overlay_filter}")
        return overlay_filter, ""
    
    elif animation_type == "slide_out":
        # Slide out to right at the end
        slide_out_start = max(start_time, end_time - anim_duration)
        slide_x = f"if(gt(t,{slide_out_start}),{x_pos}+W*(t-{slide_out_start})/{anim_duration},{x_pos})"
        overlay_filter = f"overlay=enable='between(t,{start_time},{end_time})':x='{slide_x}':y={y_pos}"
        logger.debug(f"Slide out animation: {overlay_filter}")
        return overlay_filter, ""
    
    elif animation_type == "zoom_in":
        # Zoom in effect - use scale filter and adjust position
        scale_start = 0.3
        scale_expr = f"if(lt(t-{start_time},{anim_duration}),{scale_start}+(1-{scale_start})*(t-{start_time})/{anim_duration},1)"
        scale_filter = f"scale=iw*({scale_expr}):ih*({scale_expr})"
        overlay_filter = f"overlay=enable='between(t,{start_time},{end_time})':x={x_pos}:y={y_pos}"
        logger.debug(f"Zoom in animation: scale={scale_filter}, overlay={overlay_filter}")
        return overlay_filter, scale_filter
    
    elif animation_type == "wave":
        # Wave effect - vertical movement
        wave_amplitude = 10
        wave_y = f"{y_pos}+{wave_amplitude}*sin(2*3.14159*2*(t-{start_time}))"
        overlay_filter = f"overlay=enable='between(t,{start_time},{end_time})':x={x_pos}:y='{wave_y}'"
        logger.debug(f"Wave animation: {overlay_filter}")
        return overlay_filter, ""
    
    elif animation_type == "fade_color":
        # Color fade effect - use hue rotation
        hue_filter = f"hue=h=60*(t-{start_time}):s=1"
        overlay_filter = f"overlay=enable='between(t,{start_time},{end_time})':x={x_pos}:y={y_pos}"
        logger.debug(f"Fade color animation: hue={hue_filter}, overlay={overlay_filter}")
        return overlay_filter, hue_filter
    
    elif animation_type == "glow":
        # Enhanced glow effect - brightness, contrast, and saturation variation for a more dramatic lighting effect
        # Use a combination of brightness and contrast changes with a faster cycle for more dynamic lighting
        brightness_filter = f"eq=brightness=0.25*sin(2*3.14159*3*(t-{start_time})):contrast=1.2+0.2*sin(2*3.14159*2*(t-{start_time})):saturation=1.1+0.1*sin(2*3.14159*4*(t-{start_time}))"
        
        # Add a subtle vertical movement to enhance the lighting effect
        y_offset = f"{y_pos}-2*sin(2*3.14159*1.5*(t-{start_time}))"
        overlay_filter = f"overlay=enable='between(t,{start_time},{end_time})':x={x_pos}:y='{y_offset}'"
        
        logger.debug(f"Enhanced lighting effect: {brightness_filter}, overlay={overlay_filter}")
        return overlay_filter, brightness_filter
    
    elif animation_type == "shake":
        # Shake effect - small random movements with bounce-like vertical motion
        shake_intensity_x = 2  # Reduced horizontal shake
        shake_intensity_y = 4  # Increased vertical shake for bounce effect
        
        # Use different frequencies for x and y to create more natural movement
        # Vertical movement is slower to simulate bounce
        shake_x = f"{x_pos}+{shake_intensity_x}*sin(2*3.14159*8*(t-{start_time}))"
        
        # For y-axis, use a combination of sine waves to create a more bounce-like effect
        # Main bounce is slower, with a smaller faster component
        shake_y = f"{y_pos}+{shake_intensity_y}*sin(2*3.14159*3*(t-{start_time}))+{shake_intensity_y/2}*sin(2*3.14159*6*(t-{start_time}))"
        
        overlay_filter = f"overlay=enable='between(t,{start_time},{end_time})':x='{shake_x}':y='{shake_y}'"
        logger.debug(f"Enhanced shake/bounce animation: {overlay_filter}")
        return overlay_filter, ""
    
    elif animation_type == "sparkle":
        # Sparkle effect - brightness and saturation variation
        sparkle_filter = f"eq=brightness=0.1*sin(2*3.14159*3*(t-{start_time})):saturation=1.2"
        overlay_filter = f"overlay=enable='between(t,{start_time},{end_time})':x={x_pos}:y={y_pos}"
        logger.debug(f"Sparkle animation: sparkle={sparkle_filter}, overlay={overlay_filter}")
        return overlay_filter, sparkle_filter
    
    elif animation_type == "typewriter":
        # Typewriter effect - crop from left to right gradually
        crop_width = f"if(lt(t-{start_time},{anim_duration}),w*(t-{start_time})/{anim_duration},w)"
        crop_filter = f"crop='{crop_width}':h:0:0"
        overlay_filter = f"overlay=enable='between(t,{start_time},{end_time})':x={x_pos}:y={y_pos}"
        logger.debug(f"Typewriter animation: crop={crop_filter}, overlay={overlay_filter}")
        return overlay_filter, crop_filter
    
    else:
        # Default to no animation
        overlay_filter = f"overlay=enable='between(t,{start_time},{end_time})':x={x_pos}:y={y_pos}"
        logger.warning(f"Unknown animation type '{animation_type}', using default: {overlay_filter}")
        return overlay_filter, ""

# Plan duration limits (in seconds)
PLAN_DURATION_LIMITS = {
    "free": 40,      # 40 seconds for free users
    "basic": 120,    # 2 minutes for basic plan
    "pro": 300,      # 5 minutes for pro plan
    "premium": 600,  # 10 minutes for premium plan (if exists)
    "enterprise": float('inf')  # No limit for enterprise
}

# FFmpeg Configuration
FFMPEG_BINARY = get_ffmpeg_binary()  # Get from configuration
FFMPEG_PATHS = [
    FFMPEG_BINARY,  # User-specified or configured
    "ffmpeg.exe",   # Windows executable in current directory
    "ffmpeg",       # System PATH
    *COMMON_FFMPEG_PATHS  # Paths from configuration
]

def find_ffmpeg_binary():
    """Find the best available FFmpeg binary"""
    logger.info("[FFMPEG] Searching for FFmpeg binary...")
    
    for ffmpeg_path in FFMPEG_PATHS:
        try:
            # Test if the binary exists and works
            result = subprocess.run([ffmpeg_path, "-version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Extract version info
                version_line = result.stdout.split('\n')[0]
                logger.info(f"[OK] Found FFmpeg: {ffmpeg_path}")
                logger.info(f"[INFO] Version: {version_line}")
                return ffmpeg_path
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.debug(f"FFmpeg not found at {ffmpeg_path}: {e}")
            continue
    
    logger.error("❌ FFmpeg not found! Please install FFmpeg or set FFMPEG_BINARY environment variable")
    raise Exception("FFmpeg not found. Please install FFmpeg or specify the path using FFMPEG_BINARY environment variable")

# Find and validate FFmpeg
FFMPEG_CMD = find_ffmpeg_binary()

def get_video_info(video_path):
    """
    Get video information (width, height, duration) using FFmpeg
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video information (width, height, duration)
    """
    try:
        logger.info(f"Getting video info for: {video_path}")
        
        # Run FFmpeg to get video information in JSON format
        cmd = [
            FFMPEG_CMD, 
            "-i", video_path, 
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-of", "json",
            "-hide_banner"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            logger.error(f"Error getting video info: {result.stderr}")
            # Return default values if FFmpeg fails
            return {"width": 1920, "height": 1080, "duration": 0}
        
        # Parse JSON output
        try:
            info = json.loads(result.stdout)
            stream_info = info.get("streams", [{}])[0]
            
            # Extract width, height, and duration
            width = int(stream_info.get("width", 1920))
            height = int(stream_info.get("height", 1080))
            
            # Duration might be missing in some formats, try to get it
            duration = 0
            if "duration" in stream_info:
                duration = float(stream_info["duration"])
            
            logger.info(f"Video info: {width}x{height}, duration: {duration}s")
            return {
                "width": width,
                "height": height,
                "duration": duration
            }
        except (json.JSONDecodeError, IndexError, KeyError, ValueError) as e:
            logger.error(f"Error parsing video info: {e}")
            return {"width": 1920, "height": 1080, "duration": 0}
            
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {"width": 1920, "height": 1080, "duration": 0}

# Set environment variable for FFmpeg so other libraries (like Whisper) can find it
os.environ['FFMPEG_BINARY'] = FFMPEG_CMD
# Also add the FFmpeg directory to PATH if it's not already there
ffmpeg_dir = os.path.dirname(FFMPEG_CMD)
if ffmpeg_dir and ffmpeg_dir not in os.environ.get('PATH', ''):
    os.environ['PATH'] = f"{ffmpeg_dir}:{os.environ.get('PATH', '')}"
    logger.info(f"[FFMPEG] Added {ffmpeg_dir} to PATH for FFmpeg discovery")

def get_video_duration(video_path):
    """
    Get the duration of a video file in seconds using FFprobe
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Duration in seconds as float, or None if unable to get duration
    """
    try:
        # Try to find ffprobe (usually comes with ffmpeg)
        ffprobe_paths = [
            "ffprobe.exe",  # Windows executable in current directory
            "ffprobe",      # System PATH
            FFMPEG_CMD.replace("ffmpeg", "ffprobe"),  # Same directory as ffmpeg
            os.path.join(os.path.dirname(FFMPEG_CMD), "ffprobe.exe"),
            os.path.join(os.path.dirname(FFMPEG_CMD), "ffprobe")
        ]
        
        ffprobe_cmd = None
        for path in ffprobe_paths:
            try:
                result = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    ffprobe_cmd = path
                    break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        if not ffprobe_cmd:
            logger.warning("FFprobe not found, falling back to FFmpeg for duration")
            # Fallback to using ffmpeg to get duration
            result = subprocess.run([
                FFMPEG_CMD, "-i", video_path, "-f", "null", "-"
            ], capture_output=True, text=True, timeout=30)
            
            # Parse duration from ffmpeg output
            duration_pattern = r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})"
            match = re.search(duration_pattern, result.stderr)
            if match:
                hours, minutes, seconds = match.groups()
                total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                return total_seconds
        else:
            # Use ffprobe to get precise duration
            result = subprocess.run([
                ffprobe_cmd, "-v", "quiet", "-print_format", "json", 
                "-show_format", "-show_streams", video_path
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                probe_data = json.loads(result.stdout)
                
                # Try to get duration from format first
                if "format" in probe_data and "duration" in probe_data["format"]:
                    return float(probe_data["format"]["duration"])
                
                # Fallback to stream duration
                for stream in probe_data.get("streams", []):
                    if stream.get("codec_type") == "video" and "duration" in stream:
                        return float(stream["duration"])
        
        logger.error(f"Could not determine video duration for {video_path}")
        return None
        
    except Exception as e:
        logger.error(f"Error getting video duration: {e}")
        return None

def validate_video_duration(duration_seconds, user_plan):
    """
    Validate if video duration is within plan limits
    
    Args:
        duration_seconds: Video duration in seconds
        user_plan: User's subscription plan
    
    Returns:
        Tuple of (is_valid: bool, error_message: str, limit_seconds: int)
    """
    if duration_seconds is None:
        return False, "Unable to determine video duration", 0
    
    plan_limit = PLAN_DURATION_LIMITS.get(user_plan, PLAN_DURATION_LIMITS["free"])
    
    if duration_seconds > plan_limit:
        # Format duration for user-friendly message
        def format_duration(seconds):
            if seconds < 60:
                return f"{int(seconds)} seconds"
            elif seconds < 3600:
                minutes = int(seconds // 60)
                remaining_seconds = int(seconds % 60)
                if remaining_seconds > 0:
                    return f"{minutes} minutes {remaining_seconds} seconds"
                return f"{minutes} minutes"
            else:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                return f"{hours} hours {minutes} minutes"
        
        error_msg = f"Video duration ({format_duration(duration_seconds)}) exceeds the {user_plan} plan limit of {format_duration(plan_limit)}."
        if user_plan == "free":
            error_msg += " Please upgrade to Basic plan (2 minutes) or Pro plan (5 minutes) to process longer videos."
        elif user_plan == "basic":
            error_msg += " Please upgrade to Pro plan (5 minutes) to process longer videos."
        
        return False, error_msg, plan_limit
    
    return True, "", plan_limit

# Create directories with logging
logger.info("[SETUP] Setting up directories...")
logger.info("[SETUP] Creating required directories")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CAPTION_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"[OK] Directories ready: {UPLOAD_DIR}/, {CAPTION_DIR}/, {OUTPUT_DIR}/")
logger.info(f"Directories created: {UPLOAD_DIR}/, {CAPTION_DIR}/, {OUTPUT_DIR}/")

# Mount the outputs directory so videos can be served (after directory setup)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Settings (from configuration)
VIDEO_WIDTH = VIDEO_SETTINGS["width"]
VIDEO_HEIGHT = VIDEO_SETTINGS["height"]
MAX_WIDTH = int(VIDEO_WIDTH * CAPTION_SETTINGS["max_width_percent"])
FONT_SIZE = 72
WORDS_PER_PHRASE = CAPTION_SETTINGS["words_per_phrase"]
CAPTION_Y_POSITION = CAPTION_SETTINGS["y_position"]
LINE_SPACING = CAPTION_SETTINGS["line_spacing"]
MAX_LINES = CAPTION_SETTINGS.get("max_lines", 2)

# Caption Templates
CAPTION_TEMPLATES = {
    
    "Hopelesscore": {
        "name": "Hopelesscore Style",
        "description": "Large text with random vertical positioning, nostalgic filter, and 200% audio boost",
        "hopelesscore_style": True,  # Enable Hopelesscore processing
        "hopecore_style": True,  # Special flag for hopecore processing (ensures same caption logic)
        "font_paths": [
            "fonts/DS George.otf",  # Primary font for normal words
            "DS George.otf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "emphasis_font_paths": [
            "fonts/AsikueTrial.otf",  # Font for emphasized words
            "AsikueTrial.otf",
            "fonts/Impact.ttf",  # Fallback
            "C:/Windows/Fonts/impact.ttf"
        ],
        "font_size": 110,  # Font size for normal words
        "emphasis_font_size": 120,  # Font size for emphasized words
        "emphasis_font_weight": 600,  # Medium weight for emphasized words
        "use_emphasis_font": True,  # Use AsikueTrial font for emphasized words
        "text_color": (255, 255, 255, 255),  # White
        "highlight_colors": [  # Cycling colors for words - Gold as primary color
            (255, 215, 0, 255),   # Gold (primary)
            (255, 215, 0, 255),   # Gold (repeated for emphasis)
            (255, 165, 0, 255),   # Orange
            (255, 255, 255, 255)  # White
        ],
        "line_spacing": 45,  # Increased line spacing for better readability
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 4,  # Increased stroke width for better visibility
        "shadow_color": (0, 0, 0, 180),  # Darker shadow for better contrast
        "shadow_offset": (5, 5),  # Larger shadow offset for more depth
        "glow_color": (255, 215, 0, 80),  # Subtle gold glow
        "glow_radius": 10,  # Medium glow radius
        "has_glow": True,  # Enable glow effect
        "position_zones": [
            "top_left", "top_center", "top_right",
            "center_left", "center", "center_right",
            "bottom_left", "bottom_center", "bottom_right"
        ],  # All possible position zones for varied placement
        "random_vertical_position": True,  # Enable random vertical positioning
        "audio_boost": 2.0,  # 200% audio boost
        "word_sequence_mode": True,  # Enable word sequence mode (3-4 words at a time)
        "word_sequence_count": [3, 4],  # Random sequence of 3 or 4 words
        "uppercase": True,  # Convert all text to uppercase
        "has_highlighting": True,  # Enable word-by-word highlighting
        "has_real_highlighting": True,  # Use real highlighting system
        "vertical_stack": True,  # Enable vertical word stacking
        "dynamic_positioning": True,  # Enable dynamic phrase positioning
        "sequential_word_reveal": True,  # Words appear at natural speech timestamps
        "pastel_theme": True,  # Enable pastel color processing
        "gentle_effects": True,  # Enable gentle visual effects
        "max_emphasized_words": 2,  # Limit to maximum 2 emphasized words per phrase
        "vertical_padding": 60,  # Top and bottom padding to prevent text overflow
        "max_text_height": 0.8,  # Maximum text height as fraction of screen height (80%)
        "preserve_template_settings": True  # Preserve all template-specific settings
    },

    "MrBeast": {
        "name": "MrBeast Style",
        "description": "Komikax font with cycling colors (Yellow→Green→Red), 3px stroke, and white drop shadow",
        "font_paths": [
            "fonts/Komikax.ttf",
            "Komikax.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # White
        "highlight_colors": [  # Cycling colors for words
            (255, 255, 0, 255),   # Yellow
            (0, 255, 0, 255),     # Green
            (255, 0, 0, 255)      # Red
        ],
        "line_spacing": 30,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": (255, 255, 255, 180),  # White drop shadow with good opacity
        "shadow_offset": (3, 3)  # Soft shadow offset (x, y) in pixels
    },
    "Bold Green": {
        "name": "Bold Green",
        "description": "Uni Sans Heavy font with bright green word highlighting and shadow",
        "font_paths": [
            "fonts/Uni Sans Heavy.otf",
            "Uni Sans Heavy.otf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # White
        "highlight_colors": [  # Single bright green color for words
            (0, 255, 0, 255)   # Bright Green
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4),  # Shadow offset (x, y) in pixels
        "scale_effect": True,  # Enable subtle scale effect for highlighted words
        "scale_factor": 1.1    # Scale highlighted words by 10% (subtle effect)
    },
    "Bold Sunshine": {
        "name": "Bold Sunshine",
        "description": "Theboldfont with bright yellow word highlighting, 2px outline, and extra large spacing",
        "font_paths": [
            "fonts/Theboldfont.ttf",
            "Theboldfont.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # White
        "highlight_colors": [  # Single bright yellow color for words
            (255, 255, 0, 255)   # Bright Yellow
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 2,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4)  # Shadow offset (x, y) in pixels
    },
    "Premium Orange": {
        "name": "Premium Orange",
        "description": "Poppins Bold Italic with vibrant orange highlighting, uppercase text, and dynamic spacing",
        "font_paths": [
            "fonts/Poppins-BoldItalic.ttf",
            "Poppins-BoldItalic.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # White
        "highlight_colors": [  # Single vibrant orange color for words
            (235, 91, 0, 255)   # Vibrant Orange (#EB5B00)
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4),  # Shadow offset (x, y) in pixels
        "uppercase": True  # Convert all text to uppercase
    },
    "Minimal White": {
        "name": "Minimal White",
        "description": "SpiegelSans with clean white highlighting, minimal styling, and professional spacing",
        "font_paths": [
            "fonts/SpiegelSans.otf",
            "SpiegelSans.otf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # White
        "highlight_colors": [  # Pure white color for words
            (255, 255, 255, 255)   # Pure White
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4)  # Shadow offset (x, y) in pixels
    },
    "Orange Meme": {
        "name": "Orange Meme",
        "description": "LuckiestGuy with uniform orange color, bold cartoon styling, and uppercase text",
        "font_paths": [
            "fonts/LuckiestGuy.ttf",
            "LuckiestGuy.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 140, 0, 255),  # Orange
        "highlight_colors": [  # Same orange color for uniform appearance
            (255, 140, 0, 255)   # Orange (same as text_color)
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4),  # Shadow offset (x, y) in pixels
        "uppercase": True  # Convert all text to uppercase
    },
    "Cinematic Quote": {
        "name": "Cinematic Quote",
        "description": "Proxima Nova Alt Condensed Black Italic with bright yellow highlighting and title case",
        "font_paths": [
            "fonts/Proxima Nova Alt Condensed Black Italic.otf",
            "Proxima Nova Alt Condensed Black Italic.otf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # White
        "highlight_colors": [  # Bright yellow for keywords
            (255, 255, 0, 255)   # Bright Yellow
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4),  # Shadow offset (x, y) in pixels
        "title_case": True  # Convert all text to title case
    },
    "Word by Word": {
        "name": "Word by Word",
        "description": "Poppins Black Italic with word-by-word display, enhanced font size, and uniform white color",
        "font_paths": [
            "fonts/Poppins-BlackItalic.ttf",
            "Poppins-BlackItalic.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 75,
        "text_color": (255, 255, 255, 255),  # Pure White
        "highlight_colors": [  # Same white color for uniform appearance
            (255, 255, 255, 255)   # Pure White (same as text_color)
        ],
        "line_spacing": 50,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4),  # Shadow offset (x, y) in pixels
        "word_by_word": True,  # Enable word-by-word functionality
        "enhanced_font_size": 82  # 10% increase (75 * 1.1 = 82.5, rounded to 82)
    },
    "esports_caption": {
        "name": "Esports Caption",
        "description": "Exo2-Black with vibrant red-orange highlighting, gaming-style effects, and uppercase text",
        "font_paths": [
            "fonts/Exo2-Black.ttf",
            "Exo2-Black.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # Pure White
        "highlight_colors": [  # Vibrant red-orange for keywords
            (255, 69, 0, 255)   # Red-Orange (#FF4500)
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 2,
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4),  # Shadow offset (x, y) in pixels
        "uppercase": True,  # Convert all text to uppercase
        "scale_effect": True,  # Enable scale effect for highlighted words
        "scale_factor": 1.15  # Scale highlighted words by 15% (65px -> 75px)
    },
        "Reaction Pop": {
        "name": "Reaction Pop",
        "description": "Proxima Nova Alt Condensed Black with vibrant red highlighting and title case formatting",
        "font_paths": [
            "fonts/Proxima Nova Alt Condensed Black.otf",
            "Proxima Nova Alt Condensed Black.otf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 70,
        "text_color": (255, 255, 255, 255),  # Pure White
        "highlight_colors": [  # Pure red for keywords
            (255, 0, 0, 255)   # Pure Red (#FF0000)
        ],
        "line_spacing": 45,  # Between 40px and 50px for optimal spacing
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": None,  # No shadow for clean look
        "shadow_offset": None,
        "title_case": True,  # Convert all text to title case
        "scale_effect": True,  # Enable scale effect for highlighted words
        "scale_factor": 1.15  # Scale highlighted words by 15% (70px -> 80px)
    },
    




    "Kai Cenat": {
        "name": "Kai Cenat",
        "description": "Impact with Twitch purple/gold streaming colors, hype energy effects",
        "font_paths": [
            "fonts/Impact.ttf",
            "Impact.ttf",
            "C:/Windows/Fonts/impact.ttf",  # Windows system font
            "fonts/arial.ttf"  # Fallback
        ],
        "font_size": 74,
        "text_color": (255, 255, 255, 255),  # White base text
        "highlight_colors": [  # Twitch/streaming colors
            (145, 70, 255, 255),    # Twitch Purple
            (255, 215, 0, 255),     # Gold
            (255, 20, 147, 255),    # Deep Pink
            (0, 255, 127, 255),     # Spring Green
            (255, 69, 0, 255)       # Red-Orange
        ],
        "line_spacing": 50,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 4,
        "shadow_color": (145, 70, 255, 130),  # Purple shadow
        "shadow_offset": (5, 5),
        "uppercase": True,
        "streaming_style": True,
        "hype_energy": True,
        "twitch_branding": True,
        "scale_effect": True,
        "scale_factor": 1.25,
        "pulse_effect": True,
        "pulse_frequency": 4.0
    },
    "Film noir": {
        "name": "Film Noir",
        "description": "Uni Sans Heavy font with black text and white outline for classic film noir aesthetic, all uppercase",
        "font_paths": [
            "fonts/Uni Sans Heavy.otf",
            "Uni Sans Heavy.otf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (0, 0, 0, 255),  # Black text
        "highlight_colors": [  # Same black color for highlighted words
            (0, 0, 0, 255)     # Black
        ],
        "line_spacing": 45,  # Between 40-50px as specified
        "stroke_color": (255, 255, 255, 255),  # White stroke
        "stroke_width": 3,
        "shadow_color": (255, 255, 255, 180),  # White shadow with some transparency
        "shadow_offset": (4, 4),  # Shadow offset
        "uppercase": True  # Convert all text to uppercase
    },
    "Cinematic Movie": {
        "name": "Cinematic Movie",
        "description": "Futura Bold Oblique with warm orange text and black outline for cinematic movie titles",
        "font_paths": [
            "fonts/futura-pt-bold-oblique.otf",
            "futura-pt-bold-oblique.otf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (247, 161, 101, 255),  # Warm orange text (#F7A165)
        "highlight_colors": [  # Same warm orange color for highlighted words
            (247, 161, 101, 255)  # Warm orange (#F7A165)
        ],
        "line_spacing": 45,  # Between 40-50px as specified
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 3,
        "shadow_color": None,  # No shadow
        "shadow_offset": None,
        "uppercase": True  # Convert all text to uppercase
    },

    # ===== NEW CAPCUT-INSPIRED TEMPLATES =====
    
    
    
    
    "Luxury Gold": {
        "name": "Luxury Gold",
        "description": "Premium gold style with elegant serif font and shimmer effects",
        "font_paths": [
            "fonts/Playfair-Display-Black.ttf",
            "Playfair-Display-Black.ttf",
            "fonts/arial.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 66,
        "text_color": (255, 215, 0, 255),  # Gold
        "highlight_colors": [
            (255, 215, 0, 255),    # Gold
            (255, 165, 0, 255),    # Orange Gold
            (255, 140, 0, 255)     # Dark Orange
        ],
        "line_spacing": 48,
        "stroke_color": (139, 69, 19, 255),  # Saddle Brown
        "stroke_width": 2,
        "shadow_color": (0, 0, 0, 180),
        "shadow_offset": (3, 3),
        "luxury_style": True,
        "shimmer_effect": True
    },

    
    
    
    
    "Graffiti": {
        "name": "Graffiti",
        "description": "Street art graffiti style with spray paint effects and urban colors",
        "font_paths": [
            "fonts/Bangers-Regular.ttf",
            "Bangers-Regular.ttf",
            "fonts/arial.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 70,
        "text_color": (255, 255, 255, 255),  # White
        "highlight_colors": [
            (255, 20, 147, 255),   # Deep Pink
            (50, 205, 50, 255),    # Lime Green
            (255, 165, 0, 255),    # Orange
            (138, 43, 226, 255)    # Blue Violet
        ],
        "line_spacing": 48,
        "stroke_color": (0, 0, 0, 255),
        "stroke_width": 4,
        "shadow_color": (0, 0, 0, 180),
        "shadow_offset": (5, 5),
        "graffiti_style": True,
        "spray_effect": True,
        "uppercase": True
    },

    
    "Neon Pink": {
        "name": "Neon Pink",
        "description": "Vibrant neon pink with electric glow and modern sans-serif",
        "font_paths": [
            "fonts/Montserrat-Black.ttf",
            "Montserrat-Black.ttf",
            "fonts/arial.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 66,
        "text_color": (255, 20, 147, 255),  # Deep Pink
        "highlight_colors": [
            (255, 20, 147, 255),   # Deep Pink
            (255, 105, 180, 255),  # Hot Pink
            (255, 182, 193, 255)   # Light Pink
        ],
        "line_spacing": 44,
        "stroke_color": (139, 0, 139, 255),  # Dark Magenta
        "stroke_width": 2,
        "shadow_color": (255, 20, 147, 150),
        "shadow_offset": (0, 0),
        "neon_style": True,
        "glow_effect": True,
        "uppercase": True
    },

    
    
    
    
    
    
    
    


    "Hopecore": {
        "name": "Hopecore",
        "description": "Inspirational hopecore style with Classical Premiera Italic and Monotes fonts, pastel colors, and subtle glow effects",
        "font_paths": [
            "fonts/ClassicalPremieraItalic.otf",
            "ClassicalPremieraItalic.otf",
            "fonts/georgia.ttf",  # Fallback to Georgia
            "georgia.ttf",
            "C:/Windows/Fonts/georgia.ttf",
            "fonts/arial.ttf",  # Final fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 132,  # Closer to emphasized words font size
        "text_color": (245, 245, 245, 255),  # Soft off-white text
        "highlight_colors": [
            (135, 206, 235, 255),  # Sky blue (#87CEEB) - bright, hopeful sky blue
            (255, 255, 0, 255)     # Bright yellow (#FFFF00) - vibrant, optimistic yellow
        ],
        "emphasis_font_paths": [
            "fonts/Monotes.otf",
            "Monotes.otf",
            "fonts/georgiab.ttf",  # Fallback to Georgia Bold
            "georgiab.ttf",
            "C:/Windows/Fonts/georgiab.ttf"
        ],
        "emphasis_font_size": 135,  # Closer to normal words for more subtle emphasis
        "emphasis_font_weight": 600,  # Medium weight for emphasized words
        "has_highlighting": True,  # Enable word-by-word highlighting
        "has_real_highlighting": True,  # Use real highlighting system
        "use_emphasis_font": True,  # Use Monotes font for highlighted words
        "vertical_stack": True,  # Enable vertical word stacking
        "dynamic_positioning": True,  # Enable dynamic phrase positioning
        "sequential_word_reveal": True,  # Words appear at natural speech timestamps
        "line_spacing": 30,  # Comfortable spacing between wrapped text lines
        "stroke_color": (255, 255, 255, 180),  # Soft white stroke
        "stroke_width": 2,  # Subtle stroke for gentle outline
        "shadow_color": (200, 200, 200, 60),  # Very light gray shadow
        "shadow_offset": (2, 2),  # Small shadow offset for subtle depth
        "glow_color": (255, 255, 255, 80),  # Soft white glow effect
        "glow_radius": 8,  # Medium glow radius for subtle effect
        "has_glow": True,  # Enable glow effect
        "animation_type": "glow",  # Gentle glow animation for hopecore aesthetic
        "animation_speed": 1.2,  # Slower, more gentle animation
        "font_weight": 400,  # Regular weight for main text
        "outline_width": 2,  # 2px black outline for all text
        "outline_color": (0, 0, 0, 255),  # Black outline color
        "emphasis_stroke": False,  # Disable stroke effect for emphasized words
        "emphasis_stroke_color": (0, 0, 0, 0),  # No stroke for emphasized words
        "emphasis_stroke_width": 0,  # No stroke width for emphasized words
        "emphasis_brightness": 1.3,  # Make emphasized words brighter (30% increase)
        "hopecore_style": True,  # Special flag for hopecore processing
        "pastel_theme": True,  # Enable pastel color processing
        "gentle_effects": True,  # Enable gentle visual effects
        "max_emphasized_words": 2,  # Limit to maximum 2 emphasized words per phrase
        "vertical_padding": 60,  # Top and bottom padding to prevent text overflow
        "max_text_height": 0.8,  # Maximum text height as fraction of screen height (80%)
        "vertical_centering": True,  # Center text vertically within safe area
        "safe_area_top": 0.1,  # Top safe area as fraction of screen height (10%)
        "safe_area_bottom": 0.1  # Bottom safe area as fraction of screen height (10%)
    },
    
    "ChromaFusion": {
        "name": "ChromaFusion",
        "description": "Vibrant multi-color style with creative font combinations",
        "font_paths": [
            "fonts/Auromiya.ttf",  # Primary font
            "Auromiya.ttf",
            "fonts/PTF.ttf",  # Secondary font
            "PTF.ttf",
            "fonts/Bluefine.ttf",  # Tertiary font
            "Bluefine.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 132,  # Same as Hopecore
        "text_color": (138, 43, 226, 255),  # Vibrant Purple (#8A2BE2)
        "highlight_colors": [
            (255, 0, 255, 255),   # Magenta (#FF00FF)
            (0, 255, 255, 255),   # Cyan (#00FFFF)
            (255, 165, 0, 255),   # Orange (#FFA500)
            (50, 205, 50, 255)    # Lime Green (#32CD32)
        ],
        "emphasis_font_paths": [
            "fonts/PTF.ttf",  # Use PTF for emphasized words
            "PTF.ttf",
            "fonts/Bluefine.ttf",  # Fallback to Bluefine
            "Bluefine.ttf",
            "fonts/SandeMore.otf",  # Then SandeMore
            "SandeMore.otf",
            "fonts/arial.ttf",  # Final fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "emphasis_font_size": 135,  # Same as Hopecore
        "emphasis_font_weight": 600,  # Medium weight for emphasized words
        "has_highlighting": True,  # Enable word-by-word highlighting
        "has_real_highlighting": False,  # Use color-based highlighting system
        "use_emphasis_font": True,  # Use different font for highlighted words
        "vertical_stack": True,  # Enable vertical word stacking
        "dynamic_positioning": True,  # Enable dynamic phrase positioning
        "sequential_word_reveal": True,  # Words appear at natural speech timestamps
        "line_spacing": 30,  # Comfortable spacing between wrapped text lines
        "stroke_color": (255, 255, 255, 180),  # Soft white stroke
        "stroke_width": 2,  # Subtle stroke for gentle outline
        "shadow_color": (0, 0, 0, 100),  # Light shadow
        "shadow_offset": (2, 2),  # Small shadow offset for subtle depth
        "glow_color": (255, 255, 255, 80),  # Soft white glow effect
        "glow_radius": 8,  # Medium glow radius for subtle effect
        "has_glow": True,  # Enable glow effect
        "animation_type": "glow",  # Gentle glow animation
        "animation_speed": 1.2,  # Slower, more gentle animation
        "font_weight": 400,  # Regular weight for main text
        "outline_width": 2,  # 2px outline for all text
        "outline_color": (0, 0, 0, 255),  # Black outline color
        "emphasis_stroke": False,  # Disable stroke effect for emphasized words
        "emphasis_stroke_color": (0, 0, 0, 0),  # No stroke for emphasized words
        "emphasis_stroke_width": 0,  # No stroke width for emphasized words
        "emphasis_brightness": 1.3,  # Make emphasized words brighter (30% increase)
        "hopecore_style": True,  # Special flag for hopecore processing
        "pastel_theme": True,  # Enable pastel color processing
        "gentle_effects": True,  # Enable gentle visual effects
        "max_emphasized_words": 2,  # Limit to maximum 2 emphasized words per phrase
        "emphasis_word_spacing": 15,  # Extra spacing around emphasized words (pixels)
        "vertical_padding": 60,  # Top and bottom padding to prevent text overflow
        "max_text_height": 0.8,  # Maximum text height as fraction of screen height (80%)
        "vertical_centering": True,  # Center text vertically within safe area
        "safe_area_top": 0.1,  # Top safe area as fraction of screen height (10%)
        "safe_area_bottom": 0.1  # Bottom safe area as fraction of screen height (10%)
    },
    
    "Smilecore": {
        "name": "Smilecore",
        "description": "Cheerful and uplifting style with bright colors and positive vibes",
        "font_paths": [
            "fonts/Droid Italic.ttf",  # Use Droid Italic for normal words
            "Droid Italic.ttf",
            "fonts/NotoSansJP-Italic.ttf",  # Fallback to previous fonts
            "NotoSansJP-Italic.ttf",
            "fonts/NotoSansJP-Regular.ttf",
            "NotoSansJP-Regular.ttf",
            "fonts/georgia.ttf",
            "georgia.ttf",
            "C:/Windows/Fonts/georgia.ttf",
            "fonts/arial.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 132,  # Same as Hopecore - closer to emphasized words font size
        "text_color": (245, 245, 245, 255),  # Soft off-white text
        "highlight_colors": [
            (255, 215, 0, 255),   # Gold (#FFD700) - cheerful gold color
            (255, 165, 0, 255)    # Orange (#FFA500) - warm orange for variety
        ],
        "emphasis_font_paths": [
            "fonts/grained.ttf",
            "grained.ttf",
            "fonts/Monotes.otf",  # Fallback to Monotes
            "Monotes.otf",
            "fonts/georgiab.ttf",  # Fallback to Georgia Bold
            "georgiab.ttf",
            "C:/Windows/Fonts/georgiab.ttf"
        ],
        "emphasis_font_size": 135,  # Same as Hopecore - closer to normal words for more subtle emphasis
        "emphasis_font_weight": 600,  # Medium weight for emphasized words
        "has_highlighting": True,  # Enable word-by-word highlighting
        "has_real_highlighting": True,  # Use real highlighting system
        "use_emphasis_font": True,  # Use Monotes font for highlighted words
        "vertical_stack": True,  # Enable vertical word stacking
        "dynamic_positioning": True,  # Enable dynamic phrase positioning
        "sequential_word_reveal": True,  # Words appear at natural speech timestamps
        "line_spacing": 30,  # Comfortable spacing between wrapped text lines
        "stroke_color": (255, 255, 255, 180),  # Soft white stroke
        "stroke_width": 2,  # Subtle stroke for gentle outline
        "shadow_color": (200, 200, 200, 60),  # Very light gray shadow
        "shadow_offset": (2, 2),  # Small shadow offset for subtle depth
        "glow_color": (255, 255, 255, 80),  # Soft white glow effect
        "glow_radius": 8,  # Medium glow radius for subtle effect
        "has_glow": True,  # Enable glow effect
        "animation_type": "glow",  # Gentle glow animation
        "animation_speed": 1.2,  # Slower, more gentle animation
        "font_weight": 400,  # Regular weight for main text
        "outline_width": 2,  # 2px black outline for all text
        "outline_color": (0, 0, 0, 255),  # Black outline color
        "emphasis_stroke": False,  # Disable stroke effect for emphasized words
        "emphasis_stroke_color": (0, 0, 0, 0),  # No stroke for emphasized words
        "emphasis_stroke_width": 0,  # No stroke width for emphasized words
        "emphasis_brightness": 1.3,  # Make emphasized words brighter (30% increase)
        "smilecore_style": True,  # Special flag for smilecore processing
        "pastel_theme": True,  # Enable pastel color processing
        "gentle_effects": True,  # Enable gentle visual effects
        "max_emphasized_words": 2,  # Limit to maximum 2 emphasized words per phrase
        "vertical_padding": 60,  # Top and bottom padding to prevent text overflow
        "max_text_height": 0.8,  # Maximum text height as fraction of screen height (80%)
        "vertical_centering": True,  # Center text vertically within safe area
        "safe_area_top": 0.1,  # Top safe area as fraction of screen height (10%)
        "safe_area_bottom": 0.1  # Bottom safe area as fraction of screen height (10%)
    },

    "Golden Impact": {
        "name": "Golden Impact",
        "description": "Bold word-by-word style with golden yellow text and black stroke",
        "font_paths": [
            "fonts/AntonSC-Regular.ttf",
            "AntonSC-Regular.ttf",
            "fonts/Anton-Regular.ttf",  # Fallback to regular Anton if SC not found
            "Anton-Regular.ttf",
            "fonts/arial.ttf",  # Final fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 100,
        "text_color": (255, 255, 0, 255),  # Yellow text
        "highlight_colors": [],  # No highlight colors for word-by-word
        "line_spacing": 50,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 2,  # 2px stroke as requested
        "shadow_color": (0, 0, 0, 0),  # No shadow
        "shadow_offset": (0, 0),
        "uppercase": True,  # Convert all text to uppercase
        "has_highlighting": False,  # No highlighting for word-by-word
        "has_real_highlighting": False,  # No real highlighting
        "uses_text_color": True,  # Use unified text color
        "word_by_word": True,  # Enable word-by-word display
        "sequential_word_reveal": True,  # Words appear at natural speech timestamps
        "vertical_stack": False,  # Don't stack vertically
        "dynamic_positioning": False,  # Use standard positioning
        "font_weight": 900,  # Extra bold weight
        "outline_width": 2,  # 2px black outline
        "outline_color": (0, 0, 0, 255),  # Black outline color
        "anton_word_flow_style": True  # Special flag for this template
    },

    "Blue Streak": {
        "name": "Blue Streak",
        "description": "Modern phrase-by-phrase style with white text, black stroke and blue highlight bars",
        "font_paths": [
            "D:/GPU quickcap/backend/fonts/Uni Sans Heavy.otf",  # Primary font path
            "fonts/Uni Sans Heavy.otf",  # Relative path
            "fonts/UniSansHeavy.ttf",
            "UniSansHeavy.ttf",
            "fonts/UniSans-Heavy.ttf",  # Alternative naming
            "UniSans-Heavy.ttf",
            "fonts/arial.ttf",  # Final fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,
        "text_color": (255, 255, 255, 255),  # White text
        "highlight_colors": [  # Blue highlight bars that fill the words
            (0, 104, 219, 230)   # #0068DB with 230 opacity for semi-transparent bars
        ],
        "line_spacing": 40,
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 2,  # 2px stroke as requested
        "shadow_color": (0, 0, 0, 0),  # No shadow
        "shadow_offset": (0, 0),
        "highlight_bars": True,  # Enable highlight bars like explainer pro
        "bar_padding": 15,  # Increased padding around text for highlight bars
        "has_highlighting": True,  # Enable highlighting
        "has_real_highlighting": True,  # Use real highlighting system
        "uses_text_color": True,  # Use unified text color
        "word_by_word": False,  # Phrase by phrase display
        "sequential_word_reveal": False,  # Standard phrase timing
        "vertical_stack": False,  # Don't stack vertically
        "dynamic_positioning": False,  # Use standard positioning
        "font_weight": 900,  # Heavy weight
        "outline_width": 2,  # 2px black outline
        "outline_color": (0, 0, 0, 255),  # Black outline color
        "uni_sans_pro_style": True  # Special flag for this template
    },

    "Ocean Waves": {
        "name": "Ocean Waves",
        "description": "Big Fish Casuals font with gradient blue and cyan colors, scale animation for word-by-word captions",
        "font_paths": [
            "D:/GPU quickcap/backend/fonts/Big Fish Casuals.otf",  # Primary font path
            "fonts/Big Fish Casuals.otf",  # Relative path
            "Big Fish Casuals.otf",
            "fonts/arial.ttf",  # Final fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 120,  # Much larger base font size (120px)
        "text_color": (14, 137, 194, 255),  # Default blue color
        "text_color_cycle": [  # Cycling text colors
            (14, 137, 194, 255),  # #0E89C2 - Blue
            (2, 218, 255, 255),   # #02DAFF - Cyan
            (254, 67, 69, 255),   # #FE4345 - Red
            (255, 67, 69, 255)    # #FF4345 - Red (slightly different)
        ],
        "line_spacing": 60,  # Increased line spacing for larger font
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 4,  # 4px stroke for better readability with larger font
        "shadow_color": (0, 0, 0, 128),  # Semi-transparent black shadow
        "shadow_offset": (4, 4),  # Shadow offset
        "glow_color": (255, 255, 255, 80),  # White glow behind text
        "glow_radius": 8,  # Glow radius in pixels
        "has_glow": True,  # Enable glow effect
        "has_light": True,  # Enable light effect based on text color
        "light_radius": 15,  # Light radius in pixels
        "light_opacity": 60,  # Light opacity (0-255)
        "has_highlighting": False,  # Disable highlighting - use text colors instead
        "has_real_highlighting": False,  # Don't use highlighting system
        "uses_text_color": True,  # Use text color system with cycling colors
        "cycle_text_colors": True,  # Enable text color cycling
        "word_by_word": True,  # Enable word-by-word display
        "sequential_word_reveal": True,  # Words appear at natural speech timestamps
        "vertical_stack": False,  # Don't stack vertically
        "dynamic_positioning": False,  # Use standard positioning
        "scale_effect": True,  # Enable scale effect for highlighted words
        "scale_factor": 1.3,  # Scale highlighted words by 30% (makes them bigger)
        "enhanced_font_size": 150,  # Even larger font size for word-by-word mode
        "ignore_custom_colors": True,  # Don't allow custom color overrides
        "ignore_custom_font_size": True,  # Don't allow custom font size overrides
        "has_multiple_colors": True,  # Template uses multiple colors that cycle
        "preserve_template_settings": True  # Preserve all template-specific settings
    },

    "Crimson Streak": {
        "name": "Crimson Streak",
        "description": "Anton Regular font with crimson red streak highlighting, identical to Blue Streak but with #ED2C52 color",
        "font_paths": [
            "fonts/Anton-Regular.ttf",
            "Anton-Regular.ttf",
            "fonts/arial.ttf",  # Fallback
            "C:/Windows/Fonts/arial.ttf"
        ],
        "font_size": 65,  # Match Blue Streak font size
        "text_color": (255, 255, 255, 255),  # White text
        "highlight_colors": [  # Crimson red highlight bars that fill the words
            (237, 44, 82, 230)   # #ED2C52 with 230 opacity for semi-transparent bars
        ],
        "line_spacing": 40,  # Match Blue Streak line spacing
        "stroke_color": (0, 0, 0, 255),  # Black stroke
        "stroke_width": 2,  # Match Blue Streak stroke width
        "shadow_color": (0, 0, 0, 0),  # No shadow - match Blue Streak exactly
        "shadow_offset": (0, 0),  # Match Blue Streak shadow offset
        "highlight_bars": True,  # Enable highlight bars like Blue Streak
        "bar_padding": 15,  # Match Blue Streak padding exactly
        "has_highlighting": True,  # Enable highlighting
        "has_real_highlighting": True,  # Use real highlighting system
        "uses_text_color": True,  # Use unified text color
        "word_by_word": False,  # Phrase by phrase display
        "sequential_word_reveal": False,  # Standard phrase timing
        "vertical_stack": False,  # Don't stack vertically
        "dynamic_positioning": False,  # Use standard positioning
        "font_weight": 900,  # Heavy weight for Anton font
        "outline_width": 2,  # Match Blue Streak outline width
        "outline_color": (0, 0, 0, 255),  # Black outline color
        "uppercase": True,  # Convert all text to uppercase
        "crimson_streak_style": True  # Special flag for this template
    },

    }

# Current template (can be changed)
CURRENT_TEMPLATE = "MrBeast"

def get_font(template_name=None, word_by_word_mode=False, custom_template=None):
    """Get the best available font for the specified template"""
    if template_name is None:
        template_name = CURRENT_TEMPLATE
    
    # Use the provided custom template if available, otherwise get from CAPTION_TEMPLATES
    if custom_template is not None:
        template = custom_template
    else:
        template = CAPTION_TEMPLATES.get(template_name, CAPTION_TEMPLATES["MrBeast"])
    font_paths = template["font_paths"]
    
    # Use enhanced font size for word-by-word mode if available
    if word_by_word_mode and template.get("word_by_word", False):
        font_size = template.get("enhanced_font_size", template["font_size"])
    else:
        # Make sure to use the template's font size, not the default 65px
        # Special handling for Smilecore, Hopelesscore, and ChromaFusion to ensure they use their defined font sizes
        if template_name in ["Smilecore", "Hopelesscore", "ChromaFusion"] and "font_size" in template:
            font_size = template["font_size"]  # Use the template's defined font size (132px)
        else:
            font_size = template.get("font_size", 65)  # Default to 65px for other templates
    
    logger.debug(f"Loading font for template: {template_name}, size: {font_size}")
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                logger.info(f"[OK] Font loaded: {font_path} (size: {font_size})")
                return ImageFont.truetype(font_path, font_size)
        except (OSError, IOError) as e:
            logger.debug(f"Failed to load font {font_path}: {e}")
            continue
    
    # Fallback to default font
    try:
        logger.warning(f"[WARNING] Using default font fallback (size: {font_size})")
        return ImageFont.load_default()
    except Exception as e:
        logger.error(f"⚠️  Warning: Could not load any font: {e}")
        return ImageFont.load_default()

# Helper: Split into phrases of 6 words, 3 words for single line, or 5 words for vertical stack
def chunk_words(words, caption_layout="wrapped", template_name=None):
    phrases = []
    phrase = []
    
    # Determine words per chunk based on layout and template
    if caption_layout == "single_line":
        words_per_chunk = 3
    elif template_name == "Smilecore":
        words_per_chunk = 5  # Limit to 5 words for Smilecore template
    elif template_name in ["Hopelesscore", "ChromaFusion"]:
        words_per_chunk = 5  # Limit to 5 words for Hopelesscore and ChromaFusion templates
    else:
        words_per_chunk = WORDS_PER_PHRASE
    
    for word in words:
        phrase.append(word)
        if len(phrase) == words_per_chunk:
            phrases.append(phrase)
            phrase = []
    if phrase:
        phrases.append(phrase)
    return phrases

# Helper function to get text size (compatible with newer Pillow versions)
def get_text_size(draw, text, font):
    """Get text width and height, compatible with both old and new Pillow versions"""
    try:
        # Try new method first (Pillow 10.0.0+)
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]  # width, height
    except AttributeError:
        # Fallback to old method for older Pillow versions
        return draw.textsize(text, font=font)

# Helper function to draw text with stroke, shadow, and glow
def draw_text_with_stroke(draw, position, text, font, text_color, stroke_color=None, stroke_width=0, shadow_color=None, shadow_offset=None, glow_color=None, glow_radius=0, light_color=None, light_radius=0):
    """Draw text with optional shadow, stroke/outline, and glow effect"""
    x, y = position
    
    # Draw glow effect first (behind everything)
    if glow_color and glow_radius > 0:
        # Create multiple layers of glow with decreasing opacity
        for radius in range(glow_radius, 0, -1):
            # Calculate opacity that decreases with distance from center
            opacity_factor = (glow_radius - radius + 1) / glow_radius
            glow_alpha = int(glow_color[3] * opacity_factor * 0.3)  # Reduce overall glow intensity
            current_glow_color = (glow_color[0], glow_color[1], glow_color[2], glow_alpha)
            
            # Draw glow in a circle pattern around the text
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only draw if within circular distance
                    distance = (dx * dx + dy * dy) ** 0.5
                    if distance <= radius and distance > 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=current_glow_color)

    # Draw light effect (colored light behind text based on text color)
    if light_color and light_radius > 0:
        # Create multiple layers of light with decreasing opacity
        for radius in range(light_radius, 0, -1):
            # Calculate opacity that decreases with distance from center
            opacity_factor = (light_radius - radius + 1) / light_radius
            light_alpha = int(light_color[3] * opacity_factor * 0.4)  # Light intensity
            current_light_color = (light_color[0], light_color[1], light_color[2], light_alpha)
            
            # Draw light in a circle pattern around the text
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only draw if within circular distance
                    distance = (dx * dx + dy * dy) ** 0.5
                    if distance <= radius and distance > 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=current_light_color)

    # Handle special shadow lighting that matches text color
    actual_shadow_color = shadow_color
    if shadow_color == "match_text":
        # Create a darker version of the text color for shadow lighting
        r, g, b, a = text_color
        # Reduce brightness by 40% and add some transparency
        actual_shadow_color = (int(r * 0.6), int(g * 0.6), int(b * 0.6), int(a * 0.8))
    
    # Draw shadow first (behind everything)
    if actual_shadow_color and shadow_offset:
        shadow_x = x + shadow_offset[0]
        shadow_y = y + shadow_offset[1]
        
        # Draw shadow stroke if stroke is enabled
        if stroke_color and stroke_width > 0:
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((shadow_x + dx, shadow_y + dy), text, font=font, fill=actual_shadow_color)
        
        # Draw shadow text
        draw.text((shadow_x, shadow_y), text, font=font, fill=actual_shadow_color)
    
    # Draw stroke around main text
    if stroke_color and stroke_width > 0:
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx != 0 or dy != 0:  # Don't draw at the center position yet
                    draw.text((x + dx, y + dy), text, font=font, fill=stroke_color)
    
    # Draw the main text on top
    draw.text((x, y), text, font=font, fill=text_color)

# Wrap caption text and render PNG with word highlighting
def render_caption_png_wrapped(text, output_path, highlight_word_index=None, template_name=None, custom_template=None, caption_layout="wrapped", profanity_processor=None, word_info=None, line_height_multiplier=1.2, word_spacing_px=0):
    if template_name is None:
        template_name = CURRENT_TEMPLATE
    
    # Use the provided custom template if available, otherwise get from CAPTION_TEMPLATES
    if custom_template is not None:
        template = custom_template
    else:
        template = CAPTION_TEMPLATES.get(template_name, CAPTION_TEMPLATES["MrBeast"])
    
    # Log font size for debugging
    font_size = template.get("font_size", 65)
    logger.info(f"[FONT] Rendering caption: '{text[:30]}{'...' if len(text) > 30 else ''}' with template: {template_name}, font size: {font_size}px")
    
    # Check for Hopelesscore template features
    is_hopelesscore = template.get("hopelesscore_style", False)
    # For backward compatibility, also check for hopecore_style
    if not is_hopelesscore:
        is_hopelesscore = template.get("hopecore_style", False)
    
    random_vertical = template.get("random_vertical_position", False)
    word_sequence_mode = template.get("word_sequence_mode", False)
    
    # Define is_hopecore variable based on template name or style flag
    # Handle both cases where template is named Hopecore/Hopelesscore or has the style flag
    is_hopecore = (template_name == "Hopecore" or template_name == "Hopelesscore" or 
                  template.get("hopecore_style", False) or template.get("hopelesscore_style", False))
    
    if is_hopecore:
        logger.info(f"[HOPECORE] Processing with Hopecore template features: random_vertical={random_vertical}, word_sequence_mode={word_sequence_mode}")
    
    # Convert text case if template requires it
    if template.get("uppercase", False):
        text = text.upper()
    elif template.get("title_case", False):
        text = text.title()
    elif template.get("lowercase_casual", False):
        # For casual lowercase effect, randomly make some words lowercase
        # Keep first word capitalized, randomly lowercase others with 40% probability
        words = text.split()
        if len(words) > 1:
            words[0] = words[0].capitalize()  # Always capitalize first word
            for i in range(1, len(words)):
                # Skip short words and randomly apply lowercase
                if len(words[i]) > 3 and random.random() < 0.4:
                    words[i] = words[i].lower()
                else:
                    words[i] = words[i].capitalize()
            text = ' '.join(words)
    
    # Check if this is word-by-word mode (for single word rendering)
    is_word_by_word = template.get("word_by_word", False) and len(text.split()) == 1
    font = get_font(template_name, word_by_word_mode=is_word_by_word, custom_template=template)
    text_color = template.get("text_color", (255, 255, 255, 255))  # Default to white if not specified
    # Support both single highlight_color and multiple highlight_colors
    highlight_colors = template.get("highlight_colors", [template.get("highlight_color")] if template.get("highlight_color") else [])
    base_line_spacing = template["line_spacing"]
    line_spacing = int(base_line_spacing * line_height_multiplier)
    logger.info(f"[SPACING] Line spacing: {base_line_spacing}px (base) * {line_height_multiplier} = {line_spacing}px")
    logger.info(f"[MEASURE] Word spacing: {word_spacing_px}px extra between words")
    stroke_color = template.get("stroke_color")
    stroke_width = template.get("stroke_width", 0)
    shadow_color = template.get("shadow_color")
    shadow_offset = template.get("shadow_offset")
    glow_color = template.get("glow_color")
    glow_radius = template.get("glow_radius", 0)
    light_radius = template.get("light_radius", 0)
    light_opacity = template.get("light_opacity", 60)
    has_light = template.get("has_light", False)
    
    # Log glow effect status
    if template.get("has_glow", False) and glow_color and glow_radius > 0:
        logger.info(f"[GLOW] Applying glow effect: color={glow_color}, radius={glow_radius}px")
    
    # Log light effect status
    if has_light and light_radius > 0:
        logger.info(f"[LIGHT] Applying light effect: radius={light_radius}px, opacity={light_opacity}")
    
    # Helper function to create light color based on text color
    def create_light_color(text_color, opacity):
        """Create a light color based on the text color with specified opacity"""
        if text_color and len(text_color) >= 3:
            return (text_color[0], text_color[1], text_color[2], opacity)
        return None
    
    image = Image.new("RGBA", (VIDEO_WIDTH, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    words = text.split()
    lines = []
    line_words = []
    line = ""
    
    # Track which words are on which lines
    word_line_mapping = []
    current_line_index = 0
    
    # Initialize max_lines for all caption layouts
    max_lines = MAX_LINES
    
    # Handle single-line captions differently
    if caption_layout == "single_line":
        # For single-line captions, we'll show exactly one line
        # with up to three words at a time
        
        # Calculate maximum width (80% of screen width)
        max_width_percent = 0.8
        max_single_line_width = int(VIDEO_WIDTH * max_width_percent)
        
        # Only add one line - we'll use all words provided
        # (should be max 3 words from the chunking logic)
        if len(words) > 0:
            lines.append(" ".join(words))
            
            # Map all words to line 0
            for i in range(len(words)):
                word_line_mapping.append(0)
        else:
            # If there are no words, add an empty line
            lines.append("")
    else:
        # Standard wrapped text processing with configurable line limit
        logger.info(f"[WORDS] Wrapping text with max {max_lines} lines: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Build lines with word wrapping
        current_line = ""
        current_line_words = []
        
        for word_idx, word in enumerate(words):
            test_line = f"{current_line} {word}".strip()
            w, _ = get_text_size(draw, test_line, font)
            
            if w <= MAX_WIDTH:
                # Word fits on current line
                current_line = test_line
                current_line_words.append(word_idx)
            else:
                # Word doesn't fit, need to wrap
                if current_line:
                    # Add current line to lines if we haven't reached max lines
                    if len(lines) < max_lines:
                        lines.append(current_line)
                        # Map words to current line
                        for w_idx in current_line_words:
                            word_line_mapping.append(len(lines) - 1)
                        
                        # Start new line with current word
                        current_line = word
                        current_line_words = [word_idx]
                    else:
                        # Already at max lines, try to fit more words on the last line
                        if lines:
                            # Try to add remaining words to the last line
                            remaining_words = words[word_idx:]
                            last_line = lines[-1]
                            
                            # Try to fit as many remaining words as possible
                            for remaining_word in remaining_words:
                                test_last_line = f"{last_line} {remaining_word}".strip()
                                w_test, _ = get_text_size(draw, test_last_line, font)
                                if w_test <= MAX_WIDTH:
                                    last_line = test_last_line
                                    word_line_mapping.append(len(lines) - 1)
                                else:
                                    # Can't fit more words, stop
                                    break
                            
                            # Update the last line
                            lines[-1] = last_line
                        break
                else:
                    # No previous line, start with current word
                    current_line = word
                    current_line_words.append(word_idx)
        
        # Add the last line if it exists and we haven't exceeded the line limit
        if current_line and len(lines) < max_lines:
            lines.append(current_line)
            for w_idx in current_line_words:
                word_line_mapping.append(len(lines) - 1)

    # Log final line count
    logger.info(f"[STATS] Caption wrapped into {len(lines)} lines (max allowed: {max_lines})")
    if len(lines) > 0:
        for i, line in enumerate(lines):
            logger.info(f"   Line {i+1}: '{line}'")

    total_height = sum([get_text_size(draw, l, font)[1] + line_spacing for l in lines])
    
    # Calculate y_start position
    if is_hopecore and random_vertical:
        # For Hopecore with random vertical positioning, use a random position within safe area
        safe_top = template.get("safe_area_top", 0.1)  # Default 10% from top
        safe_bottom = template.get("safe_area_bottom", 0.8)  # Default 80% from top
        
        # Calculate safe area in pixels (within the 300px height)
        safe_top_px = int(300 * safe_top)
        safe_bottom_px = int(300 * safe_bottom)
        
        # Calculate available space for positioning
        available_space = safe_bottom_px - safe_top_px - total_height
        
        if available_space > 0:
            # Use position zones from template or default to all zones
            position_zones = template.get("position_zones", [
                "top_left", "top_center", "top_right",
                "center_left", "center", "center_right",
                "bottom_left", "bottom_center", "bottom_right"
            ])
            
            # Select a random position zone
            selected_zone = random.choice(position_zones)
            logger.info(f"[HOPECORE] Selected position zone: {selected_zone}")
            
            # Store the selected zone in a new key in the template dictionary for horizontal positioning to use
            template["_position_zone"] = selected_zone
            
            # Calculate position based on selected zone
            if "top" in selected_zone:
                y_start = safe_top_px + int(available_space * 0.1)  # Near top of safe area
            elif "bottom" in selected_zone:
                y_start = safe_bottom_px - total_height - int(available_space * 0.1)  # Near bottom of safe area
            else:  # center
                y_start = safe_top_px + (available_space // 2)  # Middle of safe area
                
            logger.info(f"[HOPECORE] Using positioned vertical placement: {y_start}px (zone: {selected_zone}, safe area: {safe_top_px}-{safe_bottom_px}px)")
        else:
            # Fallback to centered if not enough space
            y_start = (300 - total_height) // 2
            template["_position_zone"] = "center"
            logger.info(f"[HOPECORE] Not enough space for positioned placement, using centered: {y_start}px")
    else:
        # Standard centered positioning
        y_start = (300 - total_height) // 2
        if is_hopecore:
            template["_position_zone"] = "center"

    # Render each line
    for line_idx, line in enumerate(lines):
        line_words = line.split()
        w, h = get_text_size(draw, line, font)
        
        # For Hopecore template, also randomize horizontal position if needed
        if is_hopecore and random_vertical:
            # Get the position zone that was selected for vertical positioning
            position_zone = template.get("_position_zone", "center")
            
            # Determine horizontal position based on zone
            if "left" in position_zone:
                # Left align with padding
                padding = int(VIDEO_WIDTH * 0.1)  # 10% padding from left
                x_start = padding
                logger.info(f"[HOPECORE] Using left-aligned position: {x_start}px")
            elif "right" in position_zone:
                # Right align with padding
                padding = int(VIDEO_WIDTH * 0.1)  # 10% padding from right
                x_start = VIDEO_WIDTH - w - padding
                logger.info(f"[HOPECORE] Using right-aligned position: {x_start}px")
            else:
                # Center align (default)
                x_start = (VIDEO_WIDTH - w) // 2
                logger.info(f"[HOPECORE] Using center-aligned position: {x_start}px")
        else:
            # Standard centered positioning
            x_start = (VIDEO_WIDTH - w) // 2
            
        y = y_start + line_idx * (h + line_spacing)
        
        # Apply word sequence mode for Hopecore template if enabled
        if is_hopecore and word_sequence_mode and highlight_word_index is not None:
            # In word sequence mode, we only show words up to the current highlighted word
            # This creates a sequential reveal effect where words appear one by one
            
            # Find which line contains the highlighted word
            highlighted_line = None
            for i, word_line in enumerate(word_line_mapping):
                if i == highlight_word_index:
                    highlighted_line = word_line
                    break
            
            # If this line is after the highlighted line, don't render it at all
            if highlighted_line is not None and line_idx > highlighted_line:
                logger.info(f"[HOPECORE] Word sequence mode: Skipping line {line_idx} (after highlighted line {highlighted_line})")
                continue
            
            # If this is the highlighted line, we may need to truncate it
            if highlighted_line is not None and line_idx == highlighted_line:
                # Find which word in this line corresponds to the highlighted word
                word_position_in_line = 0
                words_in_previous_lines = 0
                
                # Count words in previous lines
                for i in range(line_idx):
                    if i < len(lines):
                        words_in_previous_lines += len(lines[i].split())
                
                # Calculate position in current line
                word_position_in_line = highlight_word_index - words_in_previous_lines
                
                # Truncate line to only show words up to the highlighted word
                if word_position_in_line >= 0 and word_position_in_line < len(line_words):
                    truncated_line_words = line_words[:word_position_in_line + 1]
                    truncated_line = " ".join(truncated_line_words)
                    
                    # Recalculate dimensions and position for the truncated line
                    w, h = get_text_size(draw, truncated_line, font)
                    x_start = (VIDEO_WIDTH - w) // 2
                    
                    # Update line and line_words for rendering
                    line = truncated_line
                    line_words = truncated_line_words
                    
                    logger.info(f"[HOPECORE] Word sequence mode: Truncated line {line_idx} to '{truncated_line}'")
        
        
        # If no highlighting or highlight not on this line, render normally
        if not highlight_colors or highlight_word_index is None:
            # Create light color for this text if light effect is enabled
            light_color = create_light_color(text_color, light_opacity) if has_light else None
            draw_text_with_stroke(draw, (x_start, y), line, font, text_color, stroke_color, stroke_width, shadow_color, shadow_offset, glow_color, glow_radius, light_color, light_radius)
        else:
            # Render word by word with highlighting
            current_x = x_start
            word_index_in_text = 0
            
            # Find the starting word index for this line
            for i in range(len(word_line_mapping)):
                if word_line_mapping[i] == line_idx:
                    word_index_in_text = i
                    break
            
            for word_idx_in_line, word in enumerate(line_words):
                current_word_index = word_index_in_text + word_idx_in_line
                
                # Check if this word is profanity and should be highlighted in red
                is_profanity = False
                is_emoji = False
                
                # Check word_info for both profanity and emoji flags
                if word_info:
                    # Find the corresponding word info for this word
                    for w_info in word_info:
                        if w_info.get('word', '').strip() == word.strip():
                            # Check for profanity if processor is available
                            if profanity_processor:
                                is_profanity = profanity_processor.should_highlight_as_profanity(w_info)
                            # Check if this word is an emoji
                            is_emoji = w_info.get('is_emoji', False)
                            if is_emoji:
                                print(f"[DEBUG] Detected emoji word: '{word}' with is_emoji flag")
                            break
                
                # Special handling for emoji words
                if is_emoji:
                    # Use a larger font size for emojis
                    emoji_scale = 1.5  # Make emojis 50% larger
                    emoji_font_size = int(font_size * emoji_scale)
                    emoji_font = get_font(template_name, font_size=emoji_font_size, word_by_word_mode=is_word_by_word, custom_template=template)
                    
                    # Calculate vertical offset to center the emoji
                    regular_height = get_text_size(draw, word, font)[1]
                    emoji_height = get_text_size(draw, word, emoji_font)[1]
                    y_offset = (emoji_height - regular_height) // 2
                    
                    # Draw the emoji without any effects (no stroke, shadow, etc.)
                    draw.text((current_x, y - y_offset), word, font=emoji_font, fill=text_color)
                    
                    # Get the width for positioning the next word
                    word_width, _ = get_text_size(draw, word + " ", emoji_font)
                    
                    # Log emoji rendering
                    print(f"[EMOJI] Rendered emoji: {word} at position ({current_x}, {y - y_offset})")
                
                # Regular word rendering (non-emoji)
                elif current_word_index == highlight_word_index or is_profanity or template.get("highlight_all_words", False):
                    # Check if this template uses highlight bars
                    if template.get("highlight_bars", False):
                        # Draw highlight bar behind the word
                        color_index = current_word_index % len(highlight_colors)  # Use current_word_index for color cycling
                        bar_color = highlight_colors[color_index]
                        bar_padding = template.get("bar_padding", 8)
                        
                        # Calculate word dimensions
                        word_width_no_space, word_height = get_text_size(draw, word, font)
                        
                        # Draw rounded rectangle behind the word (properly aligned with text baseline)
                        bar_x1 = current_x - bar_padding
                        bar_y1 = y - bar_padding
                        bar_x2 = current_x + word_width_no_space + bar_padding
                        bar_y2 = y + word_height + bar_padding
                        
                        # Create a temporary image for the rounded rectangle with transparency
                        bar_img = Image.new("RGBA", (bar_x2 - bar_x1, bar_y2 - bar_y1), (0, 0, 0, 0))
                        bar_draw = ImageDraw.Draw(bar_img)
                        
                        # Draw rounded rectangle
                        corner_radius = 8
                        bar_draw.rounded_rectangle(
                            (0, 0, bar_x2 - bar_x1, bar_y2 - bar_y1),
                            radius=corner_radius,
                            fill=bar_color
                        )
                        
                        # Paste the bar onto the main image
                        image.paste(bar_img, (bar_x1, bar_y1), bar_img)
                        
                        # Draw text at the original baseline position (not centered in bar)
                        light_color = create_light_color(text_color, light_opacity) if has_light else None
                        draw_text_with_stroke(draw, (current_x, y), word, font, text_color, stroke_color, stroke_width, shadow_color, shadow_offset, glow_color, glow_radius, light_color, light_radius)
                        word_width, _ = get_text_size(draw, word + " ", font)
                    else:
                        # Use red for profanity, otherwise use cycling colors for highlighted words
                        if is_profanity and profanity_processor:
                            color = profanity_processor.get_profanity_color()
                        else:
                            color_index = current_word_index % len(highlight_colors)  # Use current_word_index for color cycling
                            
                            # Handle dynamic text color cycling for Dynamic Proxima template
                            if template.get("highlight_cycle_mode") == "dynamic_text":
                                # For Dynamic Proxima: Red highlight -> Green highlight -> Gold text (no highlight)
                                if color_index == 2:  # Third cycle: Gold text, no highlight
                                    # Use gold as text color and don't highlight
                                    color = text_color  # Use current text color (will be overridden below)
                                    # Override text color for this word to gold
                                    text_color_cycle = template.get("text_color_cycle", [text_color])
                                    if len(text_color_cycle) > color_index:
                                        color = text_color_cycle[color_index]
                                    else:
                                        color = highlight_colors[color_index]
                                else:
                                    # First two cycles: use highlight colors normally
                                    color = highlight_colors[color_index]
                            elif template.get("cycle_text_colors"):
                                # Handle Ocean Waves text color cycling
                                text_color_cycle = template.get("text_color_cycle", [text_color])
                                if text_color_cycle:
                                    color_index = current_word_index % len(text_color_cycle)
                                    color = text_color_cycle[color_index]
                                    print(f"[DEBUG] Ocean Waves - Word: '{word}', Index: {current_word_index}, Color Index: {color_index}, Color: {color}")
                                else:
                                    color = text_color
                            elif template.get("random_color_mode"):
                                # Handle Chaos Europa random color mode
                                random_colors = template.get("random_text_colors", [])
                                if random_colors:
                                    # Use word index as seed for consistent randomness per word
                                    random.seed(current_word_index + hash(word))
                                    color = random.choice(random_colors)
                                else:
                                    color = highlight_colors[color_index] if highlight_colors else text_color
                            else:
                                # Normal highlighting behavior
                                color = highlight_colors[color_index]
                        
                        # Check if scale effect is enabled for this template
                        should_scale = False
                        if template.get("scale_effect", False):
                            if template.get("random_scale_probability"):
                                # Random scaling for Chaos Europa
                                random.seed(current_word_index + hash(word) + 42)  # Different seed for scaling
                                should_scale = random.random() < template.get("random_scale_probability", 0.4)
                            else:
                                # Normal scaling for all words
                                should_scale = True
                        
                        if should_scale:
                            scale_factor = template.get("scale_factor", 1.2)
                            scaled_font_size = int(template["font_size"] * scale_factor)
                            scaled_font = get_font(template_name, word_by_word_mode=False, custom_template=template)
                            
                            # Create scaled font
                            try:
                                font_path = None
                                for path in template["font_paths"]:
                                    if os.path.exists(path):
                                        font_path = path
                                        break
                                
                                if font_path:
                                    scaled_font = ImageFont.truetype(font_path, scaled_font_size)
                                else:
                                    scaled_font = font  # Fallback to regular font
                            except:
                                scaled_font = font  # Fallback to regular font
                            
                            # Calculate vertical offset to center the scaled word
                            regular_height = get_text_size(draw, word, font)[1]
                            scaled_height = get_text_size(draw, word, scaled_font)[1]
                            y_offset = (scaled_height - regular_height) // 2
                            
                            light_color = create_light_color(color, light_opacity) if has_light else None
                            draw_text_with_stroke(draw, (current_x, y - y_offset), word, scaled_font, color, stroke_color, stroke_width, shadow_color, shadow_offset, glow_color, glow_radius, light_color, light_radius)
                            word_width, _ = get_text_size(draw, word + " ", scaled_font)
                        else:
                            light_color = create_light_color(color, light_opacity) if has_light else None
                            draw_text_with_stroke(draw, (current_x, y), word, font, color, stroke_color, stroke_width, shadow_color, shadow_offset, glow_color, glow_radius, light_color, light_radius)
                            word_width, _ = get_text_size(draw, word + " ", font)
                else:
                    # Handle dynamic text color for non-highlighted words in Dynamic Proxima template
                    if template.get("highlight_cycle_mode") == "dynamic_text" and highlight_word_index is not None:
                        # For Dynamic Proxima: determine text color based on current cycle
                        color_index = current_word_index % len(highlight_colors)  # Use current_word_index for color cycling
                        text_color_cycle = template.get("text_color_cycle", [text_color])
                        if len(text_color_cycle) > color_index:
                            color = text_color_cycle[color_index]
                        else:
                            color = text_color
                    elif template.get("cycle_text_colors"):
                        # Handle Ocean Waves text color cycling for non-highlighted words
                        text_color_cycle = template.get("text_color_cycle", [text_color])
                        if text_color_cycle:
                            color_index = current_word_index % len(text_color_cycle)
                            color = text_color_cycle[color_index]
                            print(f"[DEBUG] Ocean Waves (non-highlighted) - Word: '{word}', Index: {current_word_index}, Color Index: {color_index}, Color: {color}")
                        else:
                            color = text_color
                    elif template.get("random_color_mode"):
                        # Handle Chaos Europa random color mode for non-highlighted words
                        random_colors = template.get("random_text_colors", [])
                        if random_colors:
                            # Use word index as seed for consistent randomness per word
                            random.seed(current_word_index + hash(word))
                            color = random.choice(random_colors)
                        else:
                            color = text_color
                    else:
                        color = text_color
                    
                    light_color = create_light_color(color, light_opacity) if has_light else None
                    draw_text_with_stroke(draw, (current_x, y), word, font, color, stroke_color, stroke_width, shadow_color, shadow_offset, glow_color, glow_radius, light_color, light_radius)
                    word_width, _ = get_text_size(draw, word + " ", font)
                
                current_x += word_width + word_spacing_px

    image.save(output_path)
    logger.debug(f"Caption image saved: {output_path}")

# Import auth modules - DATABASE REMOVED
# from db_init import db
from auth import auth, create_demo_token
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import secrets
import json

# R2 storage utilities removed - using local storage only
# from video_manager import VideoManager  # DATABASE REMOVED



# Translation service
# Translation service removed
# Define common languages for other uses if needed
COMMON_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'ar': 'Arabic',
    'hi': 'Hindi'
}

# Demo authentication endpoint for testing
@app.post("/api/auth/demo")
async def demo_auth(plan: str = Form("pro")):
    """Create a demo authentication token for testing"""
    try:
        token = create_demo_token(plan=plan)
        return JSONResponse(content={
            "token": token,
            "user": {
                "id": "demo_user",
                "email": "demo@example.com",
                "name": "Demo User",
                "plan": plan
            }
        })
    except Exception as e:
        logger.error(f"Error creating demo token: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to create demo token"}
        )

# Password reset endpoints
@app.post("/api/auth/forgot-password")
async def forgot_password(email: str = Form(...)):
    """Send password reset email"""
    try:
        # Generate a secure reset token
        reset_token = secrets.token_urlsafe(32)
        
        # Store the reset token - DATABASE REMOVED
        # expiration = datetime.now() + timedelta(hours=1)
        expiration = datetime.now() + timedelta(hours=1)  # For demo logging only
        
        # For demo purposes, we'll just log the reset token
        # In production, you would:
        # 1. Store the token in database with user email and expiration
        # 2. Send an email with the reset link
        
        print(f"📧 Password reset requested for: {email}")
        print(f"🔑 Reset token: {reset_token}")
        print(f"⏰ Expires at: {expiration} (demo only - not stored)")
        
        # Simulate email sending
        reset_link = f"http://localhost:3000/reset-password?token={reset_token}&email={email}"
        print(f"🔗 Reset link: {reset_link}")
        
        logger.info(f"Password reset requested for email: {email}")
        
        return JSONResponse(content={
            "success": True,
            "message": "Password reset email sent successfully",
            "reset_token": reset_token,  # Remove this in production
            "reset_link": reset_link     # Remove this in production
        })
        
    except Exception as e:
        logger.error(f"Error sending password reset email: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to send password reset email"}
        )

@app.post("/api/auth/reset-password")
async def reset_password(
    token: str = Form(...),
    email: str = Form(...),
    new_password: str = Form(...)
):
    """Reset password using token"""
    try:
        # In production, you would:
        # 1. Verify the token exists in database
        # 2. Check if token has expired
        # 3. Verify the email matches the token
        # 4. Hash the new password
        # 5. Update the user's password in database
        # 6. Delete the used token
        # NOTE: DATABASE REMOVED - password reset disabled
        
        print(f"🔄 Password reset attempt for: {email}")
        print(f"🔑 Token: {token}")
        print(f"🔒 New password length: {len(new_password)} characters")
        
        # For demo purposes, we'll just validate basic requirements
        if len(new_password) < 6:
            return JSONResponse(
                status_code=400,
                content={"error": "Password must be at least 6 characters long"}
            )
        
        logger.info(f"Password reset completed for email: {email}")
        
        return JSONResponse(content={
            "success": True,
            "message": "Password reset successfully"
        })
        
    except Exception as e:
        logger.error(f"Error resetting password: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to reset password"}
        )

@app.post("/api/auth/verify-reset-token")
async def verify_reset_token(
    token: str = Form(...),
    email: str = Form(...)
):
    """Verify if reset token is valid"""
    try:
        # In production, you would:
        # 1. Check if token exists in database
        # 2. Verify token hasn't expired
        # 3. Verify email matches the token
        
        print(f"[DETAIL] Verifying reset token for: {email}")
        print(f"🔑 Token: {token}")
        
        # For demo purposes, we'll accept any token
        logger.info(f"Reset token verified for email: {email}")
        
        return JSONResponse(content={
            "valid": True,
            "email": email
        })
        
    except Exception as e:
        logger.error(f"Error verifying reset token: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to verify reset token"}
        )

# Collaborator system for pro users
@app.post("/api/invite-collaborator")
async def invite_collaborator(
    request: Request,
    email: str = Form(...),
    message: str = Form(""),
    model_size: str = Form("small"),
):
    """Invite a collaborator to join the account (Pro users only)"""
    try:
        # Require pro plan authentication
        user = auth.require_pro_plan(request)
        
        # Get user details
        user_id = user["user_id"]
        user_email = user["email"]
        user_name = user.get("name", user_email.split('@')[0])
        
        # Create invitation in database - DATABASE REMOVED
        # invitation_id = db.create_invitation(
        #     inviter_id=user_id,
        #     inviter_name=user_name,
        #     inviter_email=user_email,
        #     email=email,
        #     message=message
        # )
        
        # Generate mock invitation ID for demo
        invitation_id = f"invite_{secrets.token_urlsafe(16)}"
        
        print(f"📧 Collaborator invitation sent:")
        print(f"   From: {user_name} ({user_email})")
        print(f"   To: {email}")
        print(f"   Message: {message}")
        print(f"   Invitation ID: {invitation_id}")
        
        return JSONResponse(content={
            "success": True,
            "invitation_id": invitation_id,
            "message": "Invitation sent successfully"
        })
        
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )
    except Exception as e:
        logger.error(f"Error sending collaborator invitation: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to send invitation"}
        )

@app.get("/api/collaborators")
async def get_collaborators(request: Request):
    """Get list of active collaborators for the account"""
    try:
        # Require authentication
        user = auth.require_auth(request)
        user_id = user["user_id"]
        
        # Get collaborators from database - DATABASE REMOVED
        # collaborators = db.get_user_collaborators(user_id)
        collaborators = []  # No database - return empty list
        
        return JSONResponse(content={"collaborators": collaborators})
        
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )
    except Exception as e:
        logger.error(f"Error fetching collaborators: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch collaborators"}
        )

@app.get("/api/collaborator-invitations")
async def get_collaborator_invitations(request: Request):
    """Get list of pending collaborator invitations"""
    try:
        # Require authentication
        user = auth.require_auth(request)
        user_id = user["user_id"]
        
        # Get invitations from database - DATABASE REMOVED
        # invitations = db.get_user_invitations(user_id)
        invitations = []  # No database - return empty list
        
        return JSONResponse(content={"invitations": invitations})
        
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )
    except Exception as e:
        logger.error(f"Error fetching invitations: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch invitations"}
        )

@app.get("/api/invitation/{invitation_id}")
async def get_invitation(invitation_id: str):
    """Get invitation details for acceptance page"""
    try:
        # Get invitation from database (no auth required for public viewing) - DATABASE REMOVED
        # invitation = db.get_invitation(invitation_id)
        invitation = None  # No database - return None
        
        if not invitation:
            return JSONResponse(
                status_code=404,
                content={"error": "Invitation not found"}
            )
        
        # Check if invitation has expired
        if datetime.now() > datetime.fromisoformat(invitation["expiresAt"]):
            return JSONResponse(
                status_code=410,
                content={"error": "Invitation has expired"}
            )
        
        # Check if invitation is still pending
        if invitation["status"] != "pending":
            return JSONResponse(
                status_code=410,
                content={"error": "Invitation is no longer valid"}
            )
        
        return JSONResponse(content={"invitation": invitation})
        
    except Exception as e:
        logger.error(f"Error fetching invitation: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch invitation"}
        )

@app.post("/api/accept-invitation/{invitation_id}")
async def accept_invitation(invitation_id: str, request: Request):
    """Accept a collaborator invitation"""
    try:
        # Require authentication
        user = auth.require_auth(request)
        user_id = user["user_id"]
        user_email = user["email"]
        user_name = user.get("name", user_email.split('@')[0])
        
        # Accept invitation in database - DATABASE REMOVED
        # success = db.accept_invitation(
        #     invitation_id=invitation_id,
        #     collaborator_user_id=user_id,
        #     collaborator_email=user_email,
        #     collaborator_name=user_name
        # )
        success = False  # No database - always return False
        
        if not success:
            return JSONResponse(
                status_code=400,
                content={"error": "Unable to accept invitation. It may be expired or already accepted."}
            )
        
        print(f"[OK] Collaborator invitation accepted: {invitation_id}")
        print(f"   User: {user_name} ({user_email})")
        
        return JSONResponse(content={
            "success": True,
            "message": "Successfully joined the team"
        })
        
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )
    except Exception as e:
        logger.error(f"Error accepting invitation: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to accept invitation"}
        )

@app.delete("/api/collaborators/{collaborator_id}")
async def remove_collaborator(collaborator_id: str, request: Request):
    """Remove a collaborator from the account"""
    try:
        # Require authentication
        user = auth.require_auth(request)
        user_id = user["user_id"]
        
        # Remove collaborator from database - DATABASE REMOVED
        # success = db.remove_collaborator(collaborator_id, user_id)
        success = False  # No database - always return False
        
        if not success:
            return JSONResponse(
                status_code=404,
                content={"error": "Collaborator not found or you don't have permission to remove them"}
            )
        
        print(f"🗑️ Collaborator removed: {collaborator_id}")
        
        return JSONResponse(content={
            "success": True,
            "message": "Collaborator removed successfully"
        })
        
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )
    except Exception as e:
        logger.error(f"Error removing collaborator: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to remove collaborator"}
        )

@app.delete("/api/collaborator-invitations/{invitation_id}")
async def cancel_invitation(invitation_id: str, request: Request):
    """Cancel a pending collaborator invitation"""
    try:
        # Require authentication
        user = auth.require_auth(request)
        user_id = user["user_id"]
        
        # Cancel invitation in database - DATABASE REMOVED
        # success = db.cancel_invitation(invitation_id, user_id)
        success = False  # No database - always return False
        
        if not success:
            return JSONResponse(
                status_code=404,
                content={"error": "Invitation not found or you don't have permission to cancel it"}
            )
        
        print(f"❌ Collaborator invitation cancelled: {invitation_id}")
        
        return JSONResponse(content={
            "success": True,
            "message": "Invitation cancelled successfully"
        })
        
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )
    except Exception as e:
        logger.error(f"Error cancelling invitation: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to cancel invitation"}
        )

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/templates")
async def get_templates():
    """Get available caption templates"""
    template_list = []
    for key, template in CAPTION_TEMPLATES.items():
        # Check if template has real highlighting (highlight_colors different from text_color)
        has_real_highlighting = False
        if template.get("highlight_colors"):
            text_color = template.get("text_color", (255, 255, 255, 255))
            highlight_colors = template.get("highlight_colors", [])
            # Check if any highlight color is different from text color
            has_real_highlighting = any(
                highlight_color != text_color for highlight_color in highlight_colors
            )
        
        template_info = {
            "id": key,
            "name": template.get("name", key),
            "description": template.get("description", "No description available"),
            "font_size": template["font_size"],
            "has_highlighting": bool(template.get("highlight_colors") or template.get("highlight_color")),
            "has_real_highlighting": has_real_highlighting,  # New field to distinguish real highlighting
            "uses_text_color": True,  # All templates support text color customization
            "has_stroke": bool(template.get("stroke_color")),
            "has_shadow": bool(template.get("shadow_color"))
        }
        template_list.append(template_info)
    
    return {
        "templates": template_list,
        "current_template": CURRENT_TEMPLATE
    }

def add_watermark_to_video(input_path, output_path, watermark_path, ffmpeg_cmd):
    """
    Add a watermark to the video for free plan users.
    Positions the watermark at the bottom right with 80% brightness.
    
    Args:
        input_path: Path to the input video file
        output_path: Path to the output video file  
        watermark_path: Path to the watermark image
        ffmpeg_cmd: FFmpeg command executable
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"[IMAGES]  Adding watermark to video...")
        print(f"   - Input video: {input_path}")
        print(f"   - Watermark image: {watermark_path}")
        print(f"   - Output path: {output_path}")
        logger.info(f"Adding watermark: {watermark_path} to video: {input_path}")
        
        # Verify watermark image exists and get its properties
        if not os.path.exists(watermark_path):
            print(f"❌ Watermark image not found: {watermark_path}")
            return False
            
        # Get watermark image size for verification
        try:
            from PIL import Image
            with Image.open(watermark_path) as img:
                print(f"   - Watermark size: {img.size[0]}x{img.size[1]}")
        except Exception as e:
            print(f"   - Could not read watermark size: {e}")
        
        # Create a temporary output path for watermark processing
        temp_watermark_output = output_path.replace('.mp4', '_temp_watermark.mp4')
        print(f"   - Temp output: {temp_watermark_output}")
        
        # FFmpeg command to add watermark with positioning and opacity adjustment
        # For vertical video (1080x1920), position watermark at bottom right with padding
        # Scale watermark to appropriate size and reduce opacity to 60%
        watermark_cmd = [
            ffmpeg_cmd,
            "-i", input_path,  # Input video
            "-i", watermark_path,  # Watermark image
            "-filter_complex", 
            # Scale watermark to 162x162 pixels, position at bottom right with 20px from right, 50px from bottom
            # No trailing semicolon after scale for FFmpeg >4 compatibility
            "[1:v]scale=162:162[watermark];"
            "[0:v][watermark]overlay=main_w-overlay_w-20:main_h-overlay_h-50:format=auto,format=yuv420p",
            "-c:a", "copy",  # Copy audio without re-encoding
            "-c:v", "libx264",  # Video codec
            "-preset", "fast",  # Encoding preset for faster processing
            "-y",  # Overwrite output file
            temp_watermark_output
        ]
        
        print(f"[FFMPEG] Watermark FFmpeg command: {' '.join(watermark_cmd)}")
        logger.info(f"Watermark FFmpeg command: {' '.join(watermark_cmd)}")
        
        # Execute the watermark command
        print(f"[FFMPEG] Executing watermark command...")
        watermark_result = subprocess.run(watermark_cmd, check=True, capture_output=True, text=True)
        print(f"[OK] Watermark command completed successfully")
        
        if os.path.exists(temp_watermark_output):
            # Replace the original output with the watermarked version
            shutil.move(temp_watermark_output, output_path)
            print(f"[OK] Watermark added successfully")
            logger.info(f"Watermark added successfully to: {output_path}")
            return True
        else:
            print(f"[WARNING] Watermark processing failed - output file not created")
            logger.error(f"Watermark processing failed - output file not created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Watermark processing failed: {e}")
        logger.error(f"Watermark processing failed: {e}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        
        # Clean up temporary file if it exists
        if os.path.exists(temp_watermark_output):
            try:
                os.remove(temp_watermark_output)
            except Exception as cleanup_error:
                logger.warning(f"Could not remove temporary watermark file: {cleanup_error}")
        
        return False
    except Exception as e:
        print(f"[WARNING] Unexpected error in watermark processing: {e}")
        logger.error(f"Unexpected error in watermark processing: {e}")
        return False

@app.post("/api/check-video-duration")
async def check_video_duration_endpoint(
    file: UploadFile = File(...),
    user_plan: str = Form("free")
):
    """
    Check video duration against plan limits without processing the video
    Used for frontend validation before actual upload
    """
    try:
        # Create temporary file to check duration
        temp_id = str(uuid.uuid4())
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{temp_id}_{file.filename}")
        
        print(f"[DETAIL] Checking duration for {file.filename} (plan: {user_plan})")
        logger.info(f"Checking video duration for file: {file.filename}, plan: {user_plan}")
        
        # Save file temporarily
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        try:
            # Get video duration
            video_duration = get_video_duration(temp_path)
            
            # Validate against plan limits
            is_valid, error_message, plan_limit = validate_video_duration(video_duration, user_plan)
            
            response_data = {
                "valid": is_valid,
                "video_duration": video_duration,
                "plan_limit": plan_limit,
                "user_plan": user_plan
            }
            
            if not is_valid:
                response_data["error"] = error_message
                print(f"❌ Duration check failed: {error_message}")
                logger.warning(f"Duration check failed: {error_message}")
            else:
                duration_minutes = video_duration / 60 if video_duration else 0
                print(f"[OK] Duration check passed: {duration_minutes:.2f} minutes")
                logger.info(f"Duration check passed: {video_duration:.1f} seconds")
            
            return JSONResponse(content=response_data)
            
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Could not clean up temp file {temp_path}: {e}")
                
    except Exception as e:
        logger.error(f"Error checking video duration: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to check video duration: {str(e)}"}
        )


@app.post("/upload", include_in_schema=False)
@app.post("/upload/")
async def upload_and_process_video(
    request: Request,
    file: UploadFile = File(...),
    template: str = Form("MrBeast"),
    caption_position: float = Form(0.7),
    highlight_color: str = Form(None),
    text_color: str = Form(None),
    font_size: int = Form(65),
    caption_layout: str = Form("wrapped"),
    track_speakers: bool = Form(False),
    enable_profanity_filter: bool = Form(False),
    profanity_filter_mode: str = Form("both"),
    custom_profanity_words: str = Form(""),
    use_custom_transcription: bool = Form(False),
    custom_transcription: str = Form(None),
    animation_type: str = Form("none"),
    animation_speed: float = Form(1.0),
    line_height: float = Form(1.2),
    word_spacing: int = Form(0),
    user_plan: str = Form("free"),
    trim_video: bool = Form(False),
    trim_start: float = Form(0.0),
    trim_end: float = Form(0.0),
    user_id: str = Form(None),
    background_music: str = Form(None),
    enable_emojis: bool = Form(False),
    emoji_density: int = Form(2),
    exact_word_timestamps: bool = Form(False),
    language: str = Form("en"),
    verbose: bool = Form(False),
    enable_translation: bool = Form(False),
    target_language: str = Form("es")
):
    try:
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Initialize filter file path (will be used later)
        filter_file_path = None
        
        print(f"\n{'='*60}")
        print(f"[FFMPEG] NEW VIDEO PROCESSING REQUEST - {timestamp}")
        print(f"{'='*60}")
        logger.info(f"New video processing request started")
        
        print(f"📁 File: {file.filename} ({file.content_type})")
        print(f"🎨 Template: {template}")
        print(f"👤 Received user_id parameter: {user_id}")
        print(f"👤 User_id type: {type(user_id)}")
        print(f"👤 User_id is None: {user_id is None}")
        print(f"👤 User_id is empty string: {user_id == ''}")
        print(f"[MEASURE] Caption Position: {caption_position:.2f}")
        print(f"[FONT] Font Size: {font_size}px")
        print(f"[PHRASES] Caption Layout: {caption_layout}")
        print(f"👥 Speaker Tracking: {'Enabled' if track_speakers else 'Disabled'}")
        print(f"[CENSOR] Profanity Filter: {'Enabled' if enable_profanity_filter else 'Disabled'}")
        if enable_profanity_filter:
            print(f"🎯 Profanity Filter Mode: {profanity_filter_mode}")
        print(f"[FFMPEG] Animation Type: {animation_type}")
        print(f"⚡ Animation Speed: {animation_speed}x")
        print(f"[SPACING] Line Height: {line_height}x")
        print(f"[MEASURE] Word Spacing: {word_spacing}px")
        if animation_type != 'none':
            print(f"🎯 Animation will be applied to caption entrance effects")
        print(f"🔄 Using Custom Transcription: {'Yes' if use_custom_transcription else 'No'}")
        print(f"👤 User Plan: {user_plan}")
        if custom_profanity_words:
            custom_words = [word.strip() for word in custom_profanity_words.split(',') if word.strip()]
            print(f"[WORDS] Custom Profanity Words: {len(custom_words)} words added")
        if highlight_color:
            print(f"🌈 Highlight Color: {highlight_color}")
        if text_color:
            print(f"🎨 Text Color: {text_color}")
        if background_music:
            print(f"🎵 Background Music: {background_music}")
        print(f"🌐 Translation: {'Enabled' if enable_translation else 'Disabled'}")
        if enable_translation:
            print(f"🌍 Target Language: {target_language}")
        logger.info(f"Processing file: {file.filename}, template: {template}, position: {caption_position:.2f}, font size: {font_size}px, layout: {caption_layout}, speaker tracking: {track_speakers}, profanity filter: {enable_profanity_filter} ({profanity_filter_mode}), animation: {animation_type} ({animation_speed}x), line_height: {line_height}x, word_spacing: {word_spacing}px, highlight: {highlight_color}, text: {text_color}, custom transcription: {use_custom_transcription}, background music: {background_music is not None}, translation: {enable_translation} ({target_language})")
        
        video_id = str(uuid.uuid4())
        sequence_number = get_next_sequence_number()
        output_filename = f"Quickcap output {sequence_number}"
        
        input_path = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")
        output_path = os.path.join(OUTPUT_DIR, f"{output_filename}.mp4")
        
        print(f"🆔 Video ID: {video_id}")
        print(f"🔢 Sequence Number: {sequence_number}")
        print(f"📥 Input path: {input_path}")
        print(f"📤 Output path: {output_path}")
        logger.info(f"Video ID: {video_id}, Sequence: {sequence_number}, Input: {input_path}, Output: {output_path}")

        # Save uploaded file
        print(f"💾 Saving uploaded file...")
        logger.info("Saving uploaded file to disk")
        file_save_start = time.time()
        with open(input_path, "wb") as f:
            while chunk := await file.read(5 * 1024 * 1024):  # Read 5MB chunks
                f.write(chunk)
        file_save_time = time.time() - file_save_start
        
        # Get file size
        file_size = os.path.getsize(input_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"[OK] File saved successfully ({file_size_mb:.2f} MB) in {file_save_time:.2f} seconds")
        logger.info(f"File saved: {file_size_mb:.2f} MB in {file_save_time:.2f} seconds")
        
        # Check video duration against plan limits
        print(f"\n⏱️  Checking video duration for {user_plan} plan...")
        logger.info(f"Checking video duration for plan: {user_plan}")
        duration_check_start = time.time()
        
        video_duration = get_video_duration(input_path)
        duration_check_time = time.time() - duration_check_start
        
        if video_duration is not None:
            duration_minutes = video_duration / 60
            print(f"📹 Video duration: {duration_minutes:.2f} minutes ({video_duration:.1f} seconds)")
            logger.info(f"Video duration: {video_duration:.1f} seconds ({duration_minutes:.2f} minutes)")
        else:
            print(f"[WARNING]  Could not determine video duration")
            logger.warning("Could not determine video duration")
        
        # Validate duration against plan limits
        is_valid, error_message, plan_limit = validate_video_duration(video_duration, user_plan)
        if not is_valid:
            print(f"❌ Duration validation failed: {error_message}")
            
        # Trim video if requested
        if trim_video and trim_end > trim_start and trim_end <= video_duration:
            print(f"\n✂️  Trimming video from {trim_start:.2f}s to {trim_end:.2f}s...")
            logger.info(f"Trimming video from {trim_start:.2f}s to {trim_end:.2f}s")
            
            # Create a temporary file for the trimmed video
            trimmed_path = os.path.join(UPLOAD_DIR, f"trimmed_{video_id}.mp4")
            
            try:
                # Use FFmpeg to trim the video
                trim_command = [
                    "ffmpeg",
                    "-i", input_path,
                    "-ss", str(trim_start),
                    "-to", str(trim_end),
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-strict", "experimental",
                    "-b:a", "192k",
                    "-y",  # Overwrite output file if it exists
                    trimmed_path
                ]
                
                trim_start_time = time.time()
                subprocess.run(trim_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                trim_time = time.time() - trim_start_time
                
                # Replace the input path with the trimmed video
                if os.path.exists(trimmed_path):
                    print(f"[OK] Video trimmed successfully in {trim_time:.2f} seconds")
                    logger.info(f"Video trimmed successfully in {trim_time:.2f} seconds")
                    
                    # Update the input path to use the trimmed video
                    input_path = trimmed_path
                    
                    # Update video duration
                    video_duration = get_video_duration(input_path)
                    if video_duration is not None:
                        duration_minutes = video_duration / 60
                        print(f"📹 New video duration: {duration_minutes:.2f} minutes ({video_duration:.1f} seconds)")
                        logger.info(f"New video duration: {video_duration:.1f} seconds ({duration_minutes:.2f} minutes)")
                else:
                    print(f"[WARNING] Trimmed video not found, using original video")
                    logger.warning("Trimmed video not found, using original video")
            except Exception as e:
                print(f"[ERROR] Failed to trim video: {str(e)}")
                logger.error(f"Failed to trim video: {str(e)}")
                # Continue with the original video if trimming fails
                
        if not is_valid:
            logger.error(f"Duration validation failed for {user_plan} plan: {error_message}")
            
            # Clean up the uploaded file
            try:
                os.remove(input_path)
                print(f"🧹 Cleaned up uploaded file: {input_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up file {input_path}: {cleanup_error}")
            
            return JSONResponse(
                status_code=413,  # Payload Too Large
                content={
                    "error": "Video duration exceeds plan limit",
                    "message": error_message,
                    "video_duration": video_duration,
                    "plan_limit": plan_limit,
                    "user_plan": user_plan
                }
            )
        
        print(f"[OK] Duration validation passed ({duration_check_time:.2f}s)")
        logger.info(f"Duration validation passed in {duration_check_time:.2f} seconds")

        # Transcription processing
        if use_custom_transcription and custom_transcription:
            # Use the provided custom transcription
            print(f"\n📝 Using provided custom transcription...")
            print(f"[FONT] Font Size: {font_size}px (explicitly set for custom transcription)")
            logger.info(f"Using custom transcription provided in request with font size: {font_size}px")
            transcription_start = time.time()
            
            try:
                import json
                # Parse the custom transcription JSON
                transcription_data = json.loads(custom_transcription)
                
                # Extract segments and words from the custom transcription
                segments = transcription_data.get("segments", [])
                
                # Process words from segments
                words = []
                
                # First, check if this is a reprocessed transcription with edited text
                print(f"[WORDS] Processing custom transcription with {len(segments)} segments")
                
                # Regenerate words for each segment based on the text
                # This ensures edited words are properly processed
                for segment in segments:
                    # Get the segment text (which may have been edited)
                    segment_text = segment.get("text", "").strip()
                    print(f"   Segment text: '{segment_text}'")
                    
                    # Check if segment already has word-level timing information
                    if "words" in segment and segment["words"]:
                        # Use existing word timing information (preserve original timing)
                        # BUT update the word text to match any edits made to the segment
                        print(f"   Using existing word timing information for segment")
                        segment_words = segment["words"]
                        
                        # Split the edited segment text into words
                        edited_words = segment_text.split()
                        
                        # Check if the number of words matches the timing data
                        if len(edited_words) == len(segment_words):
                            # Perfect match - update word text while preserving timing
                            print(f"   [OK] Word count matches - updating word text with edits")
                            for i, word_obj in enumerate(segment_words):
                                if "word" in word_obj and "start" in word_obj and "end" in word_obj:
                                    # Update the word text with the edited version
                                    updated_word = {
                                        "word": edited_words[i],
                                        "start": word_obj["start"],
                                        "end": word_obj["end"]
                                    }
                                    words.append(updated_word)
                                else:
                                    print(f"   [WARNING] Invalid word object: {word_obj}")
                            
                            print(f"   [OK] Updated {len(segment_words)} words with edited text")
                        else:
                            # Word count mismatch - need to redistribute timing
                            print(f"   [WARNING] Word count mismatch: {len(edited_words)} edited vs {len(segment_words)} original")
                            print(f"   [FALLBACK] Redistributing timing for edited words")
                            
                            # Use the segment timing but redistribute among edited words
                            segment_duration = segment.get("end", 0) - segment.get("start", 0)
                            word_duration = segment_duration / len(edited_words) if edited_words else 0
                            
                            # Create word objects with redistributed timing
                            for i, word_text in enumerate(edited_words):
                                word_start = segment.get("start", 0) + (i * word_duration)
                                word_end = segment.get("start", 0) + ((i + 1) * word_duration)
                                word_obj = {
                                    "word": word_text,
                                    "start": word_start,
                                    "end": word_end
                                }
                                words.append(word_obj)
                            
                            print(f"   [OK] Redistributed timing for {len(edited_words)} edited words")
                    else:
                        # Split into words and create timing information
                        segment_words_text = segment_text.split()
                        segment_duration = segment.get("end", 0) - segment.get("start", 0)
                        word_duration = segment_duration / len(segment_words_text) if segment_words_text else 0
                        
                        print(f"   Creating word timing: {len(segment_words_text)} words, {segment_duration:.2f}s duration")
                        print(f"   [WARNING] Using estimated timing - word highlighting may not be accurate")
                        
                        # Create word objects with timing information
                        segment_words = []
                        for i, word_text in enumerate(segment_words_text):
                            word_start = segment.get("start", 0) + (i * word_duration)
                            word_end = segment.get("start", 0) + ((i + 1) * word_duration)
                            word_obj = {
                                "word": word_text,
                                "start": word_start,
                                "end": word_end
                            }
                            words.append(word_obj)
                            segment_words.append(word_obj)
                        
                        # Update the segment with the new words
                        segment["words"] = segment_words
                        print(f"   [OK] Created estimated timing for {len(segment_words)} words")
                
                # Create a result object similar to what Whisper would return
                result = {
                    "segments": segments,
                    "text": transcription_data.get("text", ""),
                    "language": transcription_data.get("language", "en")
                }
                
                transcription_time = time.time() - transcription_start
                print(f"[OK] Custom transcription processed in {transcription_time:.2f} seconds")
                logger.info(f"Custom transcription processed in {transcription_time:.2f} seconds")
                
            except Exception as e:
                print(f"[WARNING] Error processing custom transcription: {e}")
                logger.error(f"Error processing custom transcription: {e}")
                # Fall back to Whisper transcription
                print(f"\n🤖 Falling back to AI transcription with Whisper...")
                logger.info("Falling back to Whisper AI transcription")
                transcription_start = time.time()
                
                result = model.transcribe(input_path, word_timestamps=True)
                transcription_time = time.time() - transcription_start
                
                segments = result["segments"]
                # Extract words while preserving emoji information
                words = []
                for segment in segments:
                    if "words" in segment and segment["words"]:
                        for word in segment["words"]:
                            words.append(word)
                    else:
                        # Fallback for segments without word-level timing
                        segment_words_text = segment.get("text", "").split()
                        segment_duration = segment.get("end", 0) - segment.get("start", 0)
                        word_duration = segment_duration / len(segment_words_text) if segment_words_text else 0
                        
                        for i, word_text in enumerate(segment_words_text):
                            word_start = segment.get("start", 0) + (i * word_duration)
                            word_end = segment.get("start", 0) + ((i + 1) * word_duration)
                            words.append({
                                "word": word_text,
                                "start": word_start,
                                "end": word_end
                            })
        else:
            # Use the preloaded model for immediate transcription
            logger.info("Starting transcription with WhisperPreloader...")
            transcription_start = time.time()
            
            # Ensure model is loaded before proceeding
            if not whisper_preloader.model_loaded:
                logger.info("Model not loaded yet, loading now...")
                whisper_preloader.load_model()
            
            try:
                # Use the preloaded model for immediate transcription
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: whisper_preloader.transcribe(input_path, language=language, verbose=verbose)
                    ),
                    timeout=300.0  # 5 minute timeout
                )
                logger.info("Transcription completed in main process")
                
            except asyncio.TimeoutError:
                logger.warning("Main process transcription timed out, falling back to worker process...")
                
                # Fall back to worker process if main process times out
                # Check if worker process is alive before sending task
                if not transcription_process or not transcription_process.is_alive():
                    logger.warning("Transcription worker process is not alive. Restarting...")
                    await restart_transcription_worker()
                
                # The task now is a tuple of arguments for the worker
                task = (input_path, language, verbose)
                task_queue.put(task)
                
                try:
                    # Use asyncio to implement timeout for queue.get()
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, result_queue.get),
                        timeout=300.0  # 5 minute timeout
                    )
                except asyncio.TimeoutError:
                    logger.error("Transcription worker timed out after 5 minutes")
                    return JSONResponse(
                        status_code=500,
                        content={"error": "Transcription service unavailable", "details": "Transcription timed out"}
                    )
                except Exception as e:
                    logger.error(f"Transcription worker failed: {e}")
                    return JSONResponse(
                        status_code=500,
                        content={"error": "Transcription failed", "details": str(e)}
                    )
                    
            except Exception as e:
                logger.error(f"Main process transcription failed: {e}")
                
                # Fall back to worker process if main process fails
                logger.info("Falling back to worker process for transcription...")
                
                # Check if worker process is alive before sending task
                if not transcription_process or not transcription_process.is_alive():
                    logger.warning("Transcription worker process is not alive. Restarting...")
                    await restart_transcription_worker()
                
                # The task now is a tuple of arguments for the worker
                task = (input_path, language, verbose)
                task_queue.put(task)
                
                try:
                    # Use asyncio to implement timeout for queue.get()
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, result_queue.get),
                        timeout=300.0  # 5 minute timeout
                    )
                except Exception as worker_e:
                    logger.error(f"Both main and worker transcription failed: {worker_e}")
                    return JSONResponse(
                        status_code=500,
                        content={"error": "Transcription failed", "details": f"Main: {str(e)}, Worker: {str(worker_e)}"}
                    )

            transcription_time = time.time() - transcription_start

            # Check if the worker process returned an error
            if isinstance(result, dict) and 'error' in result:
                logger.error(f"Transcription failed in worker process: {result['error']}")
                # Handle the error appropriately, maybe return an error response
                return JSONResponse(
                    status_code=500,
                    content={"error": "Transcription failed", "details": result['error']}
                )
            
            segments = result["segments"]
            # Extract words while preserving emoji information
            words = []
            for segment in segments:
                if "words" in segment and segment["words"]:
                    for word in segment["words"]:
                        words.append(word)
                else:
                    # Fallback for segments without word-level timing
                    segment_words_text = segment.get("text", "").split()
                    segment_duration = segment.get("end", 0) - segment.get("start", 0)
                    word_duration = segment_duration / len(segment_words_text) if segment_words_text else 0
                    
                    for i, word_text in enumerate(segment_words_text):
                        word_start = segment.get("start", 0) + (i * word_duration)
                        word_end = segment.get("start", 0) + ((i + 1) * word_duration)
                        words.append({
                            "word": word_text,
                            "start": word_start,
                            "end": word_end
                        })
        
        # Debug: Check if emojis are preserved in words list
        emoji_words_in_list = [word for word in words if word.get('is_emoji', False)]
        if emoji_words_in_list:
            print(f"[DEBUG] Found {len(emoji_words_in_list)} emoji words in final words list: {[w['word'] for w in emoji_words_in_list]}")
        else:
            print(f"[DEBUG] No emoji words found in final words list")
        
        # Process translation if enabled
        if enable_translation and target_language != language:
            print(f"\n🌐 Starting translation from {language} to {target_language}...")
            logger.info(f"Starting translation from {language} to {target_language}")
            translation_start = time.time()
            
            try:
                # Translate the segments
                translated_segments = translation_service.translate_segments(
                    segments, 
                    source_lang=language, 
                    target_lang=target_language
                )
                
                # Update segments with translated content
                segments = translated_segments
                
                # Update words list with translated words
                words = []
                for segment in segments:
                    if "words" in segment and segment["words"]:
                        for word in segment["words"]:
                            words.append(word)
                    else:
                        # Fallback for segments without word-level timing
                        segment_words_text = segment.get("text", "").split()
                        segment_duration = segment.get("end", 0) - segment.get("start", 0)
                        word_duration = segment_duration / len(segment_words_text) if segment_words_text else 0
                        
                        for i, word_text in enumerate(segment_words_text):
                            word_start = segment.get("start", 0) + (i * word_duration)
                            word_end = segment.get("start", 0) + ((i + 1) * word_duration)
                            words.append({
                                "word": word_text,
                                "start": word_start,
                                "end": word_end
                            })
                
                translation_time = time.time() - translation_start
                print(f"✅ Translation completed in {translation_time:.2f} seconds")
                logger.info(f"Translation completed in {translation_time:.2f} seconds")
                
                # Debug: Show first few translated segments
                if segments:
                    print(f"[TRANSLATION] First segment: '{segments[0].get('text', '')[:100]}...'")
                    if 'original_text' in segments[0]:
                        print(f"[TRANSLATION] Original: '{segments[0].get('original_text', '')[:100]}...'")
                
            except Exception as e:
                print(f"❌ Translation failed: {str(e)}")
                logger.error(f"Translation failed: {str(e)}")
                # Continue with original segments if translation fails
        elif enable_translation and target_language == language:
            print(f"🌐 Translation skipped - source and target languages are the same ({language})")
            logger.info(f"Translation skipped - same language: {language}")
        
        # Process background music if provided
        music_file_path = None
        music_volume = 0.3  # Default volume
        if background_music:
            try:
                import json
                print(f"🎵 Raw background_music parameter: {background_music}")
                music_data = json.loads(background_music)
                print(f"🎵 Parsed music data: {music_data}")
                
                if music_data.get("enabled", False):
                    music_filename = music_data.get("filename")
                    music_volume = music_data.get("volume", 0.3)
                    music_name = music_data.get("name", "Unknown")
                    
                    print(f"🎵 Music enabled - filename: {music_filename}, volume: {music_volume}, name: {music_name}")
                    
                    if music_filename:
                        # Google Drive File IDs mapping
                        GOOGLE_DRIVE_FILE_MAPPING = {
                            'The Beginning.mp3': '1QlE84ZyCoGkhu1AcKNsdmLoDt5AeYyxm',
                            'The Beginning (Acoustic).mp3': '1Mi5gsfFSnXFoEOMrGdlNUJWvv2u_mB5T',
                            'Flawed Mangoes - Swimming.mp3': '1-HSGvQzSDrjZjgE1nkorg9MrYi9CfhH9',
                            'Flawed Mangoes - Swimming ⧸ Dramamine (Live Version) On The Radar Live Performance.mp3': '1od-3tish81am4cCGS3s2hMpCyt3PZ0BL',
                            'Flawed Mangoes - Tunnel Vision (Visualizer).mp3': '1IZzRRzxPgX-MlvlpqzUwX_CpPFVBQORF',
                            'Flawed Mangoes - The Edge of Memory (Visualizer).mp3': '12-dbAxiDi9I_FMw8b9VtLdg4qP_b2x0L',
                            'Flawed Mangoes - Scattered (Official Music Video).mp3': '1tK6jJOXhjrnIcsJ4YPcrDKLDhzHHea2x',
                            'Flawed Mangoes - Scattered [Instrumental] (Visualizer).mp3': '1E952uSbLwt0BgO72D3QCFYrIncTd67BG',
                            'Flawed Mangoes - Sleepwalking (Visualizer).mp3': '1T_fXjo7so8htI-Cceg72-b1Wf4bkJVF_',
                            'Flawed Mangoes - Take Your Aim (w⧸ Scarlet House) [Visualizer].mp3': '11ZyRdvDySFRDelXRaBsFrWyiDEtnLs64',
                            'Flawed Mangoes with HUGO - 2am (Official Music Video).mp3': '1md3QQwdLE4XGzKkXEdcldj77SYR-c_JU',
                            'Killswitch Lullaby.mp3': '1yg-Mx6i2r56ShsGZiNsjok9W6deZrnSE',
                            'will hyde x Øneheart x Flawed Mangoes - lost a friend (Flawed Mangoes Visualizer).mp3': '13nh3lb50xSj-U9n4EMk9i6quhQXtwxD9',
                            'Riff 2.mp3': '1SKfpR2u05JUYavwavN29AVLfuMdmOQ81',
                            'Birdsong.mp3': '1FSyypyNhgDO6xedmC1-v0MEGO7jGxTRT',
                            'Flawed Mangoes - Cold (Visualizer).mp3': '187T-2HcQvTMY9L0iP8JMeP7DF8kVI6Zb',
                            'Flawed Mangoes - Cold (Acoustic) [Visualizer].mp3': '1ipuBk2wBFNuxgJzg4iOv6z507Y1Fyug3',
                            'Flawed Mangoes - Dramamine (Visualizer).mp3': '1jVsybyuocCW6IsqueCZWsCZCidVdTu0p',
                            'Flawed Mangoes - Entrance (Visualizer).mp3': '1Ke121XYnFgiodUprQtpza4DJMFT3jO97',
                            'Flawed Mangoes - Event Horizon (Visualizer).mp3': '14trx8pqZGZ2K9uP6IHZzeBxK3dxVcfna',
                            'Flawed Mangoes - Fragility (Visualizer).mp3': '1_4wn2c48Nfr-1xgZBjxMDIHAOXYp--5M',
                            'Flawed Mangoes - Immaterial (Visualizer).mp3': '1yeBrISK-UF5ak-Oik3BG4S3SJbXXMaMX',
                            'Flawed Mangoes - Leave A Message (w⧸ aldn) [Visualizer].mp3': '1zuhKoE4KVUHbKVQC8Yac_s88Ea4bGo4a',
                            'Flawed Mangoes - Melody From a Dream (Visualizer).mp3': '1QrJWApjTV-2gUw4tqSFM5MwpTACArJmR',
                            'Flawed Mangoes - Midnight (Visualizer).mp3': '1Mlnc8XsmubhD7_g1oGpJ1vOJosPR_YFn',
                            'Flawed Mangoes - Mindgame (Visualizer).mp3': '1yaMvRSD6-lQiDQLU5dT8BubR7VDZmyal',
                            'Flawed Mangoes - Missing Pieces (Visualizer).mp3': '1sE3UiqzJYURFBuBAlBLZbxZNVtadQ0Cm',
                            'Flawed Mangoes - Missing Pieces (Slowed) [Visualizer].mp3': '1SvjQt_XU-_s7q1vkYUh3ZYGKq8pjGouV',
                            'Flawed Mangoes - Obsession (Visualizer).mp3': '1CFEgNWkDHzik8rVjRzYpFXAqiY2VWESg',
                            'Flawed Mangoes - Pattern Recognition (Visualizer).mp3': '1ChtlEW8TKhD0GUsOOm9HRU9H_1fMS7Cs',
                            'Flawed Mangoes - Run On Sentence (Visualizer).mp3': '1-WKos_Zxp0bhQBl72PEClqG3L1Ik9fLk',
                            'Flawed Mangoes - Run On Sentence (Acoustic) (Visualizer).mp3': '1ATfYhRlsKC_PL1hVBPvFhzAKCpsyETN9',
                            'Flawed Mangoes - Unraveling (Visualizer).mp3': '1-M1lDti3w0VnS3gjcTdrR77zoqY15sey',
                            'Flawed Mangoes - Vague (Visualizer).mp3': '1HcCWOyKkPHSk1P9w4a16ELg1A3KhUAtN',
                            'Flawed Mangoes - Vague (Acoustic) [Visualizer].mp3': '16HlOSTXOfXkvjoqYtm1h571XyTQuSWZi',
                            'Flawed Mangoes - Waking Up Alone (Visualizer).mp3': '1EFO8KosiUsBPky222Meuf6RovtjFIo8M',
                        }
                        
                        # Check if this is a Google Drive file
                        if music_filename in GOOGLE_DRIVE_FILE_MAPPING:
                            file_id = GOOGLE_DRIVE_FILE_MAPPING[music_filename]
                            music_file_path = f"https://drive.google.com/uc?export=download&id={file_id}"
                            print(f"🎵 ✅ Using Google Drive file: {music_name}")
                            print(f"🎵 ✅ Google Drive URL: {music_file_path}")
                            print(f"🎵 ✅ Volume: {int(music_volume * 100)}%")
                            logger.info(f"Background music enabled from Google Drive: {music_name}, volume: {int(music_volume * 100)}%")
                        else:
                            # Fallback to local file system (for backward compatibility)
                            if IS_PRODUCTION:
                                music_file_path = f"/tmp/Music/{music_filename}"
                            else:
                                music_file_path = f"../Clip/App/public/Music/{music_filename}"
                            
                            print(f"🎵 Looking for local music file at: {music_file_path}")
                            
                            # Verify the music file exists locally
                            if os.path.exists(music_file_path):
                                print(f"🎵 ✅ Background music enabled: {music_name}")
                                print(f"🎵 ✅ Local music file found: {music_file_path}")
                                print(f"🎵 ✅ Volume: {int(music_volume * 100)}%")
                                logger.info(f"Background music enabled: {music_name}, volume: {int(music_volume * 100)}%")
                            else:
                                print(f"🎵 ❌ Music file not found locally: {music_file_path}")
                                logger.warning(f"Music file not found: {music_file_path}")
                                music_file_path = None
                    else:
                        print(f"🎵 ❌ No music filename provided")
                        logger.warning("No music filename provided")
                else:
                    print(f"🎵 Music not enabled in data")
            except Exception as e:
                print(f"🎵 ❌ Error processing background music: {e}")
                logger.warning(f"Error processing background music: {e}")
        else:
            print(f"🎵 No background_music parameter provided")
        
        # Initialize profanity processor
        custom_words = []
        if custom_profanity_words:
            custom_words = [word.strip() for word in custom_profanity_words.split(',') if word.strip()]
        
        profanity_processor = ProfanityProcessor(enable_filter=enable_profanity_filter, custom_words=custom_words, filter_mode=profanity_filter_mode)
        
        # Process words for profanity if filtering is enabled
        profanity_timestamps = []
        if enable_profanity_filter:
            print(f"\n🚫 Processing profanity filter...")
            logger.info("Starting profanity filtering")
            words, profanity_timestamps = profanity_processor.process_words_for_profanity(words)
            
            if profanity_timestamps:
                print(f"[CENSOR] Found {len(profanity_timestamps)} profane words to censor")
                logger.info(f"Found {len(profanity_timestamps)} profane words for censoring")
            else:
                print(f"[OK] No profanity detected in transcription")
                logger.info("No profanity detected in transcription")
        
        # Initialize emoji processor
        emoji_processor = None
        if enable_emojis:
            print(f"\n✨ Processing emoji enhancement...")
            logger.info("Starting emoji processing")
            from emoji_processor import EmojiProcessor
            
            # Use the exact_word_timestamps parameter from form data
            print(f"[EMOJI] exact_word_timestamps parameter: {exact_word_timestamps}")
            
            # Create emoji processor with appropriate settings
            emoji_processor = EmojiProcessor(
                max_emojis_per_segment=emoji_density,
                exact_word_timestamps=exact_word_timestamps
            )
            
            if exact_word_timestamps:
                print(f"[EMOJI] Using exact word timestamps for emoji placement")
                logger.info("Using exact word timestamps for emoji placement")
                
                # Process the entire segments list with word-level timing
                segments = emoji_processor.process_transcription_with_word_timestamps(segments)
                
                # Count emojis added
                emoji_count = 0
                for segment in segments:
                    if 'words' in segment:
                        emoji_count += sum(1 for word in segment['words'] if word.get('is_emoji', False))
                
                print(f"[OK] Added {emoji_count} emojis with exact word timestamps")
                logger.info(f"Added {emoji_count} emojis with exact word timestamps")
            else:
                # Process segments with emojis (text-only approach)
                for segment in segments:
                    if 'text' in segment:
                        original_text = segment['text']
                        segment['text'] = emoji_processor.add_emojis_to_text(original_text)
                        if segment['text'] != original_text:
                            print(f"[EMOJI] Enhanced: {original_text} → {segment['text']}")
            
            print(f"[OK] Emoji enhancement completed with max {emoji_density} emojis per segment")
            logger.info(f"Emoji enhancement completed with density {emoji_density}")
            
            # Debug: Check if emojis are in the segments
            total_emojis_in_segments = 0
            for segment in segments:
                if 'words' in segment:
                    emoji_words = [word for word in segment['words'] if word.get('is_emoji', False)]
                    total_emojis_in_segments += len(emoji_words)
                    if emoji_words:
                        print(f"[DEBUG] Segment has {len(emoji_words)} emoji words: {[w['word'] for w in emoji_words]}")
            
            print(f"[DEBUG] Total emojis in segments: {total_emojis_in_segments}")
        
        phrases = chunk_words(words, caption_layout, template)
        
        total_words = len(words)
        total_phrases = len(phrases)
        detected_language = result.get("language", "unknown")
        
        # Determine words per phrase based on layout and template
        if caption_layout == "single_line":
            words_per_phrase = 3
        elif template == "Smilecore":
            words_per_phrase = 5  # Limit to 5 words for Smilecore template
        elif template in ["Hopecore", "ChromaFusion"]:
            words_per_phrase = 5  # Limit to 5 words for Hopecore and ChromaFusion templates
        else:
            words_per_phrase = WORDS_PER_PHRASE
        
        print(f"[OK] Transcription completed in {transcription_time:.2f} seconds")
        print(f"[SPEECH]  Language detected: {detected_language}")
        print(f"[WORDS] Total words: {total_words}")
        print(f"[PHRASES] Total phrases: {total_phrases} ({words_per_phrase} words per phrase)")
        
        # Speaker tracking if enabled
        speaker_segments = {}
        speaker_tracking_time = 0
        track_speakers_success = False
        
        if track_speakers:
            print(f"\n👥 Starting speaker tracking...")
            logger.info("Starting speaker tracking with pyannote.audio")
            speaker_tracking_start = time.time()
            
            try:
                # Initialize speaker tracker
                speaker_tracker = SpeakerTracker(ffmpeg_cmd=FFMPEG_CMD)
                
                # Get speaker diarization data
                speaker_segments = speaker_tracker.get_speaker_data(input_path)
                
                # Check if we got any speaker segments
                if speaker_segments:
                    # Count unique speakers
                    unique_speakers = len(set(speaker_segments.values()))
                    
                    speaker_tracking_time = time.time() - speaker_tracking_start
                    print(f"[OK] Speaker tracking completed in {speaker_tracking_time:.2f} seconds")
                    print(f"[SPEECH]  Detected {unique_speakers} unique speakers")
                    logger.info(f"Speaker tracking completed: {speaker_tracking_time:.2f}s, {unique_speakers} speakers detected")
                    track_speakers_success = True
                else:
                    print(f"[WARNING] Speaker tracking failed: No speakers detected")
                    logger.warning("Speaker tracking failed: No speakers detected")
                    track_speakers = False  # Disable speaker tracking for the rest of the process
            except Exception as e:
                speaker_tracking_time = time.time() - speaker_tracking_start
                print(f"[WARNING] Speaker tracking failed: {e}")
                logger.error(f"Speaker tracking failed: {e}")
                track_speakers = False  # Disable speaker tracking for the rest of the process
        logger.info(f"Transcription completed: {transcription_time:.2f}s, Language: {detected_language}, Words: {total_words}, Phrases: {total_phrases}")

        overlay_cmds = []
        input_files = []
        input_file_count = 1  # Start from 1 because 0 is the input video
        
        # Template Processing
        print(f"\n🎨 Processing caption template...")
        logger.info(f"Processing template: {template}")
        
        # Validate template selection
        if template not in CAPTION_TEMPLATES:
            print(f"[WARNING]  Template '{template}' not found, using MrBeast as fallback")
            logger.warning(f"Template '{template}' not found, using MrBeast fallback")
            template = "MrBeast"  # Default fallback
        
        # Function to customize template with highlight color and font size
        def customize_template(template_name, custom_highlight_color=None, custom_font_size=None, custom_text_color=None):
            """Get a template by name and apply customizations"""
            template_copy = CAPTION_TEMPLATES.get(template_name, CAPTION_TEMPLATES["MrBeast"]).copy()
            
            # Check if template should ignore customizations
            ignore_custom_colors = template_copy.get("ignore_custom_colors", False)
            ignore_custom_font_size = template_copy.get("ignore_custom_font_size", False)
            has_multiple_colors = template_copy.get("has_multiple_colors", False)
            preserve_template_settings = template_copy.get("preserve_template_settings", False)
            
            # If template preserves its settings, skip all customizations
            if preserve_template_settings:
                print(f"[INFO] Template '{template_name}' preserves its original settings - skipping customizations")
                return template_copy
            
            # Check if this template uses text color instead of highlighting
            uses_text_color = False
            has_real_highlighting = False
            
            if template_copy.get("highlight_colors"):
                text_color = template_copy.get("text_color", (255, 255, 255, 255))
                highlight_colors = template_copy.get("highlight_colors", [])
                
                # Check if all highlight colors are the same as text color (no real highlighting)
                uses_text_color = all(highlight_color == text_color for highlight_color in highlight_colors)
                # Check if template has real highlighting (different colors)
                has_real_highlighting = not uses_text_color and len(highlight_colors) > 0
            else:
                # Template has no highlight_colors defined, so it uses text color only
                uses_text_color = True
                has_real_highlighting = False
            
            # Handle color customization based on template type
            # Priority: if template doesn't have real highlighting, treat any color input as text color
            if not has_real_highlighting:
                color_to_apply = custom_text_color or custom_highlight_color  # Text color takes priority
            else:
                color_to_apply = custom_highlight_color or custom_text_color  # Highlight color takes priority
            
            # Skip color customization if template ignores custom colors or has multiple colors
            if (ignore_custom_colors or has_multiple_colors) and color_to_apply:
                print(f"[INFO] Template '{template_name}' uses multiple colors - skipping color customization")
                color_to_apply = None
            
            if color_to_apply:
                try:
                    # Convert hex color string to RGBA tuple
                    if color_to_apply.startswith('#'):
                        r = int(color_to_apply[1:3], 16)
                        g = int(color_to_apply[3:5], 16)
                        b = int(color_to_apply[5:7], 16)
                        rgba_color = (r, g, b, 255)  # Full opacity
                        
                        if not has_real_highlighting:
                            # For templates without real highlighting (single color), update text color
                            # and set highlight_colors to match if they exist
                            template_copy["text_color"] = rgba_color
                            if "highlight_colors" in template_copy:
                                template_copy["highlight_colors"] = [rgba_color]
                            if "highlight_color" in template_copy:
                                template_copy["highlight_color"] = rgba_color
                            
                            color_type = "text color"
                            print(f"[OK] Applied custom {color_type} (no highlighting): {color_to_apply} → {rgba_color}")
                        else:
                            # For templates with real highlighting, update highlight colors
                            if "highlight_colors" in template_copy:
                                template_copy["highlight_colors"] = [rgba_color]
                            if "highlight_color" in template_copy:
                                template_copy["highlight_color"] = rgba_color
                            
                            # Also update text color if custom_text_color was provided
                            if custom_text_color and color_to_apply == custom_text_color:
                                template_copy["text_color"] = rgba_color
                                color_type = "text color"
                            else:
                                color_type = "highlight color"
                            
                            print(f"[OK] Applied custom {color_type}: {color_to_apply} → {rgba_color}")
                        
                        # Update the template description to reflect the custom color
                        if "description" in template_copy:
                            template_copy["description"] = f"Custom {color_to_apply} {color_type}"
                            
                    else:
                        print(f"[WARNING]  Invalid color format: {color_to_apply}, using default")
                except Exception as e:
                    print(f"[WARNING]  Error applying custom color: {str(e)}")
                    logger.warning(f"Error applying custom color: {str(e)}")
            
            # Apply custom font size if provided, but respect template-specific font size requirements
            if custom_font_size:
                try:
                    # Check if this template has special font size requirements that should be preserved
                    # Check if this template has special font size requirements that should be preserved
                    if template_name in ["Hopecore", "ChromaFusion", "Vertical Stack", "Golden Impact"]:
                        # if it's significantly different from the default (65px), indicating intentional customization
                        if int(custom_font_size) != 65:  # Only apply if user explicitly changed from default
                            # Ensure font size is within reasonable bounds (30-150px for special templates)
                            font_size = max(30, min(150, int(custom_font_size)))
                            
                            # Store the original font size for reference
                            original_font_size = template_copy.get("font_size", 65)
                            
                            # Update the font size
                            template_copy["font_size"] = font_size
                            
                            # If the template has emphasis_font_size, scale it proportionally
                            if "emphasis_font_size" in template_copy:
                                # Calculate the scale factor from the original template
                                scale_factor = template_copy["emphasis_font_size"] / original_font_size
                                template_copy["emphasis_font_size"] = int(font_size * scale_factor)
                            
                            print(f"[OK] Applied custom font size to special template: {font_size}px")
                        else:
                            print(f"[OK] Using template's default font size: {template_copy.get('font_size', 65)}px")
                    else:
                        # For regular templates, apply custom font size normally
                        # Ensure font size is within reasonable bounds (30-90px)
                        font_size = max(30, min(90, int(custom_font_size)))
                        
                        # Store the original font size for reference
                        original_font_size = template_copy.get("font_size", 65)
                        
                        # Update the font size
                        template_copy["font_size"] = font_size
                        
                        # If the template has enhanced_font_size for word-by-word mode, scale it proportionally
                        if "enhanced_font_size" in template_copy:
                            # Calculate the scale factor from the original template
                            scale_factor = template_copy["enhanced_font_size"] / original_font_size
                            template_copy["enhanced_font_size"] = int(font_size * scale_factor)
                        
                        print(f"[OK] Applied custom font size: {font_size}px")
                except Exception as e:
                    print(f"[WARNING]  Error applying custom font size: {str(e)}")
                    logger.warning(f"Error applying custom font size: {str(e)}")
            
            return template_copy
        
        # Get customized template
        selected_template = customize_template(template, highlight_color, font_size, text_color)
        
        # Use template's animation settings if defined
        if selected_template.get("animation_type") and animation_type == "none":
            animation_type = selected_template.get("animation_type")
            print(f"[ANIMATION] Using template's animation type: {animation_type}")
        
        if selected_template.get("animation_speed") and animation_speed == 1.0:
            animation_speed = selected_template.get("animation_speed")
            print(f"[ANIMATION] Using template's animation speed: {animation_speed}x")
            
        # Check for both old and new highlighting systems
        use_highlighting = (selected_template.get("highlight_color") is not None or 
                           (selected_template.get("highlight_colors") and len(selected_template.get("highlight_colors", [])) > 0))
        
        print(f"🎨 Using caption template: {template} ({selected_template.get('name', template)})")
        print(f"✨ Word highlighting: {'Enabled' if use_highlighting else 'Disabled'}")
        if use_highlighting and selected_template.get("highlight_colors"):
            colors = len(selected_template.get("highlight_colors", []))
            
            # Check if we're using a custom highlight color
            if highlight_color and colors == 1:
                # Get color components for a more descriptive message
                r, g, b, _ = selected_template.get("highlight_colors")[0]
                
                # Create a color description based on RGB values
                color_name = "Custom"
                if r > 200 and g > 200 and b < 100:
                    color_name = "Yellow"
                elif r > 200 and g < 100 and b < 100:
                    color_name = "Red"
                elif r < 100 and g > 200 and b < 100:
                    color_name = "Green"
                elif r < 100 and g < 100 and b > 200:
                    color_name = "Blue"
                elif r > 200 and g < 100 and b > 200:
                    color_name = "Purple"
                elif r > 200 and g > 100 and b < 100:
                    color_name = "Orange"
                elif r > 200 and g > 200 and b > 200:
                    color_name = "White"
                
                # Display the custom color information
                print(f"🎨 Using custom {color_name} highlight color ({highlight_color})")
            else:
                # Use the default template descriptions
                if template == "MrBeast":
                    print(f"🌈 Cycling through {colors} highlight colors (Yellow→Green→Red)")
                elif template == "Bold Green":
                    print(f"🟢 Using {colors} highlight color (Bright Green)")
                elif template == "Bold Sunshine":
                    print(f"🟡 Using {colors} highlight color (Bright Yellow)")
                elif template == "Premium Orange":
                    print(f"🟠 Using {colors} highlight color (Vibrant Orange)")
                elif template == "Minimal White":
                    print(f"⚪ Using {colors} highlight color (Pure White)")
                elif template == "Orange Meme":
                    print(f"🧡 Using uniform orange color (all text highlighted)")
                elif template == "Cinematic Quote":
                    print(f"🟡 Using {colors} highlight color (Bright Yellow for keywords)")
                elif template == "Word by Word":
                    # Check if we're using a custom text color
                    if text_color and selected_template.get("text_color"):
                        r, g, b, _ = selected_template.get("text_color")
                        
                        # Create a color description based on RGB values
                        color_name = "Custom"
                        if r > 200 and g > 200 and b < 100:
                            color_name = "Yellow"
                        elif r > 200 and g < 100 and b < 100:
                            color_name = "Red"
                        elif r < 100 and g > 200 and b < 100:
                            color_name = "Green"
                        elif r < 100 and g < 100 and b > 200:
                            color_name = "Blue"
                        elif r > 200 and g < 100 and b > 200:
                            color_name = "Purple"
                        elif r > 200 and g > 100 and b < 100:
                            color_name = "Orange"
                        elif r > 200 and g > 200 and b > 200:
                            color_name = "White"
                        
                        print(f"🎨 Using custom {color_name} text color ({text_color}) for word-by-word display")
                    else:
                        print(f"⚪ Using uniform white color (word-by-word display)")
                elif template == "esports_caption":
                    scale_factor = selected_template.get("scale_factor", 1.0)
                    print(f"🔴 Using {colors} highlight color (Red-Orange for gaming)")
                    if selected_template.get("scale_effect", False):
                        print(f"[MEASURE] Scale effect: {scale_factor}x size for highlighted words")
                elif template == "explainer_pro":
                    print(f"🟠 Using {colors} highlight color (Semi-transparent orange bars)")
                    if selected_template.get("highlight_bars", False):
                        bar_padding = selected_template.get("bar_padding", 8)
                        print(f"[STATS] Highlight bars: Enabled with {bar_padding}px padding")
                elif template == "Reaction Pop":
                    scale_factor = selected_template.get("scale_factor", 1.0)
                    print(f"🔴 Using {colors} highlight color (Pure Red for reactions)")
                    if selected_template.get("scale_effect", False):
                        print(f"[MEASURE] Scale effect: {scale_factor}x size for highlighted words")
                else:
                    print(f"🌈 Using {colors} highlight color(s)")
        print(f"[WORDS] Font size: {selected_template['font_size']}px")
        if selected_template.get("word_by_word", False):
            enhanced_size = selected_template.get("enhanced_font_size", selected_template['font_size'])
            print(f"[FONT] Word-by-word mode: Enhanced font size {enhanced_size}px (+10%)")
        if selected_template.get("uppercase", False):
            print(f"[FONT] Text case: UPPERCASE")
        elif selected_template.get("title_case", False):
            print(f"[FONT] Text case: Title Case")
        elif selected_template.get("lowercase_casual", False):
            print(f"[FONT] Text case: Casual Lowercase (random words)")
        stroke_info = f"{selected_template.get('stroke_width', 0)}px black" if selected_template.get('stroke_color') else 'None'
        print(f"🖌️  Text stroke: {stroke_info}")
        shadow_info = f"{selected_template.get('shadow_offset', (0,0))[0]}px offset" if selected_template.get('shadow_color') else 'None'
        print(f"🌑 Text shadow: {shadow_info}")
        
        # Log animation settings
        print(f"🎬 Animation: {animation_type} (speed: {animation_speed}x)")
        if selected_template.get("bounce_keywords", False):
            print(f"🔄 Bounce effect on keywords: Enabled")
        
        # Caption Generation
        print(f"\n🖼️  Generating caption images...")
        logger.info("Starting caption image generation")
        caption_generation_start = time.time()
        
        # Check if this is word-by-word template
        is_word_by_word_template = selected_template.get("word_by_word", False)
        if is_word_by_word_template:
            if animation_type == 'none':
                print(f"[FONT] Word-by-word mode: Creating individual word captions")
                logger.info("Using word-by-word caption generation mode")
                # For word-by-word template, create individual word captions
                word_counter = 0
                for phrase_idx, phrase in enumerate(phrases):
                    print(f"[WORDS] Processing phrase {phrase_idx + 1}/{total_phrases}: {len(phrase)} words")
            
                    for word_idx, word in enumerate(phrase):
                        word_text = word['word'].strip()
                        word_start = word['start']
                        word_end = word['end']
                        
                        caption_path = os.path.join(CAPTION_DIR, f"{video_id}_word_{word_counter}.png")
                        render_caption_png_wrapped(word_text, caption_path, template_name=template, custom_template=selected_template, caption_layout=caption_layout, profanity_processor=profanity_processor, word_info=[word], line_height_multiplier=line_height, word_spacing_px=word_spacing)
                        
                        # Apply vertical constraints for Hopecore and Smilecore templates
                        if selected_template.get("safe_area_top") and selected_template.get("safe_area_bottom"):
                            safe_top = selected_template.get("safe_area_top", 0.1)
                            safe_bottom = selected_template.get("safe_area_bottom", 0.1)
                            max_y_position = 1.0 - safe_bottom  # Don't exceed bottom safe area
                            
                            # Ensure caption position stays within safe area
                            adjusted_caption_position = max(safe_top, min(max_y_position, caption_position))
                            
                            print(f"[VERTICAL] Template {selected_template['name']}: Caption position adjusted from {caption_position:.3f} to {adjusted_caption_position:.3f} (safe area: {safe_top}-{max_y_position})")
                            caption_y = f"H*{adjusted_caption_position}"
                        else:
                            caption_y = f"H*{caption_position}"
                        
                        input_files.extend(["-i", caption_path])
                        
                        # Simple overlay without animation for word-by-word
                        if input_file_count == 1:
                            overlay_cmds.append(f"[0:v][{input_file_count}:v] overlay=enable='between(t,{word_start},{word_end})':x=(W-w)/2:y={caption_y} [v{input_file_count}];")
                        else:
                            overlay_cmds.append(f"[v{input_file_count-1}][{input_file_count}:v] overlay=enable='between(t,{word_start},{word_end})':x=(W-w)/2:y={caption_y} [v{input_file_count}];")
                        
                        input_file_count += 1
                        word_counter += 1
            else:
                print(f"[FFMPEG] Word-by-word mode with animation: Creating phrase-wide animated captions")
                logger.info("Using phrase-wide animation mode (word-by-word disabled during animation)")
                # When animations are enabled with word-by-word template, create phrase captions instead
                for phrase_idx, phrase in enumerate(phrases):
                    text = " ".join([w['word'] for w in phrase]).strip()
                    phrase_start = phrase[0]['start']
                    phrase_end = phrase[-1]['end']
                    
                    print(f"[WORDS] Processing animated phrase {phrase_idx + 1}/{total_phrases}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                    
                    caption_path = os.path.join(CAPTION_DIR, f"{video_id}_phrase_{phrase_idx}_animated.png")
                    render_caption_png_wrapped(text, caption_path, template_name=template, custom_template=selected_template, caption_layout=caption_layout, profanity_processor=profanity_processor, word_info=phrase, line_height_multiplier=line_height, word_spacing_px=word_spacing)
                    
                    input_files.extend(["-i", caption_path])
                    
                    # Generate animation filter for the entire phrase duration
                    overlay_filter, additional_filter = generate_animation_filter(
                        animation_type, animation_speed, phrase_start, phrase_end, 
                        "(W-w)/2", f"H*{caption_position}"
                    )
                    
                    if additional_filter and additional_filter.strip():
                        # Apply additional filter (like scale, color effects) before overlay
                        if input_file_count == 1:
                            overlay_cmds.append(f"[{input_file_count}:v] {additional_filter} [filtered_{input_file_count}];")
                            overlay_cmds.append(f"[0:v][filtered_{input_file_count}] {overlay_filter} [v{input_file_count}];")
                        else:
                            overlay_cmds.append(f"[{input_file_count}:v] {additional_filter} [filtered_{input_file_count}];")
                            overlay_cmds.append(f"[v{input_file_count-1}][filtered_{input_file_count}] {overlay_filter} [v{input_file_count}];")
                    else:
                        # Standard overlay with animation
                        if input_file_count == 1:
                            overlay_cmds.append(f"[0:v][{input_file_count}:v] {overlay_filter} [v{input_file_count}];")
                        else:
                            overlay_cmds.append(f"[v{input_file_count-1}][{input_file_count}:v] {overlay_filter} [v{input_file_count}];")
                    
                    input_file_count += 1
        else:
            # Standard phrase-based processing
            print(f"[PHRASES] Standard phrase mode: Creating phrase-based captions")
            logger.info("Using standard phrase-based caption generation")
            for phrase_idx, phrase in enumerate(phrases):
                text = " ".join([w['word'] for w in phrase]).strip()
                phrase_start = phrase[0]['start']
                phrase_end = phrase[-1]['end']
                
                print(f"[WORDS] Processing phrase {phrase_idx + 1}/{total_phrases}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                if use_highlighting and animation_type == 'none':
                    # Generate caption with word-by-word highlighting (only when no animation)
                    for word_idx, word in enumerate(phrase):
                        word_start = word['start']
                        word_end = word['end']
                        
                        caption_path = os.path.join(CAPTION_DIR, f"{video_id}_p{phrase_idx}_w{word_idx}.png")
                        render_caption_png_wrapped(text, caption_path, highlight_word_index=word_idx, template_name=template, custom_template=selected_template, caption_layout=caption_layout, profanity_processor=profanity_processor, word_info=phrase, line_height_multiplier=line_height, word_spacing_px=word_spacing)
                        
                        # Apply vertical constraints for Hopecore and Smilecore templates
                        if selected_template.get("safe_area_top") and selected_template.get("safe_area_bottom"):
                            safe_top = selected_template.get("safe_area_top", 0.1)
                            safe_bottom = selected_template.get("safe_area_bottom", 0.1)
                            max_y_position = 1.0 - safe_bottom  # Don't exceed bottom safe area
                            
                            # Ensure caption position stays within safe area
                            adjusted_caption_position = max(safe_top, min(max_y_position, caption_position))
                            
                            print(f"[VERTICAL] Template {selected_template['name']}: Caption position adjusted from {caption_position:.3f} to {adjusted_caption_position:.3f} (safe area: {safe_top}-{max_y_position})")
                            caption_y = f"H*{adjusted_caption_position}"
                        else:
                            caption_y = f"H*{caption_position}"
                        
                        # Check if this is a Hopecore, Hopelesscore, Smilecore, or ChromaFusion template and the current word is 6 or more alphabets
                        is_hopecore_or_smilecore = template in ["Hopecore", "Hopelesscore", "Smilecore", "ChromaFusion"]
                        current_word = word['word'].strip()
                        is_long_word = len(current_word) >= 6
                        
                        # For Hopecore, Smilecore, and ChromaFusion, center long words horizontally
                        if is_hopecore_or_smilecore and is_long_word:
                            print(f"[CENTERING] Word '{current_word}' is {len(current_word)} characters - centering horizontally")
                            # Always center long words
                            x_position = "(W-w)/2"
                        else:
                            # Use default horizontal positioning
                            x_position = "(W-w)/2"
                        
                        input_files.extend(["-i", caption_path])
                        
                        # Simple overlay without animation for highlighting
                        if input_file_count == 1:
                            overlay_cmds.append(f"[0:v][{input_file_count}:v] overlay=enable='between(t,{word_start},{word_end})':x={x_position}:y={caption_y} [v{input_file_count}];")
                        else:
                            overlay_cmds.append(f"[v{input_file_count-1}][{input_file_count}:v] overlay=enable='between(t,{word_start},{word_end})':x={x_position}:y={caption_y} [v{input_file_count}];")
                        
                        input_file_count += 1
                elif use_highlighting and animation_type != 'none':
                    # Check if this template uses vertical stacking with sequential word reveal
                    if selected_template.get("vertical_stack", False) and selected_template.get("sequential_word_reveal", False):
                        # Smilecore and Hopecore templates: Words appear sequentially from top to bottom
                        print(f"[FFMPEG] Creating vertical stack with sequential word reveal")
                        
                        # Calculate dynamic positioning for this phrase
                        position_cycle = ["center", "right", "left", "center"]  # Cycle through positions
                        phrase_position = position_cycle[phrase_idx % len(position_cycle)]
                        
                        # Set base horizontal position for the phrase
                        if phrase_position == "center":
                            base_x_percent = 0.5  # 50% from left (center)
                        elif phrase_position == "right":
                            base_x_percent = 0.75  # 75% from left (right side)
                        else:  # left
                            base_x_percent = 0.25  # 25% from left (left side)
                        
                        print(f"[VERTICAL] Phrase {phrase_idx + 1} positioned: {phrase_position}")
                        
                        # Create individual word captions with vertical stacking
                        for word_idx, word in enumerate(phrase):
                            word_start = word['start']
                            word_end = word['end']
                            word_text = word['word']
                            
                            # Randomly highlight some words with template-specific logic
                            import random
                            
                            if template == "Hopecore":
                                # For Hopecore: limit to maximum emphasized words per phrase (configurable)
                                max_emphasized = selected_template.get("max_emphasized_words", 2)
                                emphasized_count = sum(1 for w in phrase if w.get('is_emphasized', False))
                                should_highlight = emphasized_count < max_emphasized and random.random() < 0.5
                                if should_highlight:
                                    word['is_emphasized'] = True
                            elif template == "ChromaFusion":
                                # For ChromaFusion: use a similar approach to Hopecore but with more colorful emphasis
                                max_emphasized = selected_template.get("max_emphasized_words", 2)
                                emphasized_count = sum(1 for w in phrase if w.get('is_emphasized', False))
                                should_highlight = emphasized_count < max_emphasized and random.random() < 0.6  # Slightly higher chance
                                if should_highlight:
                                    word['is_emphasized'] = True
                            elif template == "Smilecore":
                                # For Smilecore: follow the same emphasis pattern as Hopecore
                                # Emphasize words that appear after two normal words
                                max_emphasized = selected_template.get("max_emphasized_words", 2)
                                emphasized_count = sum(1 for w in phrase if w.get('is_emphasized', False))
                                
                                # Count consecutive normal words before this one
                                consecutive_normal_words = 0
                                for i in range(word_idx - 1, -1, -1):
                                    if not phrase[i].get('is_emphasized', False):
                                        consecutive_normal_words += 1
                                    else:
                                        break
                                
                                # Emphasize if we have at least 2 normal words before this one
                                # and we haven't exceeded the maximum emphasized words limit
                                if consecutive_normal_words >= 2 and emphasized_count < max_emphasized:
                                    should_highlight = True
                                    print(f"[EMPHASIS] Emphasizing word '{word_text}' after {consecutive_normal_words} normal words")
                                # If this is the last word and no words have been emphasized yet, force emphasis
                                elif word_idx == len(phrase) - 1 and emphasized_count == 0:
                                    should_highlight = True
                                    print(f"[EMPHASIS] Forcing emphasis on last word '{word_text}' to ensure at least one emphasized word in phrase")
                                else:
                                    should_highlight = False
                                
                                if should_highlight:
                                    word['is_emphasized'] = True
                            else:
                                # For other templates: use standard highlighting chance
                                highlight_chance = 0.3
                                should_highlight = random.random() < highlight_chance
                            highlight_word_index = word_idx if should_highlight else None
                            
                            # If highlighting, choose color based on template type
                            if should_highlight and selected_template.get("highlight_colors"):
                                if template == "Hopecore":
                                    # For Hopecore: alternate between sky blue and yellow
                                    # Count how many emphasized words we've had so far in this phrase
                                    emphasized_words_so_far = sum(1 for w in phrase[:word_idx] if w.get('is_emphasized', False))
                                    highlight_color_index = emphasized_words_so_far % 2  # Alternates between 0 and 1
                                elif template == "ChromaFusion":
                                    # For ChromaFusion: cycle through all 4 colors for a vibrant effect
                                    # Count how many emphasized words we've had so far in this phrase
                                    emphasized_words_so_far = sum(1 for w in phrase[:word_idx] if w.get('is_emphasized', False))
                                    highlight_color_index = emphasized_words_so_far % 4  # Cycles through 0, 1, 2, 3
                                else:
                                    # For other templates: randomly choose from available colors
                                    highlight_color_index = random.randint(0, len(selected_template["highlight_colors"]) - 1)
                                
                                # Store the color choice for this word (we'll use it in rendering)
                                word['highlight_color_index'] = highlight_color_index
                            
                            # Create single-word caption for vertical stacking
                            single_word_text = word_text
                            caption_path = os.path.join(CAPTION_DIR, f"{video_id}_p{phrase_idx}_w{word_idx}_vertical.png")
                            
                            # Create a custom template for this word if it's highlighted
                            word_template = selected_template.copy()
                            if should_highlight and selected_template.get("use_emphasis_font", False):
                                # Use emphasis font, bigger size, and lighter weight for highlighted words
                                word_template["font_paths"] = selected_template.get("emphasis_font_paths", selected_template["font_paths"])
                                word_template["font_size"] = selected_template.get("emphasis_font_size", selected_template["font_size"])
                                word_template["font_weight"] = selected_template.get("emphasis_font_weight", selected_template.get("font_weight", 700))
                                
                                # Apply emphasis stroke settings if defined
                                if "emphasis_stroke" in selected_template:
                                    if not selected_template.get("emphasis_stroke", True):
                                        # Disable stroke for emphasized words
                                        word_template["stroke_width"] = selected_template.get("emphasis_stroke_width", 0)
                                        word_template["stroke_color"] = selected_template.get("emphasis_stroke_color", (0, 0, 0, 0))
                                    else:
                                        # Use custom emphasis stroke settings if provided
                                        word_template["stroke_width"] = selected_template.get("emphasis_stroke_width", word_template.get("stroke_width", 0))
                                        word_template["stroke_color"] = selected_template.get("emphasis_stroke_color", word_template.get("stroke_color", (0, 0, 0, 255)))
                                
                                # Add shadow lighting effect for emphasized words
                                word_template["shadow_color"] = "match_text"  # Shadow matches text color for lighting effect
                                word_template["shadow_offset"] = (0, 0)  # No offset for pure lighting effect
                                
                                if template == "Hopecore":
                                    stroke_info = f"no stroke" if word_template.get("stroke_width", 0) == 0 else f"stroke: {word_template.get('stroke_width', 0)}px"
                                    print(f"[EMPHASIS] Using Monotes font (size: {word_template['font_size']}, weight: {word_template['font_weight']}, {stroke_info}) with gentle glow for highlighted word: '{word_text}'")
                                else:
                                    stroke_info = f"no stroke" if word_template.get("stroke_width", 0) == 0 else f"stroke: {word_template.get('stroke_width', 0)}px"
                                    print(f"[EMPHASIS] Using Borisna font (size: {word_template['font_size']}, weight: {word_template['font_weight']}, {stroke_info}) with shadow lighting for highlighted word: '{word_text}'")
                            
                            # Use a custom render function for single words
                            render_caption_png_wrapped(single_word_text, caption_path, 
                                                     highlight_word_index=0 if should_highlight else None, 
                                                     template_name=template, 
                                                     custom_template=word_template, 
                                                     caption_layout=caption_layout, 
                                                     profanity_processor=profanity_processor, 
                                                     word_info=[word],  # Single word info
                                                     line_height_multiplier=line_height, 
                                                     word_spacing_px=word_spacing)
                            
                            input_files.extend(["-i", caption_path])
                            
                            # Calculate zigzag horizontal positioning based on phrase position
                            if phrase_position == "center":
                                # Center phrases stay centered (no zigzag)
                                zigzag_offset = 0
                            elif phrase_position == "right":
                                # Right phrases zigzag leftward (negative offset)
                                zigzag_offset = word_idx * -0.05  # 5% leftward shift per word
                            else:  # left
                                # Left phrases zigzag rightward (positive offset)
                                zigzag_offset = word_idx * 0.05  # 5% rightward shift per word
                            
                            word_x_percent = base_x_percent + zigzag_offset
                            
                            # Keep words within screen bounds (10% to 90%)
                            word_x_percent = max(0.1, min(0.9, word_x_percent))
                            x_position = f"W*{word_x_percent}-w/2"
                            
                            # Calculate vertical position for stacking (top to bottom)
                            # Start from top third of screen and stack downward
                            base_y_position = 0.3  # Start at 30% from top
                            
                            # ChromaFusion template: consistent spacing with extra gaps for emphasized words
                            if template == "ChromaFusion":
                                # Base spacing between all words
                                base_spacing = 0.055  # 5.5% spacing between words
                                emphasis_spacing = selected_template.get("emphasis_word_spacing", 15) / 1080  # Convert pixels to relative
                                
                                # Calculate cumulative spacing with emphasis gaps
                                cumulative_spacing = 0
                                for i in range(word_idx):
                                    cumulative_spacing += base_spacing
                                    # Add extra spacing after each emphasized word
                                    if phrase[i].get('is_emphasized', False):
                                        cumulative_spacing += emphasis_spacing
                                
                                # Add base spacing for current word
                                cumulative_spacing += base_spacing
                                
                                # Add extra spacing if current word is emphasized
                                if word.get('is_emphasized', False):
                                    cumulative_spacing += emphasis_spacing
                                    
                                word_vertical_offset = cumulative_spacing
                                print(f"[CHROMAFUSION] Word {word_idx} '{word_text}' - cumulative spacing: {cumulative_spacing:.3f}, emphasized: {word.get('is_emphasized', False)}")
                            else:
                                # Standard spacing for other templates
                                word_vertical_offset = word_idx * 0.055  # 5.5% spacing between words
                            
                            # Apply vertical constraints for Hopecore and Smilecore templates
                            if selected_template.get("safe_area_top") and selected_template.get("safe_area_bottom"):
                                safe_top = selected_template.get("safe_area_top", 0.1)
                                safe_bottom = selected_template.get("safe_area_bottom", 0.1)
                                max_y_position = 1.0 - safe_bottom  # Don't exceed bottom safe area
                                
                                # Calculate final y position
                                final_y_position = base_y_position + word_vertical_offset
                                
                                # Check if this is a Hopecore, Hopelesscore, Smilecore, or ChromaFusion template and the word is 6 or more alphabets
                                is_hopecore_or_smilecore = template in ["Hopecore", "Hopelesscore", "Smilecore", "ChromaFusion"]
                                word_text_clean = word_text.strip()
                                is_long_word = len(word_text_clean) >= 6
                                
                                # For Hopecore, Smilecore, and ChromaFusion, center long words horizontally to avoid exceeding vertical screen
                                if is_hopecore_or_smilecore and is_long_word:
                                    # Override the x position to center the word
                                    x_position = "(W-w)/2"
                                    print(f"[CENTERING] Word '{word_text_clean}' is {len(word_text_clean)} characters - centering horizontally")
                                
                                # Ensure text doesn't exceed screen bounds
                                if final_y_position > max_y_position:
                                    # Adjust the vertical offset to fit within screen
                                    adjusted_offset = max_y_position - base_y_position
                                    if word_idx > 0:
                                        adjusted_offset = min(adjusted_offset, (max_y_position - base_y_position) / len(phrase))
                                        final_y_position = base_y_position + (word_idx * adjusted_offset)
                                    else:
                                        final_y_position = max_y_position
                                
                                # Ensure minimum top margin
                                final_y_position = max(safe_top, final_y_position)
                                
                                print(f"[VERTICAL] Template {selected_template['name']}: Word {word_idx} positioned at {final_y_position:.3f} (safe area: {safe_top}-{max_y_position})")
                                y_position = f"H*{final_y_position}"
                            else:
                                y_position = f"H*{base_y_position + word_vertical_offset}"
                            
                            # Use actual word timestamp for natural speech timing
                            animation_start = word_start
                            animation_duration = 0.3 / animation_speed  # Quick fade-in
                            
                            # Generate fade-in animation for this word
                            overlay_filter, additional_filter = generate_animation_filter(
                                "fade_in", animation_speed, animation_start, animation_start + animation_duration,
                                x_position, y_position
                            )
                            
                            # Calculate phrase end time (when all words in this phrase should disappear)
                            phrase_end_time = max(w['end'] for w in phrase)
                            
                            # After fade-in, show the word normally until the entire phrase ends (kinetic typography)
                            if animation_start + animation_duration < phrase_end_time:
                                # First apply the fade-in animation, then static display until phrase ends
                                if input_file_count == 1:
                                    overlay_cmds.append(f"[0:v][{input_file_count}:v] {overlay_filter} [v{input_file_count}_temp];")
                                    overlay_cmds.append(f"[v{input_file_count}_temp][{input_file_count}:v] overlay=enable='between(t,{animation_start + animation_duration},{phrase_end_time})':x={x_position}:y={y_position} [v{input_file_count}];")
                                else:
                                    overlay_cmds.append(f"[v{input_file_count-1}][{input_file_count}:v] {overlay_filter} [v{input_file_count}_temp];")
                                    overlay_cmds.append(f"[v{input_file_count}_temp][{input_file_count}:v] overlay=enable='between(t,{animation_start + animation_duration},{phrase_end_time})':x={x_position}:y={y_position} [v{input_file_count}];")
                            else:
                                # Just the fade-in animation covers the remaining phrase duration
                                if input_file_count == 1:
                                    overlay_cmds.append(f"[0:v][{input_file_count}:v] {overlay_filter} [v{input_file_count}];")
                                else:
                                    overlay_cmds.append(f"[v{input_file_count-1}][{input_file_count}:v] {overlay_filter} [v{input_file_count}];")
                            
                            if additional_filter:
                                additional_filters.append(additional_filter)
                            
                            input_file_count += 1
                            
                            highlight_status = "highlighted" if should_highlight else "normal"
                            zigzag_direction = "centered" if phrase_position == "center" else ("leftward" if phrase_position == "right" else "rightward")
                            print(f"[{phrase_position.upper()}] Word '{word_text}' ({highlight_status}) at position {word_idx + 1}, x={word_x_percent:.1%} ({zigzag_direction}), reveals at {animation_start:.2f}s, stays until {phrase_end_time:.2f}s")
                    
                    else:
                        # Standard animated phrase caption for other templates
                        print(f"[FFMPEG] Creating animated phrase caption with highlights")
                        caption_path = os.path.join(CAPTION_DIR, f"{video_id}_p{phrase_idx}_animated.png")
                        
                        # Find the middle word to highlight for better visual effect
                        words = text.split()
                        middle_word_index = len(words) // 2 if len(words) > 1 else 0
                        
                        # If bounce_keywords is enabled, we'll highlight multiple words for better effect
                        if selected_template.get("bounce_keywords", False) and len(words) > 2:
                            # For longer phrases, highlight multiple important words
                            # Skip common words like "the", "a", "and", etc.
                            common_words = {"the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "by", "is", "are", "was", "were"}
                            highlight_candidates = []
                            
                            # Find words that are not common and are longer than 3 characters
                            for i, word in enumerate(words):
                                if word.lower() not in common_words and len(word) > 3:
                                    highlight_candidates.append(i)
                            
                            # If we found candidates, use the middle one, otherwise use the middle word
                            if highlight_candidates:
                                middle_word_index = highlight_candidates[len(highlight_candidates) // 2]
                            
                            print(f"[HIGHLIGHT] Highlighting word: '{words[middle_word_index]}' (index: {middle_word_index})")
                        
                        render_caption_png_wrapped(text, caption_path, highlight_word_index=middle_word_index, template_name=template, custom_template=selected_template, caption_layout=caption_layout, profanity_processor=profanity_processor, word_info=phrase, line_height_multiplier=line_height, word_spacing_px=word_spacing)
                        
                        input_files.extend(["-i", caption_path])
                        
                        # Generate animation filter for the entire phrase duration
                        animation_type_to_use = animation_type
                        
                        # If bounce_keywords is enabled and we're not already using bounce animation,
                        # add a slight bounce effect to the animation
                        if selected_template.get("bounce_keywords", False) and animation_type != 'bounce':
                            # Use shake animation for a subtle bounce effect on keywords
                            animation_type_to_use = 'shake'
                            print(f"[ANIMATION] Adding bounce effect to {animation_type} animation")
                        
                        overlay_filter, additional_filter = generate_animation_filter(
                            animation_type_to_use, animation_speed, phrase_start, phrase_end, 
                            "(W-w)/2", f"H*{caption_position}"
                        )
                        
                        if additional_filter and additional_filter.strip():
                            # Apply additional filter (like scale, color effects) before overlay
                            if input_file_count == 1:
                                overlay_cmds.append(f"[{input_file_count}:v] {additional_filter} [filtered_{input_file_count}];")
                                overlay_cmds.append(f"[0:v][filtered_{input_file_count}] {overlay_filter} [v{input_file_count}];")
                            else:
                                overlay_cmds.append(f"[{input_file_count}:v] {additional_filter} [filtered_{input_file_count}];")
                                overlay_cmds.append(f"[v{input_file_count-1}][filtered_{input_file_count}] {overlay_filter} [v{input_file_count}];")
                        else:
                            # Standard overlay with animation
                            if input_file_count == 1:
                                overlay_cmds.append(f"[0:v][{input_file_count}:v] {overlay_filter} [v{input_file_count}];")
                            else:
                                overlay_cmds.append(f"[v{input_file_count-1}][{input_file_count}:v] {overlay_filter} [v{input_file_count}];")
                        
                        input_file_count += 1
                else:
                    # Generate static caption for the whole phrase
                    caption_path = os.path.join(CAPTION_DIR, f"{video_id}_{phrase_idx}.png")
                    render_caption_png_wrapped(text, caption_path, template_name=template, custom_template=selected_template, caption_layout=caption_layout, profanity_processor=profanity_processor, word_info=phrase, line_height_multiplier=line_height, word_spacing_px=word_spacing)
                    
                    input_files.extend(["-i", caption_path])
                    
                    # Generate animation filter for this phrase
                    overlay_filter, additional_filter = generate_animation_filter(
                        animation_type, animation_speed, phrase_start, phrase_end, 
                        "(W-w)/2", f"H*{caption_position}"
                    )
                    
                    if additional_filter and additional_filter.strip():
                        # Apply additional filter (like scale, color effects) before overlay
                        if input_file_count == 1:
                            overlay_cmds.append(f"[{input_file_count}:v] {additional_filter} [filtered_{input_file_count}];")
                            overlay_cmds.append(f"[0:v][filtered_{input_file_count}] {overlay_filter} [v{input_file_count}];")
                        else:
                            overlay_cmds.append(f"[{input_file_count}:v] {additional_filter} [filtered_{input_file_count}];")
                            overlay_cmds.append(f"[v{input_file_count-1}][filtered_{input_file_count}] {overlay_filter} [v{input_file_count}];")
                    else:
                        # Standard overlay with animation
                        if input_file_count == 1:
                            overlay_cmds.append(f"[0:v][{input_file_count}:v] {overlay_filter} [v{input_file_count}];")
                        else:
                            overlay_cmds.append(f"[v{input_file_count-1}][{input_file_count}:v] {overlay_filter} [v{input_file_count}];")
                    
                    input_file_count += 1

        caption_generation_time = time.time() - caption_generation_start
        total_captions = input_file_count - 1
        print(f"[OK] Caption generation completed in {caption_generation_time:.2f} seconds")
        print(f"[IMAGES]  Generated {total_captions} caption images")
        print(f"[FFMPEG] Total overlay commands generated: {len(overlay_cmds)}")
        if animation_type != 'none' and overlay_cmds:
            print(f"[DETAIL] Sample overlay command: {overlay_cmds[0][:100]}...")
        logger.info(f"Caption generation completed: {caption_generation_time:.2f}s, {total_captions} images")

        # Video Processing
        print(f"\n🎬 Starting video processing with FFmpeg...")
        logger.info("Starting FFmpeg video processing")
        video_processing_start = time.time()
        
        # Create a temporary output path for intermediate processing if speaker tracking is enabled
        temp_output_path = None
        if track_speakers and track_speakers_success:
            temp_output_path = os.path.join(OUTPUT_DIR, f"{video_id}_temp.mp4")
        
        final_output_path = output_path
        
        # Standard caption processing
        last_output = f"[v{input_file_count-1}]" if overlay_cmds else "[0:v]"

        # Construct the complete filter chain including crop/scale and overlays
        # For speaker tracking, we'll use a centered crop to keep the speaker in the middle
        if track_speakers and track_speakers_success:
            # Use a centered crop for vertical video to keep the speaker in the middle
            video_filters = "crop=(in_h*9/16):in_h:iw/2-(in_h*9/16)/2:0,scale=1080:1920[scaled];"
        else:
            # Standard crop for non-speaker tracking videos
            # Use min to ensure crop width doesn't exceed input width, and center the crop
            video_filters = "crop=w=min(in_w\,in_h*9/16):h=in_h:x=(in_w-min(in_w\,in_h*9/16))/2:y=0,scale=1080:1920[scaled];"
        if overlay_cmds:
            # Filter out any empty commands to prevent empty filters
            original_count = len(overlay_cmds)
            overlay_cmds = [cmd for cmd in overlay_cmds if cmd.strip()]
            filtered_count = len(overlay_cmds)
            
            if original_count != filtered_count:
                print(f"[WARNING]  Filtered out {original_count - filtered_count} empty overlay commands")
                logger.warning(f"Filtered out {original_count - filtered_count} empty overlay commands")
            
            if overlay_cmds:
                # Replace [0:v] with [scaled] in the first overlay command
                overlay_cmds[0] = overlay_cmds[0].replace("[0:v]", "[scaled]")
                
                # Fix for FFmpeg version >4: Ensure there's no trailing semicolon in the last command
                # This prevents the "Empty filterchain" error in newer FFmpeg versions
                if overlay_cmds[-1].endswith(';'):
                    overlay_cmds[-1] = overlay_cmds[-1][:-1]
                    print(f"[FFMPEG] Removed trailing semicolon from last filter command for FFmpeg >4 compatibility")
                    logger.info("Removed trailing semicolon from last filter command for FFmpeg >4 compatibility")
                
                complete_filter = video_filters + "".join(overlay_cmds)
                print(f"[DETAIL] Complete filter preview: {complete_filter[:200]}{'...' if len(complete_filter) > 200 else ''}")
                logger.info(f"Complete filter length: {len(complete_filter)} characters")
            else:
                complete_filter = video_filters
                last_output = "[scaled]"
                print(f"[WARNING]  No overlay commands after filtering, using video filters only")
                logger.warning("No overlay commands after filtering, using video filters only")
        else:
            complete_filter = video_filters
            last_output = "[scaled]"
            print(f"[WARNING]  No overlay commands generated")
            logger.warning("No overlay commands generated")
            
        # If speaker tracking is enabled and successful, we'll use the temporary output path
        if track_speakers and track_speakers_success and temp_output_path:
            output_path = temp_output_path

        # Check if timeline expressions are being used (CUDA filters don't support them)
        uses_timeline_expressions = "enable='between(t," in complete_filter
        
        # Write filter complex to temporary file to avoid Windows command line length limits
        filter_file_path = os.path.join(UPLOAD_DIR, f"filter_{video_id}.txt")
        
        # Since we're using timeline expressions (enable='between(t,...)'), we need to use CPU-based filters
        # CUDA filters don't support timeline expressions
        if uses_timeline_expressions:
            print(f"🎨 Using CPU processing (CUDA doesn't support timeline expressions)")
            logger.info("Using CPU processing due to timeline expressions")
            
            # Write filter complex to file
            with open(filter_file_path, 'w', encoding='utf-8') as f:
                f.write(complete_filter)
            
            # Build FFmpeg command with optional background music
            ffmpeg_cmd = [
                FFMPEG_CMD,
                "-i", input_path,
                *input_files
            ]
            
            # Add background music input if provided
            if music_file_path and (music_file_path.startswith('http') or os.path.exists(music_file_path)):
                print(f"🎵 [CPU] Adding music input to FFmpeg: {music_file_path}")
                print(f"🎵 [CPU] Number of input files: {len(input_files)}")
                print(f"🎵 [CPU] Input files: {input_files[:5]}...")  # Show first 5 for debugging
                ffmpeg_cmd.extend(["-i", music_file_path])
                # Calculate correct input index: input_files contains ["-i", "file1", "-i", "file2", ...]
                # So number of actual inputs = len(input_files) / 2, plus 1 for main video input
                music_input_index = (len(input_files) // 2) + 1
                print(f"🎵 [CPU] Music input index: {music_input_index}")
                
                # Modify the filter to include audio mixing
                audio_filter = f"[{music_input_index}:a]volume={music_volume}[music_vol]; [0:a][music_vol]amix=inputs=2:duration=first:dropout_transition=2[mixed_audio]"
                print(f"🎵 [CPU] Audio filter: {audio_filter}")
                # Remove trailing semicolon from video filter and add audio filter
                if complete_filter.endswith(';'):
                    complete_filter = complete_filter[:-1]
                complete_filter += f"; {audio_filter}"
                
                # Write the updated filter to file
                print(f"🎵 [CPU] Complete filter with music: {complete_filter}")
                with open(filter_file_path, 'w', encoding='utf-8') as f:
                    f.write(complete_filter)
                
                ffmpeg_cmd.extend([
                    "-filter_complex_script", filter_file_path,
                    "-map", last_output,
                    "-map", "[mixed_audio]",  # Use mixed audio instead of original
                    "-c:v", "libx264",
                    "-c:a", VIDEO_SETTINGS["audio_codec"],
                    "-preset", VIDEO_SETTINGS["preset"],
                    "-b:v", VIDEO_SETTINGS["bitrate"],
                    output_path
                ])
            else:
                ffmpeg_cmd.extend([
                    "-filter_complex_script", filter_file_path,
                    "-map", last_output,
                    "-map", "0:a",  # Copy audio from original video
                    "-c:v", "libx264",
                    "-c:a", VIDEO_SETTINGS["audio_codec"],
                    "-preset", VIDEO_SETTINGS["preset"],
                    "-b:v", VIDEO_SETTINGS["bitrate"],
                    output_path
                ])
        else:
            # Try CUDA-accelerated version for simple captions without timeline expressions
            print(f"[LAUNCH] Using GPU-accelerated processing...")
            logger.info("Using GPU-accelerated processing")
            
            # Build FFmpeg command with optional background music for CUDA
            ffmpeg_cmd = [
                FFMPEG_CMD, "-hwaccel", "cuda",
                "-i", input_path,
                *input_files
            ]
            
            # Add background music input if provided
            if music_file_path and (music_file_path.startswith('http') or os.path.exists(music_file_path)):
                print(f"🎵 [CUDA] Adding music input to FFmpeg: {music_file_path}")
                print(f"🎵 [CUDA] Number of input files: {len(input_files)}")
                print(f"🎵 [CUDA] Input files: {input_files[:5]}...")  # Show first 5 for debugging
                ffmpeg_cmd.extend(["-i", music_file_path])
                # Calculate correct input index: input_files contains ["-i", "file1", "-i", "file2", ...]
                # So number of actual inputs = len(input_files) / 2, plus 1 for main video input
                music_input_index = (len(input_files) // 2) + 1
                print(f"🎵 [CUDA] Music input index: {music_input_index}")
                
                # Modify the filter to include audio mixing
                audio_filter = f"[{music_input_index}:a]volume={music_volume}[music_vol]; [0:a][music_vol]amix=inputs=2:duration=first:dropout_transition=2[mixed_audio]"
                print(f"🎵 [CUDA] Audio filter: {audio_filter}")
                # Remove trailing semicolon from video filter and add audio filter
                if complete_filter.endswith(';'):
                    complete_filter = complete_filter[:-1]
                complete_filter += f"; {audio_filter}"
                
                # Write the updated filter to file
                cuda_filter = complete_filter.replace("scale=", "scale_cuda=").replace("overlay", "overlay_cuda")
                print(f"🎵 [CUDA] Complete filter with music: {cuda_filter}")
                with open(filter_file_path, 'w', encoding='utf-8') as f:
                    f.write(cuda_filter)
                
                ffmpeg_cmd.extend([
                    "-filter_complex_script", filter_file_path,
                    "-map", last_output,
                    "-map", "[mixed_audio]",  # Use mixed audio instead of original
                    "-c:v", "h264_nvenc",
                    "-c:a", VIDEO_SETTINGS["audio_codec"],
                    "-preset", VIDEO_SETTINGS["preset"],
                    "-b:v", VIDEO_SETTINGS["bitrate"],
                    output_path
                ])
            else:
                # Write filter complex to file
                with open(filter_file_path, 'w', encoding='utf-8') as f:
                    f.write(complete_filter.replace("scale=", "scale_cuda=").replace("overlay", "overlay_cuda"))
                
                ffmpeg_cmd.extend([
                    "-filter_complex_script", filter_file_path,
                    "-map", last_output,
                    "-map", "0:a",  # Copy audio from original video
                    "-c:v", "h264_nvenc",
                    "-c:a", VIDEO_SETTINGS["audio_codec"],
                    "-preset", VIDEO_SETTINGS["preset"],
                    "-b:v", VIDEO_SETTINGS["bitrate"],
                    output_path
                ])

        print(f"[FFMPEG] FFmpeg command: {' '.join(ffmpeg_cmd)}")
        logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        
        try:
            print(f"⚙️  Executing FFmpeg...")
            ffmpeg_result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            video_processing_time = time.time() - video_processing_start
            
            print("Video processing completed successfully")
            logger.info(f"FFmpeg processing completed successfully in {video_processing_time:.2f} seconds")
            
            # Apply speaker tracking if enabled and initial diarization was successful
            speaker_tracking_processing_time = 0
            if track_speakers and track_speakers_success and temp_output_path and os.path.exists(temp_output_path):
                print(f"\n👥 Applying speaker tracking to video...")
                logger.info("Starting speaker tracking visual processing")
                speaker_tracking_processing_start = time.time()
                
                try:
                    # Initialize speaker tracker
                    speaker_tracker = SpeakerTracker(ffmpeg_cmd=FFMPEG_CMD)
                    
                    # Process the video with speaker tracking
                    success = speaker_tracker.process_video(temp_output_path, final_output_path, highlight_speaker=True)
                    
                    if success and os.path.exists(final_output_path):
                        # Update output path to the final path
                        output_path = final_output_path
                        
                        speaker_tracking_processing_time = time.time() - speaker_tracking_processing_start
                        print(f"[OK] Speaker tracking applied in {speaker_tracking_processing_time:.2f} seconds")
                        logger.info(f"Speaker tracking applied in {speaker_tracking_processing_time:.2f} seconds")
                    else:
                        # If speaker tracking failed, use the original output
                        print(f"[WARNING] Speaker tracking visual processing failed, using original output")
                        logger.warning("Speaker tracking visual processing failed, using original output")
                        # Copy the temp file to the final output path
                        shutil.copy2(temp_output_path, final_output_path)
                        output_path = final_output_path
                except Exception as e:
                    print(f"[WARNING] Error in speaker tracking visual processing: {e}")
                    logger.error(f"Error in speaker tracking visual processing: {e}")
                    # Copy the temp file to the final output path
                    if os.path.exists(temp_output_path) and os.path.exists(os.path.dirname(final_output_path)):
                        shutil.copy2(temp_output_path, final_output_path)
                        output_path = final_output_path
                
                # Clean up temporary file
                try:
                    if os.path.exists(temp_output_path):
                        os.remove(temp_output_path)
                except Exception as e:
                    logger.warning(f"Could not remove temporary file: {temp_output_path}, error: {e}")
                    
                # Update total speaker tracking time
                speaker_tracking_time += speaker_tracking_processing_time
            
            # Apply precise beep sounds for profanity using librosa if enabled
            beep_processing_time = 0
            if enable_profanity_filter and profanity_timestamps:
                print(f"\n🚫 Applying precise beep sounds for profanity using librosa...")
                logger.info("Starting librosa-based beep sound processing for profanity")
                beep_processing_start = time.time()
                
                try:
                    # Initialize librosa beep processor
                    beep_processor = LibrosaBeepProcessor(sample_rate=44100, beep_frequency=1000.0)
                    
                    # Create temporary output path for beep processing
                    temp_beep_output = output_path.replace('.mp4', '_temp_beep.mp4')
                    
                    # Apply precise beeps to video using librosa
                    success = beep_processor.process_video_with_beeps(
                        input_video_path=output_path,
                        output_video_path=temp_beep_output,
                        profanity_timestamps=profanity_timestamps,
                        ffmpeg_cmd=FFMPEG_CMD,
                        beep_amplitude=0.6,
                        crossfade_duration=0.05
                    )
                    
                    if success and os.path.exists(temp_beep_output):
                        # Replace original with beep-processed version
                        shutil.move(temp_beep_output, output_path)
                        beep_processing_time = time.time() - beep_processing_start
                        print(f"[OK] Precise beep sounds applied in {beep_processing_time:.2f} seconds")
                        logger.info(f"Librosa-based beep sounds applied in {beep_processing_time:.2f} seconds")
                    else:
                        print(f"[WARNING] Failed to apply beep sounds, using original audio")
                        logger.warning("Failed to apply librosa-based beep sounds, using original audio")
                            
                except Exception as e:
                    beep_processing_time = time.time() - beep_processing_start
                    print(f"[WARNING] Error in beep processing: {e}")
                    logger.error(f"Error in beep processing: {e}")
            
            # Apply watermark for free plan users
            watermark_processing_time = 0
            print(f"\n🖼️  Checking user plan for watermark: '{user_plan}'")
            if user_plan.lower() == "free":
                print(f"[IMAGES]  Applying watermark for free plan user...")
                logger.info("Starting watermark processing for free plan user")
                watermark_processing_start = time.time()
                
                # Path to the watermark image (absolute path)
                watermark_image_path = "D:\\GPU quickcap\\Clip\\App\\public\\images\\watermark.png"
                
                # Check if watermark image exists
                if os.path.exists(watermark_image_path):
                    try:
                        watermark_success = add_watermark_to_video(output_path, output_path, watermark_image_path, FFMPEG_CMD)
                        watermark_processing_time = time.time() - watermark_processing_start
                        
                        if watermark_success:
                            print(f"[OK] Watermark applied in {watermark_processing_time:.2f} seconds")
                            logger.info(f"Watermark applied in {watermark_processing_time:.2f} seconds")
                        else:
                            print(f"[WARNING] Failed to apply watermark, continuing without watermark")
                            logger.warning("Failed to apply watermark, continuing without watermark")
                            
                    except Exception as e:
                        watermark_processing_time = time.time() - watermark_processing_start
                        print(f"[WARNING] Error in watermark processing: {e}")
                        logger.error(f"Error in watermark processing: {e}")
                else:
                    print(f"[WARNING] Watermark image not found at: {watermark_image_path}")
                    logger.warning(f"Watermark image not found at: {watermark_image_path}")
            else:
                print(f"🎯 Premium user ({user_plan}) - skipping watermark")
                logger.info(f"Premium user ({user_plan}) - skipping watermark")
            
            # Get output file size
            output_size = os.path.getsize(output_path)
            output_size_mb = output_size / (1024 * 1024)
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            print(f"\n{'='*60}")
            print(f"🎉 PROCESSING COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"[STATS] PROCESSING SUMMARY:")
            print(f"   📁 Input file: {file_size_mb:.2f} MB")
            print(f"   📁 Output file: {output_size_mb:.2f} MB")
            print(f"   ⏱️  File save: {file_save_time:.2f}s")
            print(f"   🤖 Transcription: {transcription_time:.2f}s")
            print(f"   🖼️  Caption generation: {caption_generation_time:.2f}s")
            print(f"   🎬 Video processing: {video_processing_time:.2f}s")
            if track_speakers:
                print(f"   👥 Speaker tracking: {speaker_tracking_time:.2f}s")
            if enable_profanity_filter and profanity_timestamps:
                print(f"   🚫 Profanity beep processing: {beep_processing_time:.2f}s")
                print(f"   🚫 Profane words censored: {len(profanity_timestamps)}")
            if user_plan.lower() == "free" and watermark_processing_time > 0:
                print(f"   🖼️  Watermark processing: {watermark_processing_time:.2f}s")
            print(f"   ⏱️  Total time: {total_time:.2f}s")
            print(f"   🎨 Template: {template}")
            print(f"   📝 Words: {total_words}")
            print(f"   📄 Phrases: {total_phrases}")
            print(f"   🖼️  Captions: {total_captions}")
            print(f"{'='*60}")
            
            logger.info(f"Processing completed successfully - Total time: {total_time:.2f}s, Template: {template}, Words: {total_words}, Output: {output_size_mb:.2f}MB")
            
        except subprocess.CalledProcessError as e:
            video_processing_time = time.time() - video_processing_start
            total_time = time.time() - start_time
            
            print(f"\n{'='*60}")
            print(f"❌ PROCESSING FAILED!")
            print(f"{'='*60}")
            print(f"❌ FFmpeg error (exit code: {e.returncode})")
            print(f"[TIMER]  Failed after: {total_time:.2f}s")
            print(f"Command: {' '.join(e.cmd)}")
            
            logger.error(f"FFmpeg processing failed after {video_processing_time:.2f}s - Exit code: {e.returncode}")
            logger.error(f"FFmpeg command: {' '.join(e.cmd)}")
            
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
                logger.error(f"FFmpeg STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
                logger.error(f"FFmpeg STDERR: {e.stderr}")
            
            print(f"{'='*60}")
            # Clean up filter file in case of error
            try:
                if filter_file_path and os.path.exists(filter_file_path):
                    os.remove(filter_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Could not remove filter file: {filter_file_path}, error: {cleanup_error}")
            raise Exception(f"Video processing failed: {e.stderr if e.stderr else 'Unknown error'}")

        # Create a JSON file with the transcription data
        json_output_path = output_path.replace('.mp4', '.json')
        with open(json_output_path, 'w') as json_file:
            import json
            # Get the full text from the result
            full_text = result.get("text", "")
            
            # Ensure word timing information is preserved in transcription file
            transcription_data = {
                "segments": segments,
                "language": detected_language,
                "text": full_text
            }
            
            # Add word timing debug info
            total_words_with_timing = 0
            for segment in segments:
                if "words" in segment and segment["words"]:
                    total_words_with_timing += len(segment["words"])
            
            print(f"[TRANSCRIPTION] Saving transcription with {len(segments)} segments and {total_words_with_timing} words with timing")
            json.dump(transcription_data, json_file, indent=2)
        
        # Create a metadata file with the processing parameters
        metadata_output_path = output_path.replace('.mp4', '_metadata.json')
        with open(metadata_output_path, 'w') as metadata_file:
            processing_metadata = {
                "template": template,
                "font_size": font_size,
                "highlight_color": highlight_color,
                "text_color": text_color,
                "caption_layout": caption_layout,
                "use_highlighting": use_highlighting,
                "track_speakers": track_speakers,
                "caption_position": caption_position,
                "enable_profanity_filter": enable_profanity_filter,
                "profanity_filter_mode": profanity_filter_mode,
                "custom_profanity_words": custom_profanity_words,
                "animation_type": animation_type,
                "animation_speed": animation_speed,
                "line_height": line_height,
                "word_spacing": word_spacing,
                "user_plan": user_plan,
                "video_id": video_id,
                "sequence_number": sequence_number,
                "processing_timestamp": time.time(),
                "enable_emojis": enable_emojis,
                "emoji_density": emoji_density,
                "selected_template_config": selected_template  # Store the full template config
            }
            json.dump(processing_metadata, metadata_file, indent=2)
        
        # Get the base filename without extension for consistent naming
        output_basename = os.path.splitext(os.path.basename(output_path))[0]
        
        # Initialize video manager for database operations - DATABASE REMOVED
        # try:
        #     video_manager = VideoManager()
        # except Exception as e:
        #     print(f"[WARNING]  Failed to initialize video manager: {e}")
        #     logger.error(f"Failed to initialize video manager: {e}")
        #     video_manager = None
        video_manager = None  # DATABASE REMOVED
        
        # Extract user_id from form data or try to get from authentication
        if user_id and user_id.strip():
            print(f"👤 User ID from form data: {user_id}")
            logger.info(f"Processing video for user: {user_id}")
        else:
            # Try to get user_id from authentication header if available
            try:
                user_info = auth.get_user_from_request(request)
                if user_info and user_info.get('user_id'):
                    user_id = user_info['user_id']
                    print(f"👤 User ID extracted from authentication: {user_id}")
                    logger.info(f"Processing video for authenticated user: {user_id}")
                else:
                    # If we can't get a user_id from authentication, use a unique ID based on client info
                    client_ip = request.client.host
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    user_id = f"user_{client_ip}_{timestamp}"
                    print(f"👤 No user ID provided and no authentication found, using generated ID: {user_id}")
                    logger.info(f"No user ID provided and no authentication found, using generated ID: {user_id}")
            except Exception as e:
                # If there's an error in authentication, use a unique ID based on client info
                client_ip = request.client.host
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                user_id = f"user_{client_ip}_{timestamp}"
                print(f"👤 Authentication error ({str(e)}), using generated ID: {user_id}")
                logger.info(f"Authentication error, using generated ID: {user_id}")
        
        # Create video record in database - DATABASE REMOVED
        # if video_manager:
        #     try:
        #         print(f"💾 Creating video record in database...")
        #         logger.info("Creating video database record")
        #         
        #         # Get video file size and duration
        #         video_file_size = os.path.getsize(output_path)
        #         video_duration = get_video_duration(output_path)
        #         
        #         # Create video record
        #         db_video_id = video_manager.create_video_record(
        #         user_id=user_id,  # TODO: Extract from authentication
        #         filename=f"Quickcap output {sequence_number}.mp4",
        #         title=f"Quickcap Video {sequence_number}",
        #         description=f"Generated with {template} template",
        #         original_filename=file.filename,
        #         file_size=video_file_size,
        #         duration_seconds=video_duration,
        #         # R2 fields removed - using local storage only
        #         local_path=output_path,
        #         thumbnail_url=None,  # TODO: Generate thumbnail
        #         status="completed",
        #         template_used=template,
        #         processing_metadata={
        #             "template": template,
        #             "caption_position": caption_position,
        #             "font_size": font_size,
        #             "caption_layout": caption_layout,
        #             "animation_type": animation_type,
        #             "animation_speed": animation_speed,
        #             "user_plan": user_plan,
        #             "processing_time": f"{time.time() - start_time:.2f}s",
        #             "storage_type": "local"
        #         }
        #     )
        #     
        #         print(f"[OK] Video record created in database: {db_video_id}")
        #         logger.info(f"Video database record created: {db_video_id}")
        #         
        #     except Exception as e:
        #         print(f"[WARNING]  Database record creation failed: {e}")
        #         logger.error(f"Database record creation failed: {e}")
        # else:
        #     print(f"[LOADING]️  Video manager not available, skipping database record creation")
        #     logger.info("Video manager not available, skipping database record")
        
        print(f"ℹ️  Database operations disabled - video processed locally only")
        logger.info("Database operations disabled - video processed locally only")
        
        # Clean up filter file
        try:
            if filter_file_path and os.path.exists(filter_file_path):
                os.remove(filter_file_path)
        except Exception as cleanup_error:
            logger.warning(f"Could not remove filter file: {filter_file_path}, error: {cleanup_error}")
        
        # Return the video file with additional headers for the transcription
        headers = {
            "X-Transcription-File": f"{output_basename}.json",
            "X-Video-ID": video_id,
            "X-Sequence-Number": sequence_number,
            "X-Storage-Type": "local"
        }
        
        # Clean up trimmed video if it exists and is different from the input path
        if trim_video and 'trimmed_' in input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
                print(f"🧹 Cleaned up trimmed video: {input_path}")
                logger.info(f"Cleaned up trimmed video: {input_path}")
            except Exception as cleanup_error:
                print(f"⚠️  Could not remove trimmed video: {cleanup_error}")
                logger.warning(f"Could not remove trimmed video: {input_path}, error: {cleanup_error}")
        
        # Prepare headers for immediate response
        headers["X-Storage-Type"] = "local-only" if not r2_storage.enabled else "local+r2-scheduled"
        if r2_storage.enabled:
            headers["X-R2-Status"] = "upload-scheduled"
        
        # Create the FileResponse that will be returned immediately
        response = FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"Quickcap output {sequence_number}.mp4",
            headers=headers
        )
        
        # Schedule R2 upload as background task using FastAPI's BackgroundTasks
        if r2_storage.enabled:
            print(f"\n☁️  Video ready for user - R2 upload will start in background...")
            logger.info("Video ready for user - R2 upload will start in background")
            
            # Prepare metadata for R2 storage
            video_metadata = {
                'template': template,
                'duration': total_duration if 'total_duration' in locals() else 0,
                'processing_time': total_time,
                'font_size': font_size,
                'highlight_color': highlight_color,
                'text_color': text_color,
                'caption_layout': caption_layout,
                'animation_type': animation_type,
                'animation_speed': animation_speed,
                'profanity_filter': str(enable_profanity_filter),
                'file_size': output_size,
                'total_words': total_words,
                'total_phrases': total_phrases,
                'total_captions': total_captions,
                'detected_language': detected_language if 'detected_language' in locals() else 'unknown'
            }
            
            # Use a simple thread to run R2 upload after response is sent
            import threading
            def run_r2_upload():
                import asyncio
                try:
                    # Create new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(upload_to_r2_background(
                        output_path=output_path,
                        video_id=video_id,
                        video_metadata=video_metadata,
                        template=template,
                        total_time=total_time
                    ))
                except Exception as e:
                    print(f"[ERROR] Background R2 upload thread error: {e}")
                    logger.error(f"Background R2 upload thread error: {e}")
                finally:
                    loop.close()
            
            # Start the upload thread after a small delay to ensure response is sent
            upload_thread = threading.Timer(0.1, run_r2_upload)
            upload_thread.daemon = True
            upload_thread.start()
        else:
            print(f"ℹ️  R2 storage disabled - video stored locally only")
            logger.info("R2 storage disabled - video stored locally only")
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Video processing failed: {str(e)}"}
        )

@app.get("/api/transcription/{filename}")
async def get_transcription(filename: str):
    """Get transcription JSON file"""
    # Construct the path to the JSON file
    json_path = os.path.join(OUTPUT_DIR, filename)
    
    # Check if the file exists
    if not os.path.exists(json_path):
        logger.error(f"Transcription file not found: {json_path}")
        return JSONResponse(
            status_code=404,
            content={"error": "Transcription file not found", "path": json_path}
        )
    
    try:
        # Read the JSON file and return its contents
        with open(json_path, 'r') as f:
            import json
            data = json.load(f)
            return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Error reading transcription file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error reading transcription file: {str(e)}"}
        )

@app.post("/api/reprocess-video")
async def reprocess_video(request: Request):
    """Reprocess video with updated transcription by using the upload endpoint"""
    try:
        import json  # Import json module at the beginning of the function
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Initialize filter file path (will be used later)
        filter_file_path = None
        
        print(f"\n{'='*60}")
        print(f"🔄 VIDEO REPROCESSING REQUEST - {timestamp}")
        print(f"{'='*60}")
        
        # Get request data
        data = await request.json()
        updated_transcription = data.get("transcription")
        filename = data.get("filename")
        
        # Validate required fields
        if not updated_transcription or not filename:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required fields: transcription, filename"}
            )
            
        # Try to load original processing parameters from metadata file
        original_metadata = None
        if filename:
            print(f"[DETAIL] Looking for metadata file for: {filename}")
            print(f"📂 OUTPUT_DIR: {OUTPUT_DIR}")
            
            # List files in output directory for debugging
            try:
                output_files = os.listdir(OUTPUT_DIR)
                metadata_files = [f for f in output_files if f.endswith('_metadata.json')]
                print(f"[INFO] Available metadata files: {metadata_files}")
            except Exception as e:
                print(f"[WARNING]  Could not list output directory: {e}")
            
            # Look for the metadata file based on the filename
            # First try with .mp4 extension replaced
            metadata_filename = filename.replace('.mp4', '_metadata.json')
            metadata_path = os.path.join(OUTPUT_DIR, metadata_filename)
            print(f"[DETAIL] Trying metadata path: {metadata_path}")
            
            # If not found, try with .json extension replaced
            if not os.path.exists(metadata_path):
                metadata_filename = filename.replace('.json', '_metadata.json')
                metadata_path = os.path.join(OUTPUT_DIR, metadata_filename)
                print(f"[DETAIL] Trying alternative metadata path: {metadata_path}")
            
            # If still not found, try to find a metadata file that contains the base filename
            if not os.path.exists(metadata_path):
                base_filename = filename.replace('.mp4', '').replace('.json', '')
                print(f"[DETAIL] Searching for metadata files containing: {base_filename}")
                
                try:
                    for file in output_files:
                        if file.endswith('_metadata.json') and base_filename in file:
                            metadata_path = os.path.join(OUTPUT_DIR, file)
                            metadata_filename = file
                            print(f"[OK] Found matching metadata file: {file}")
                            break
                except:
                    pass
            
            # If still not found and the filename doesn't look like an output file,
            # use the most recent metadata file as fallback
            if not os.path.exists(metadata_path) and not filename.startswith('Quickcap output'):
                print(f"[DETAIL] Filename doesn't match output pattern, looking for most recent metadata file...")
                try:
                    metadata_files = [f for f in output_files if f.endswith('_metadata.json')]
                    if metadata_files:
                        # Sort by modification time (newest first)
                        metadata_files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
                        most_recent_metadata = metadata_files[0]
                        metadata_path = os.path.join(OUTPUT_DIR, most_recent_metadata)
                        metadata_filename = most_recent_metadata
                        print(f"🎯 Using most recent metadata file: {most_recent_metadata}")
                except Exception as e:
                    print(f"[WARNING]  Error finding recent metadata file: {e}")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        original_metadata = json.load(f)
                        print(f"[OK] Loaded original processing parameters from {metadata_filename}")
                except Exception as e:
                    print(f"[WARNING]  Could not load metadata file: {e}")
                    logger.warning(f"Could not load metadata file: {e}")
            else:
                print(f"[WARNING]  Metadata file not found: {metadata_path}")
                print(f"[DETAIL] Tried paths:")
                print(f"   - {filename.replace('.mp4', '_metadata.json')}")
                print(f"   - {filename.replace('.json', '_metadata.json')}")
        
        # Use request data first (for customizations), then fall back to original metadata, then defaults
        template = data.get("template") or (original_metadata.get("template") if original_metadata else "MrBeast")
        font_size = data.get("font_size") or (original_metadata.get("font_size") if original_metadata else 65)
        highlight_color = data.get("highlight_color") or (original_metadata.get("highlight_color") if original_metadata else None)
        text_color = data.get("text_color") or (original_metadata.get("text_color") if original_metadata else None)
        caption_layout = data.get("caption_layout") or (original_metadata.get("caption_layout") if original_metadata else "wrapped")
        track_speakers = data.get("track_speakers")
        if track_speakers is None:
            track_speakers = original_metadata.get("track_speakers", False) if original_metadata else False
        caption_position = data.get("caption_position") or (original_metadata.get("caption_position") if original_metadata else 0.7)
        
        # Profanity filtering parameters
        enable_profanity_filter = data.get("enable_profanity_filter")
        if enable_profanity_filter is None:
            enable_profanity_filter = original_metadata.get("enable_profanity_filter", False) if original_metadata else False
        profanity_filter_mode = data.get("profanity_filter_mode") or (original_metadata.get("profanity_filter_mode") if original_metadata else "both")
        custom_profanity_words = data.get("custom_profanity_words") or (original_metadata.get("custom_profanity_words") if original_metadata else "")
        
        # User plan parameter (default to free if not provided)
        user_plan = data.get("user_plan") or (original_metadata.get("user_plan") if original_metadata else "free")
        
        # Animation and formatting parameters
        animation_type = data.get("animation_type") or (original_metadata.get("animation_type") if original_metadata else "none")
        animation_speed = data.get("animation_speed") or (original_metadata.get("animation_speed") if original_metadata else 1.0)
        line_height = data.get("line_height") or (original_metadata.get("line_height") if original_metadata else 1.2)
        word_spacing = data.get("word_spacing") or (original_metadata.get("word_spacing") if original_metadata else 0)
        
        # Emoji settings
        enable_emojis = data.get("enable_emojis")
        if enable_emojis is None:
            enable_emojis = original_metadata.get("enable_emojis", False) if original_metadata else False
        emoji_density = data.get("emoji_density") or (original_metadata.get("emoji_density") if original_metadata else 2)
        exact_word_timestamps = data.get("exact_word_timestamps")
        if exact_word_timestamps is None:
            exact_word_timestamps = original_metadata.get("exact_word_timestamps", False) if original_metadata else False
        
        # Template config parameter - preserve original template customizations
        selected_template_config = original_metadata.get("selected_template_config") if original_metadata else None
        
        # Translation parameters
        enable_translation = data.get("enable_translation")
        if enable_translation is None:
            enable_translation = original_metadata.get("enable_translation", False) if original_metadata else False
        target_language = data.get("target_language") or (original_metadata.get("target_language") if original_metadata else "es")
        language = data.get("language") or (original_metadata.get("language") if original_metadata else "en")
        
        print(f"[WORDS] Processing parameters:")
        print(f"   - filename: {filename}")
        print(f"   - template: {template}")
        print(f"   - font_size: {font_size}px")
        print(f"   - highlight_color: {highlight_color}")
        print(f"   - text_color: {text_color}")
        print(f"   - caption_layout: {caption_layout}")
        print(f"   - track_speakers: {track_speakers}")
        print(f"   - caption_position: {caption_position}")
        print(f"   - enable_profanity_filter: {enable_profanity_filter}")
        if enable_profanity_filter:
            print(f"   - profanity_filter_mode: {profanity_filter_mode}")
        print(f"   - user_plan: {user_plan}")
        print(f"   - animation_type: {animation_type}")
        print(f"   - animation_speed: {animation_speed}x")
        print(f"   - line_height: {line_height}x")
        print(f"   - word_spacing: {word_spacing}px")
        print(f"   - enable_emojis: {enable_emojis}")
        if enable_emojis:
            print(f"   - emoji_density: {emoji_density}")
            print(f"   - exact_word_timestamps: {exact_word_timestamps}")
        print(f"   - selected_template_config: {'Available' if selected_template_config else 'None'}")
        if selected_template_config:
            print(f"     └─ Template name: {selected_template_config.get('name', 'Unknown')}")
            print(f"     └─ Description: {selected_template_config.get('description', 'No description')}")
        print(f"   - enable_translation: {enable_translation}")
        if enable_translation:
            print(f"   - target_language: {target_language}")
            print(f"   - source_language: {language}")
        print(f"   - transcription segments: {len(updated_transcription.get('segments', [])) if updated_transcription else 'None'}")
        
        # Find the original video file in outputs directory
        print(f"[DETAIL] Searching for original video file...")
        print(f"   Target filename: {filename}")
        
        original_video_path = None
        
        # First check if the file exists directly in outputs
        direct_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(direct_path):
            print(f"   Found exact match in outputs: {filename}")
            original_video_path = direct_path
        else:
            # Check for files containing the filename in outputs
            print(f"   Checking outputs directory: {OUTPUT_DIR}")
            output_files = os.listdir(OUTPUT_DIR)
            print(f"   Files in outputs: {output_files}")
            
            # First try to find an exact match with the sequence number
            for file in output_files:
                if file == filename and file.endswith('.mp4'):
                    print(f"   Found exact match in outputs: {file}")
                    original_video_path = os.path.join(OUTPUT_DIR, file)
                    break
            
            # If not found, try to find a file containing the filename
            if not original_video_path:
                for file in output_files:
                    if filename.replace('.mp4', '') in file and file.endswith('.mp4'):
                        print(f"   Found partial match in outputs: {file}")
                        original_video_path = os.path.join(OUTPUT_DIR, file)
                        break
            
            # If not found in outputs, check uploads directory
            if not original_video_path or not os.path.exists(original_video_path):
                # First try direct match in uploads
                direct_upload_path = os.path.join(UPLOAD_DIR, filename)
                if os.path.exists(direct_upload_path):
                    print(f"   Found exact match in uploads: {filename}")
                    original_video_path = direct_upload_path
                else:
                    # Check for files containing the filename in uploads
                    print(f"   Checking uploads directory: {UPLOAD_DIR}")
                    upload_files = os.listdir(UPLOAD_DIR)
                    print(f"   Files in uploads: {upload_files}")
                    
                    # Try to find any file that might match
                    for file in upload_files:
                        if file.endswith('.mp4'):
                            # Check if filename is part of the file or vice versa
                            if filename in file or file in filename:
                                original_video_path = os.path.join(UPLOAD_DIR, file)
                                print(f"   Match found in uploads: {file}")
                                break
        
        if not original_video_path or not os.path.exists(original_video_path):
            # As a last resort, try to find ANY video file in the outputs directory
            print(f"[WARNING] Could not find the specific video file. Looking for any video file...")
            
            # Find the most recent video file in the outputs directory
            video_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mp4')]
            if video_files:
                # Sort by modification time (newest first)
                video_files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
                fallback_file = video_files[0]
                original_video_path = os.path.join(OUTPUT_DIR, fallback_file)
                print(f"🔄 Using fallback video file: {fallback_file}")
            else:
                # If no video files found, return an error
                print(f"❌ No video files found in outputs directory")
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Video file not found for reprocessing: {filename}. No video files available."}
                )
        
        file_size = os.path.getsize(original_video_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"[OK] Found original video file ({file_size_mb:.2f} MB): {original_video_path}")
        
        # Create a temporary file from the original video
        try:
            print(f"📂 Reading video file: {original_video_path}")
            with open(original_video_path, 'rb') as f:
                video_data = f.read()
            
            file_size = len(video_data)
            print(f"[OK] Read {file_size / (1024 * 1024):.2f} MB of video data")
            
            # Create a file-like object from the video data
            from io import BytesIO
            file_obj = BytesIO(video_data)
            
            # Create a FastAPI UploadFile object
            from fastapi import UploadFile
            upload_file = UploadFile(filename=os.path.basename(original_video_path), file=file_obj)
            print(f"[OK] Created UploadFile object with filename: {os.path.basename(original_video_path)}")
        except Exception as e:
            print(f"❌ Error reading video file: {str(e)}")
            logger.error(f"Error reading video file: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Error reading video file: {str(e)}"}
            )
        
        # Convert custom_profanity_words to string if it's a list
        if isinstance(custom_profanity_words, list):
            custom_profanity_words = ",".join(custom_profanity_words)
        
        print(f"🔄 Processing video as new with custom transcription")
        print(f"[STATS] Transcription segments: {len(updated_transcription.get('segments', []))}")
        print(f"[STATS] Font size: {font_size}px")
        print(f"[STATS] Template: {template}")
        print(f"[STATS] Original video path: {original_video_path}")
        logger.info(f"Reprocessing video as new with custom transcription: {filename}")
        
        # Call the upload_video function directly with the same parameters
        print(f"🔄 Calling upload_video function with parameters:")
        print(f"   - template: {template}")
        print(f"   - caption_position: {float(caption_position)}")
        print(f"   - font_size: {int(font_size)}")
        print(f"   - caption_layout: {caption_layout}")
        print(f"   - track_speakers: {bool(track_speakers)}")
        print(f"   - enable_profanity_filter: {bool(enable_profanity_filter)}")
        print(f"   - profanity_filter_mode: {profanity_filter_mode}")
        print(f"   - animation_type: {animation_type}")
        print(f"   - animation_speed: {float(animation_speed)}")
        print(f"   - line_height: {float(line_height)}")
        print(f"   - word_spacing: {int(word_spacing)}")
        print(f"   - custom_transcription: {len(json.dumps(updated_transcription))} bytes")
        
        # Ensure the transcription has the correct format
        if updated_transcription:
            print(f"[WORDS] Verifying transcription format...")
            
            # Check if segments exist
            if "segments" not in updated_transcription:
                print(f"[WARNING] No segments found in transcription")
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid transcription format: no segments found"}
                )
            
            # Check if each segment has text
            for i, segment in enumerate(updated_transcription["segments"]):
                if "text" not in segment:
                    print(f"[WARNING] Segment {i} has no text")
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Invalid transcription format: segment {i} has no text"}
                    )
            
            print(f"[OK] Transcription format verified: {len(updated_transcription['segments'])} segments")
        
        try:
            # Convert the transcription to JSON string
            transcription_json = json.dumps(updated_transcription)
            print(f"[WORDS] Transcription JSON size: {len(transcription_json)} bytes")
            
            try:
                # Create a mock UploadFile object from the original video file
                import io
                from fastapi import UploadFile
                
                # Read the original video file
                with open(original_video_path, 'rb') as f:
                    file_content = f.read()
                
                # Create an UploadFile object
                file_obj = UploadFile(
                    file=io.BytesIO(file_content),
                    filename=os.path.basename(original_video_path),
                    headers={"content-type": "video/mp4"}
                )
                
                result = await upload_and_process_video(
                    request=request,
                    file=file_obj,
                    template=template,
                    caption_position=float(caption_position),
                    highlight_color=highlight_color,
                    text_color=text_color,
                    font_size=int(font_size),
                    caption_layout=caption_layout,
                    track_speakers=bool(track_speakers),
                    enable_profanity_filter=bool(enable_profanity_filter),
                    profanity_filter_mode=profanity_filter_mode,
                    custom_profanity_words=custom_profanity_words,
                    use_custom_transcription=True,
                    custom_transcription=transcription_json,
                    animation_type=animation_type,
                    animation_speed=float(animation_speed),
                    line_height=float(line_height),
                    word_spacing=int(word_spacing),
                    user_plan=user_plan,
                    trim_video=False,
                    trim_start=0.0,
                    trim_end=0.0,
                    user_id=None,
                    enable_emojis=bool(enable_emojis),
                    emoji_density=int(emoji_density),
                    exact_word_timestamps=bool(exact_word_timestamps),
                    language=language,
                    verbose=False,
                    enable_translation=bool(enable_translation),
                    target_language=target_language
                )
                print(f"[OK] upload_video function returned successfully")
                
                # Extract the video URL from the result
                if isinstance(result, FileResponse):
                    # Get the filename from the FileResponse
                    output_filename = result.filename
                    
                    # Ensure the output filename is properly formatted
                    if not output_filename.endswith('.mp4'):
                        output_filename = f"{output_filename}.mp4"
                    
                    # Create a full URL path that can be accessed by the browser
                    video_url = f"http://localhost:8080/outputs/{output_filename}"
                    
                    # Get headers from the FileResponse
                    transcription_file = result.headers.get("X-Transcription-File")
                    sequence_number = result.headers.get("X-Sequence-Number")
                    
                    # Calculate total processing time
                    total_time = time.time() - start_time
                    
                    print(f"[OK] Reprocessing completed successfully in {total_time:.2f} seconds")
                    print(f"[STATS] Output filename: {output_filename}")
                    print(f"[STATS] Video URL: {video_url}")
                    print(f"[STATS] Sequence number: {sequence_number}")
                    
                    # Verify that the output file exists
                    output_file_path = os.path.join(OUTPUT_DIR, output_filename)
                    if os.path.exists(output_file_path):
                        print(f"[OK] Output file exists: {output_file_path}")
                        print(f"[STATS] File size: {os.path.getsize(output_file_path) / (1024 * 1024):.2f} MB")
                    else:
                        print(f"[WARNING] Output file does not exist: {output_file_path}")
                        # Try to find the actual file
                        output_files = os.listdir(OUTPUT_DIR)
                        for file in output_files:
                            if sequence_number in file and file.endswith('.mp4'):
                                print(f"[OK] Found alternative output file: {file}")
                                output_filename = file
                                video_url = f"http://localhost:8080/outputs/{output_filename}"
                                break
                    
                    # Ensure the video URL is accessible
                    # Check if the file exists in the outputs directory
                    output_file_path = os.path.join(OUTPUT_DIR, output_filename)
                    if not os.path.exists(output_file_path):
                        print(f"[WARNING] Warning: Output file not found at {output_file_path}")
                        # Try to find the file with the sequence number
                        for file in os.listdir(OUTPUT_DIR):
                            if f"Quickcap output {sequence_number}" in file and file.endswith('.mp4'):
                                output_filename = file
                                video_url = f"http://localhost:8080/outputs/{output_filename}"
                                print(f"[OK] Found file with sequence number: {output_filename}")
                                break
                    
                    # Return the response with the video URL
                    response_data = {
                        "success": True,
                        "video_url": video_url,
                        "message": "Video processed successfully with updated transcription",
                        "sequence_number": sequence_number,
                        "processing_time": total_time,
                        "transcription_file": transcription_file
                    }
                    
                    print(f"[STATS] Response data: {response_data}")
                    
                    return JSONResponse(
                        content=response_data
                    )
                else:
                    # If the result is not a FileResponse, it's likely an error response
                    print(f"[WARNING] Result is not a FileResponse: {type(result)}")
                    return result
                    
            except Exception as e:
                print(f"❌ Error calling upload_video function: {str(e)}")
                logger.error(f"Error calling upload_video function: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
                
        except Exception as e:
            logger.error(f"Error reprocessing video: {str(e)}")
            import traceback
            error_traceback = traceback.format_exc()
            traceback.print_exc()
            
            print(f"❌ Reprocessing failed: {str(e)}")
            
            # Log detailed error information
            logger.error(f"Reprocessing error details:\n{error_traceback}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": f"Reprocessing failed: {str(e)}",
                    "details": str(e)
                }
            )
        
        # Convert segments to the format expected by the video processing pipeline
        words = []
        for segment in segments:
            segment_words = segment["text"].split()
            segment_duration = segment["end"] - segment["start"]
            word_duration = segment_duration / len(segment_words) if segment_words else 0
            
            for i, word_text in enumerate(segment_words):
                word_start = segment["start"] + (i * word_duration)
                word_end = segment["start"] + ((i + 1) * word_duration)
                words.append({
                    "word": word_text,
                    "start": word_start,
                    "end": word_end
                })
        
        # Initialize profanity processor for reprocessing
        custom_words = []
        if custom_profanity_words:
            custom_words = [word.strip() for word in custom_profanity_words.split(',') if word.strip()]
        
        profanity_processor = ProfanityProcessor(enable_filter=enable_profanity_filter, custom_words=custom_words, filter_mode=profanity_filter_mode)
        
        # Process words for profanity if filtering is enabled
        profanity_timestamps = []
        if enable_profanity_filter:
            print(f"\n🚫 Processing profanity filter for reprocessing...")
            logger.info("Starting profanity filtering for reprocessing")
            words, profanity_timestamps = profanity_processor.process_words_for_profanity(words)
            
            if profanity_timestamps:
                print(f"[CENSOR] Found {len(profanity_timestamps)} profane words to censor in reprocessing")
                logger.info(f"Found {len(profanity_timestamps)} profane words for censoring in reprocessing")
            else:
                print(f"[OK] No profanity detected in reprocessing transcription")
                logger.info("No profanity detected in reprocessing transcription")
        
        # Chunk words into phrases for captioning
        phrases = chunk_words(words, caption_layout, template)
        
        total_words = len(words)
        total_phrases = len(phrases)
        
        print(f"[WORDS] Processed transcription: {total_words} words, {total_phrases} phrases")
        
        # Generate captions using the updated transcription
        print(f"\n🖼️  Generating captions with updated transcription...")
        caption_generation_start = time.time()
        
        # Function to customize template with highlight color, font size, and text color
        def customize_template(template_name, custom_highlight_color=None, custom_font_size=None, custom_text_color=None):
            """Get a template by name and apply customizations"""
            template_copy = CAPTION_TEMPLATES.get(template_name, CAPTION_TEMPLATES["MrBeast"]).copy()
            
            # Check if template should ignore customizations
            preserve_template_settings = template_copy.get("preserve_template_settings", False)
            ignore_custom_colors = template_copy.get("ignore_custom_colors", False)
            ignore_custom_font_size = template_copy.get("ignore_custom_font_size", False)
            has_multiple_colors = template_copy.get("has_multiple_colors", False)
            
            # If template preserves its settings, skip all customizations
            if preserve_template_settings:
                print(f"[INFO] Template '{template_name}' preserves its original settings - skipping customizations")
                return template_copy
            
            # Check if this template uses text color instead of highlighting
            uses_text_color = False
            has_real_highlighting = False
            
            if template_copy.get("highlight_colors"):
                text_color = template_copy.get("text_color", (255, 255, 255, 255))
                highlight_colors = template_copy.get("highlight_colors", [])
                
                # Check if all highlight colors are the same as text color (no real highlighting)
                uses_text_color = all(highlight_color == text_color for highlight_color in highlight_colors)
                # Check if template has real highlighting (different colors)
                has_real_highlighting = not uses_text_color and len(highlight_colors) > 0
            else:
                # Template has no highlight_colors defined, so it uses text color only
                uses_text_color = True
                has_real_highlighting = False
            
            # Handle color customization based on template type
            # Priority: if template doesn't have real highlighting, treat any color input as text color
            if not has_real_highlighting:
                color_to_apply = custom_text_color or custom_highlight_color  # Text color takes priority
            else:
                color_to_apply = custom_highlight_color or custom_text_color  # Highlight color takes priority
            
            if color_to_apply:
                try:
                    # Convert hex color string to RGBA tuple
                    if color_to_apply.startswith('#'):
                        r = int(color_to_apply[1:3], 16)
                        g = int(color_to_apply[3:5], 16)
                        b = int(color_to_apply[5:7], 16)
                        rgba_color = (r, g, b, 255)  # Full opacity
                        
                        if not has_real_highlighting:
                            # For templates without real highlighting (single color), update text color
                            # and set highlight_colors to match if they exist
                            template_copy["text_color"] = rgba_color
                            if "highlight_colors" in template_copy:
                                template_copy["highlight_colors"] = [rgba_color]
                            if "highlight_color" in template_copy:
                                template_copy["highlight_color"] = rgba_color
                            
                            color_type = "text color"
                            print(f"[OK] Applied custom {color_type} (no highlighting): {color_to_apply} → {rgba_color}")
                        else:
                            # For templates with real highlighting, update highlight colors
                            if "highlight_colors" in template_copy:
                                template_copy["highlight_colors"] = [rgba_color]
                            if "highlight_color" in template_copy:
                                template_copy["highlight_color"] = rgba_color
                            
                            # Also update text color if custom_text_color was provided
                            if custom_text_color and color_to_apply == custom_text_color:
                                template_copy["text_color"] = rgba_color
                                color_type = "text color"
                            else:
                                color_type = "highlight color"
                            
                            print(f"[OK] Applied custom {color_type}: {color_to_apply} → {rgba_color}")
                        
                        # Update the template description to reflect the custom color
                        if "description" in template_copy:
                            template_copy["description"] = f"Custom {color_to_apply} {color_type}"
                            
                    else:
                        print(f"[WARNING]  Invalid color format: {color_to_apply}, using default")
                except Exception as e:
                    print(f"[WARNING]  Error applying custom color: {str(e)}")
                    logger.warning(f"Error applying custom color: {str(e)}")
            
            # Apply custom font size if provided, but respect template-specific font size requirements
            if custom_font_size:
                try:
                    # Check if this template has special font size requirements that should be preserved
                    # Check if this template has special font size requirements that should be preserved
                    if template_name in ["Hopecore", "ChromaFusion", "Vertical Stack", "Golden Impact"]:
                        # if it's significantly different from the default (65px), indicating intentional customization
                        if int(custom_font_size) != 65:  # Only apply if user explicitly changed from default
                            # Ensure font size is within reasonable bounds (30-150px for special templates)
                            font_size = max(30, min(150, int(custom_font_size)))
                            
                            # Store the original font size for reference
                            original_font_size = template_copy.get("font_size", 65)
                            
                            # Update the font size
                            template_copy["font_size"] = font_size
                            
                            # If the template has emphasis_font_size, scale it proportionally
                            if "emphasis_font_size" in template_copy:
                                # Calculate the scale factor from the original template
                                scale_factor = template_copy["emphasis_font_size"] / original_font_size
                                template_copy["emphasis_font_size"] = int(font_size * scale_factor)
                            
                            print(f"[OK] Applied custom font size to special template: {font_size}px")
                        else:
                            print(f"[OK] Using template's default font size: {template_copy.get('font_size', 65)}px")
                    else:
                        # For regular templates, apply custom font size normally
                        # Ensure font size is within reasonable bounds (30-90px)
                        font_size = max(30, min(90, int(custom_font_size)))
                        
                        # Store the original font size for reference
                        original_font_size = template_copy.get("font_size", 65)
                        
                        # Update the font size
                        template_copy["font_size"] = font_size
                        
                        # If the template has enhanced_font_size for word-by-word mode, scale it proportionally
                        if "enhanced_font_size" in template_copy:
                            # Calculate the scale factor from the original template
                            scale_factor = template_copy["enhanced_font_size"] / original_font_size
                            template_copy["enhanced_font_size"] = int(font_size * scale_factor)
                        
                        print(f"[OK] Applied custom font size: {font_size}px")
                except Exception as e:
                    print(f"[WARNING]  Error applying custom font size: {str(e)}")
                    logger.warning(f"Error applying custom font size: {str(e)}")
            
            return template_copy
        
        # Get customized template with highlight color, font size, and text color
        if selected_template_config:
            # Use the original template configuration if available
            selected_template = selected_template_config
            print(f"🎨 Using original template configuration")
        else:
            # Customize template with the parameters
            selected_template = customize_template(template, highlight_color, font_size, text_color)
            print(f"🎨 Customizing template with current parameters")
        caption_paths = []
        
        # Check if the template supports highlighting
        use_highlighting_for_template = use_highlighting and (
            selected_template.get("highlight_colors") or 
            selected_template.get("highlight_color")
        )
        
        print(f"🎨 Using caption template: {template} ({selected_template.get('name', template)})")
        print(f"✨ Word highlighting: {'Enabled' if use_highlighting_for_template else 'Disabled'}")
        
        # Process each phrase
        for phrase_idx, phrase in enumerate(phrases):
            phrase_text = " ".join([word["word"] for word in phrase])
            
            if use_highlighting_for_template:
                # Generate caption with word-by-word highlighting
                for word_idx, word in enumerate(phrase):
                    word_start = word['start']
                    word_end = word['end']
                    
                    caption_path = f"captions/caption_{new_video_id}_p{phrase_idx}_w{word_idx}.png"
                    
                    # Render caption with highlighting for the current word
                    render_caption_png_wrapped(
                        phrase_text, 
                        caption_path, 
                        highlight_word_index=word_idx,
                        template_name=template,
                        custom_template=selected_template,
                        caption_layout=caption_layout,
                        profanity_processor=profanity_processor,
                        word_info=phrase,
                        line_height_multiplier=line_height,
                        word_spacing_px=word_spacing
                    )
                    
                    caption_paths.append(caption_path)
            else:
                # Generate static caption for the whole phrase
                caption_path = f"captions/caption_{new_video_id}_{phrase_idx:04d}.png"
                
                # Render caption without highlighting
                render_caption_png_wrapped(
                    phrase_text, 
                    caption_path, 
                    template_name=template,
                    custom_template=selected_template,
                    caption_layout=caption_layout,
                    profanity_processor=profanity_processor,
                    word_info=phrase,
                    line_height_multiplier=line_height,
                    word_spacing_px=word_spacing
                )
                
                caption_paths.append(caption_path)
        
        caption_generation_time = time.time() - caption_generation_start
        total_captions = len(caption_paths)
        
        print(f"[OK] Generated {total_captions} caption images in {caption_generation_time:.2f} seconds")
        
        # Create video with updated captions
        print(f"\n🎬 Processing video with updated captions...")
        video_processing_start = time.time()
        
        output_path = os.path.join(OUTPUT_DIR, f"{output_filename}.mp4")
        
        # Build FFmpeg filter complex for overlaying captions
        filter_parts = []
        input_files = []
        
        # Add base video scaling
        filter_parts.append(f"[0:v]scale={VIDEO_SETTINGS['width']}:{VIDEO_SETTINGS['height']}[scaled]")
        
        # Add caption overlays
        last_output = "scaled"
        input_index = 1
        
        # Calculate Y position (70% from top by default)
        y_position = int(VIDEO_SETTINGS["height"] * 0.7)
        
        if use_highlighting_for_template:
            # For word-by-word highlighting, we need to handle each word separately
            for phrase_idx, phrase in enumerate(phrases):
                for word_idx, word in enumerate(phrase):
                    word_start = word['start']
                    word_end = word['end']
                    
                    caption_path = f"captions/caption_{new_video_id}_p{phrase_idx}_w{word_idx}.png"
                    
                    # Add input file
                    input_files.extend(["-i", caption_path])
                    
                    # Create overlay filter
                    overlay_filter = f"[{last_output}][{input_index}:v]overlay=(W-w)/2:{y_position}:enable='between(t,{word_start},{word_end})'[v{input_index}]"
                    filter_parts.append(overlay_filter)
                    
                    last_output = f"v{input_index}"
                    input_index += 1
        else:
            # For static captions, we handle each phrase
            for i, phrase in enumerate(phrases):
                caption_path = f"captions/caption_{new_video_id}_{i:04d}.png"
                input_files.extend(["-i", caption_path])
                
                start_time_overlay = phrase[0]["start"]
                end_time_overlay = phrase[-1]["end"]
                
                overlay_filter = f"[{last_output}][{input_index}:v]overlay=(W-w)/2:{y_position}:enable='between(t,{start_time_overlay},{end_time_overlay})'[v{input_index}]"
                filter_parts.append(overlay_filter)
                
                last_output = f"v{input_index}"
                input_index += 1
        
        # Join filter parts with semicolons
        complete_filter = ";".join(filter_parts)
        
        # Fix for FFmpeg version >4: Ensure there's no trailing empty filterchain
        # This prevents the "Empty filterchain" error in newer FFmpeg versions
        if complete_filter.endswith(';'):
            complete_filter = complete_filter[:-1]
            print(f"[FFMPEG] Removed trailing semicolon for FFmpeg >4 compatibility")
            logger.info("Removed trailing semicolon for FFmpeg >4 compatibility")
        
        # Write filter complex to temporary file to avoid Windows command line length limits
        filter_file_path = os.path.join(UPLOAD_DIR, f"filter_reprocess_{new_video_id}.txt")
        with open(filter_file_path, 'w', encoding='utf-8') as f:
            f.write(complete_filter)
        
        # Build FFmpeg command
        ffmpeg_cmd = [
            FFMPEG_CMD,
            "-i", original_video_path,
            *input_files,
            "-filter_complex_script", filter_file_path,
            "-map", f"[{last_output}]",
            "-map", "0:a",  # Copy audio from original video
            "-c:v", "libx264",
            "-c:a", VIDEO_SETTINGS["audio_codec"],
            "-preset", VIDEO_SETTINGS["preset"],
            "-b:v", VIDEO_SETTINGS["bitrate"],
            "-y",  # Overwrite output file
            output_path
        ]
        
        print(f"[FFMPEG] Executing FFmpeg for reprocessing...")
        logger.info(f"FFmpeg reprocessing command: {' '.join(ffmpeg_cmd)}")
        
        ffmpeg_result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        video_processing_time = time.time() - video_processing_start
        
        print("Video reprocessing completed successfully")
        
        # Apply precise beep sounds for profanity using librosa if enabled in reprocessing
        beep_processing_time = 0
        if enable_profanity_filter and profanity_timestamps:
            print(f"\n🚫 Applying precise beep sounds for profanity in reprocessing using librosa...")
            logger.info("Starting librosa-based beep sound processing for profanity in reprocessing")
            beep_processing_start = time.time()
            
            try:
                # Initialize librosa beep processor
                beep_processor = LibrosaBeepProcessor(sample_rate=44100, beep_frequency=1000.0)
                
                # Create temporary output path for beep processing
                temp_beep_output = output_path.replace('.mp4', '_temp_beep.mp4')
                
                # Apply precise beeps to video using librosa
                success = beep_processor.process_video_with_beeps(
                    input_video_path=output_path,
                    output_video_path=temp_beep_output,
                    profanity_timestamps=profanity_timestamps,
                    ffmpeg_cmd=FFMPEG_CMD,
                    beep_amplitude=0.6,
                    crossfade_duration=0.05
                )
                
                if success and os.path.exists(temp_beep_output):
                    # Replace original with beep-processed version
                    shutil.move(temp_beep_output, output_path)
                    beep_processing_time = time.time() - beep_processing_start
                    print(f"[OK] Precise beep sounds applied in reprocessing in {beep_processing_time:.2f} seconds")
                    logger.info(f"Librosa-based beep sounds applied in reprocessing in {beep_processing_time:.2f} seconds")
                else:
                    print(f"[WARNING] Failed to apply beep sounds in reprocessing, using original audio")
                    logger.warning("Failed to apply librosa-based beep sounds in reprocessing, using original audio")
                        
            except Exception as e:
                beep_processing_time = time.time() - beep_processing_start
                print(f"[WARNING] Error in beep processing during reprocessing: {e}")
                logger.error(f"Error in beep processing during reprocessing: {e}")
        
        # Create updated JSON file with the new transcription
        json_output_path = output_path.replace('.mp4', '.json')
        with open(json_output_path, 'w') as json_file:
            import json
            json.dump(updated_transcription, json_file)
        
        # Create metadata file with the processing parameters for the reprocessed video
        metadata_output_path = output_path.replace('.mp4', '_metadata.json')
        with open(metadata_output_path, 'w') as metadata_file:
            reprocessing_metadata = {
                "template": template,
                "font_size": font_size,
                "highlight_color": highlight_color,
                "text_color": text_color,
                "caption_layout": caption_layout,
                "use_highlighting": use_highlighting,
                "track_speakers": track_speakers,
                "caption_position": caption_position,
                "enable_profanity_filter": enable_profanity_filter,
                "profanity_filter_mode": profanity_filter_mode,
                "custom_profanity_words": custom_profanity_words,
                "animation_type": animation_type,
                "animation_speed": animation_speed,
                "line_height": line_height,
                "word_spacing": word_spacing,
                "user_plan": user_plan,
                "video_id": new_video_id,
                "sequence_number": sequence_number,
                "processing_timestamp": time.time(),
                "selected_template_config": selected_template,  # Store the template config used
                "is_reprocessed": True,
                "original_filename": filename  # Keep reference to original file
            }
            json.dump(reprocessing_metadata, metadata_file, indent=2)
        
        # Clean up temporary caption files (keep original video)
        try:
            for caption_path in caption_paths:
                if os.path.exists(caption_path):
                    os.remove(caption_path)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary caption files: {e}")
        
        # Get output file info
        output_size = os.path.getsize(output_path)
        output_size_mb = output_size / (1024 * 1024)
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"🎉 REPROCESSING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"[STATS] REPROCESSING SUMMARY:")
        print(f"   📁 Output file: {output_size_mb:.2f} MB")
        print(f"   🖼️  Caption generation: {caption_generation_time:.2f}s")
        if enable_profanity_filter and profanity_timestamps:
            print(f"   🚫 Profanity beep processing: {beep_processing_time:.2f}s")
            print(f"   🚫 Profane words censored: {len(profanity_timestamps)}")
        print(f"   🎬 Video processing: {video_processing_time:.2f}s")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   📝 Words: {total_words}")
        print(f"   📄 Phrases: {total_phrases}")
        print(f"   🖼️  Captions: {total_captions}")
        print(f"{'='*60}")
        
        logger.info(f"Video reprocessing completed successfully - Total time: {total_time:.2f}s, Output: {output_size_mb:.2f}MB")
        
        # Clean up filter file
        try:
            if filter_file_path and os.path.exists(filter_file_path):
                os.remove(filter_file_path)
        except Exception as cleanup_error:
            logger.warning(f"Could not remove filter file: {filter_file_path}, error: {cleanup_error}")
        
        # Return the new video URL
        # Construct the video URL based on the server's base URL
        base_url = str(request.base_url).rstrip('/')
        video_url = f"{base_url}/outputs/{output_filename}.mp4"
        
        return JSONResponse(content={
            "success": True,
            "video_url": video_url,
            "message": "Video reprocessed successfully",
            "sequence_number": sequence_number,
            "processing_time": total_time,
            "transcription_file": f"{output_filename}.json"
        })
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg reprocessing failed - Exit code: {e.returncode}")
        logger.error(f"FFmpeg STDERR: {e.stderr}")
        # Clean up filter file in case of error
        try:
            if filter_file_path and os.path.exists(filter_file_path):
                os.remove(filter_file_path)
        except Exception as cleanup_error:
            logger.warning(f"Could not remove filter file: {filter_file_path}, error: {cleanup_error}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Video reprocessing failed: {e.stderr if e.stderr else 'FFmpeg error'}"}
        )
    except Exception as e:
        logger.error(f"Error reprocessing video: {str(e)}")
        # Clean up filter file in case of error
        try:
            if filter_file_path and os.path.exists(filter_file_path):
                os.remove(filter_file_path)
        except Exception as cleanup_error:
            logger.warning(f"Could not remove filter file: {filter_file_path}, error: {cleanup_error}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Video reprocessing failed: {str(e)}"}
        )

@app.post("/api/generate-titles")
async def generate_viral_titles(request: Request):
    """Generate viral titles based on video transcription using Hugging Face Transformers"""
    try:
        data = await request.json()
        transcription = data.get("transcription", "")
        platform = data.get("platform", "youtube")
        
        if not transcription or len(transcription.strip()) < 10:
            return JSONResponse(
                status_code=400,
                content={"error": "Transcription is required and must be at least 10 characters long"}
            )
        
        # Generate unique viral titles using the new title generator
        logger.info(f"Generating titles for platform: {platform}, transcription length: {len(transcription)}")
        
        # Use the title generator
        titles = generate_titles(transcription, platform)
        
        logger.info(f"Successfully generated {len(titles)} titles for platform: {platform}")
        
        return JSONResponse(
            status_code=200,
            content={"titles": titles, "platform": platform}
        )
        
    except Exception as e:
        logger.error(f"Error generating viral titles: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate viral titles: {str(e)}"}
        )

# The old GPT-2 and Hugging Face pipeline functions have been replaced
# with the new standalone title_generator module for better reliability

@app.get("/api/demos")
async def get_demos():
    """Get list of available demo projects"""
    try:
        demo_index_path = os.path.join("d:", "GPU quickcap", "Clip", "App", "public", "demo", "index.json")
        
        if os.path.exists(demo_index_path):
            with open(demo_index_path, 'r', encoding='utf-8') as f:
                demo_data = json.load(f)
            return demo_data
        else:
            # Return default demo structure if index.json doesn't exist
            return {
                "demos": [
                    {
                        "id": "demo01",
                        "title": "Perfect Pasta Carbonara Tutorial",
                        "description": "A cooking tutorial showcasing complete QuickCap workflow",
                        "category": "Cooking",
                        "duration": "5:24",
                        "status": "active"
                    }
                ],
                "statistics": {
                    "total_demos": 1,
                    "categories": ["Cooking"]
                }
            }
    except Exception as e:
        logger.error(f"Error loading demos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load demos: {str(e)}")

@app.get("/api/demos/{demo_id}")
async def get_demo_details(demo_id: str):
    """Get detailed information about a specific demo"""
    try:
        demo_path = os.path.join("d:", "GPU quickcap", "Clip", "App", "public", "demo", demo_id)
        
        if not os.path.exists(demo_path):
            raise HTTPException(status_code=404, detail=f"Demo {demo_id} not found")
        
        # Load all demo files
        demo_data = {}
        
        # Load metadata
        metadata_path = os.path.join(demo_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                demo_data["metadata"] = json.load(f)
        
        # Add GPU availability logging before transcription
        logger.info("Checking for GPU availability...")
        try:
            import torch
            is_cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {is_cuda_available}")
            if is_cuda_available:
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("CUDA not available, transcription will run on CPU.")
        except ImportError:
            logger.warning("PyTorch not found, cannot check for GPU. Transcription will run on CPU.")
        except Exception as e:
            logger.error(f"An error occurred while checking for GPU: {e}")

        # Load transcription
        transcription_path = os.path.join(demo_path, "transcription.json")
        if os.path.exists(transcription_path):
            with open(transcription_path, 'r', encoding='utf-8') as f:
                demo_data["transcription"] = json.load(f)
        
        # Load viral titles
        titles_path = os.path.join(demo_path, "viral_titles.json")
        if os.path.exists(titles_path):
            with open(titles_path, 'r', encoding='utf-8') as f:
                demo_data["viral_titles"] = json.load(f)
        
        # Load subtitles
        subtitles_path = os.path.join(demo_path, "subtitles.srt")
        if os.path.exists(subtitles_path):
            with open(subtitles_path, 'r', encoding='utf-8') as f:
                demo_data["subtitles"] = f.read()
        
        # Check for video files
        input_path = os.path.join(demo_path, "input")
        processed_path = os.path.join(demo_path, "processed")
        
        demo_data["files"] = {
            "input_available": os.path.exists(input_path) and len(os.listdir(input_path)) > 0,
            "processed_available": os.path.exists(processed_path) and len(os.listdir(processed_path)) > 0
        }
        
        if demo_data["files"]["input_available"]:
            input_files = os.listdir(input_path)
            demo_data["files"]["input_files"] = input_files
        
        if demo_data["files"]["processed_available"]:
            processed_files = os.listdir(processed_path)
            demo_data["files"]["processed_files"] = processed_files
        
        return demo_data
        
    except Exception as e:
        logger.error(f"Error loading demo {demo_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load demo: {str(e)}")

@app.get("/api/demos/{demo_id}/download/{file_type}")
async def download_demo_file(demo_id: str, file_type: str):
    """Download demo files (input, processed, subtitles, etc.)"""
    try:
        demo_path = os.path.join("d:", "GPU quickcap", "Clip", "App", "public", "demo", demo_id)
        
        if not os.path.exists(demo_path):
            raise HTTPException(status_code=404, detail=f"Demo {demo_id} not found")
        
        file_path = None
        media_type = "application/octet-stream"
        filename = f"{demo_id}_{file_type}"
        
        if file_type == "input":
            input_dir = os.path.join(demo_path, "input")
            if os.path.exists(input_dir):
                files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
                if files:
                    file_path = os.path.join(input_dir, files[0])
                    media_type = "video/mp4"
                    filename = files[0]
        
        elif file_type == "processed":
            processed_dir = os.path.join(demo_path, "processed")
            if os.path.exists(processed_dir):
                files = [f for f in os.listdir(processed_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
                if files:
                    file_path = os.path.join(processed_dir, files[0])
                    media_type = "video/mp4"
                    filename = files[0]
        
        elif file_type == "subtitles":
            file_path = os.path.join(demo_path, "subtitles.srt")
            media_type = "text/plain"
            filename = f"{demo_id}_subtitles.srt"
        
        elif file_type == "transcription":
            file_path = os.path.join(demo_path, "transcription.json")
            media_type = "application/json"
            filename = f"{demo_id}_transcription.json"
        
        elif file_type == "viral_titles":
            file_path = os.path.join(demo_path, "viral_titles.json")
            media_type = "application/json"
            filename = f"{demo_id}_viral_titles.json"
        
        elif file_type == "metadata":
            file_path = os.path.join(demo_path, "metadata.json")
            media_type = "application/json"
            filename = f"{demo_id}_metadata.json"
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {file_type} not found for demo {demo_id}")
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"Error downloading demo file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")

def extract_key_content(transcription):
    """Extract key content summary from transcription for GPT-2 prompt"""
    # Take first 200 characters and clean up
    summary = transcription[:200].strip()
    
    # Remove filler words and clean up
    filler_words = ['um', 'uh', 'like', 'you know', 'so', 'basically', 'actually', 'literally']
    words = summary.split()
    cleaned_words = [word for word in words if word.lower() not in filler_words]
    
    # Rejoin and ensure it ends properly
    cleaned_summary = ' '.join(cleaned_words)
    if not cleaned_summary.endswith(('.', '!', '?')):
        cleaned_summary += '.'
    
    return cleaned_summary

def generate_ai_enhanced_titles(transcription, platform, summary):
    """Generate AI-enhanced titles using content analysis and pattern recognition"""
    import re
    import random
    
    # Advanced content analysis
    content_analysis = analyze_content_for_titles(transcription)
    
    # Get platform-specific viral patterns
    viral_patterns = get_viral_title_patterns(platform)
    
    # Generate titles using AI insights
    ai_titles = []
    
    # Pattern 1: Question-based titles
    if content_analysis.get('has_questions', False):
        question_templates = [
            f"Why {content_analysis['main_topic']} is {content_analysis['sentiment']}?",
            f"What if {content_analysis['main_topic']} Could {content_analysis['action']}?",
            f"How {content_analysis['main_topic']} Changed Everything"
        ]
        ai_titles.extend(random.sample(question_templates, min(2, len(question_templates))))
    
    # Pattern 2: Emotional titles
    if content_analysis.get('emotional_words'):
        emotion = content_analysis['emotional_words'][0]
        emotional_templates = [
            f"This {content_analysis['main_topic']} Will Make You {emotion}",
            f"The {emotion} Truth About {content_analysis['main_topic']}",
            f"I Was {emotion} When I Discovered {content_analysis['main_topic']}"
        ]
        ai_titles.extend(random.sample(emotional_templates, min(2, len(emotional_templates))))
    
    # Pattern 3: Number-based titles
    if content_analysis.get('numbers'):
        number = content_analysis['numbers'][0]
        number_templates = [
            f"{number} {content_analysis['main_topic']} Secrets Nobody Talks About",
            f"{number} Ways {content_analysis['main_topic']} Will Change Your Life",
            f"I Tried {content_analysis['main_topic']} for {number} Days - Here's What Happened"
        ]
        ai_titles.extend(random.sample(number_templates, min(2, len(number_templates))))
    
    # Pattern 4: Trend-based titles
    trend_templates = [
        f"{content_analysis['main_topic']} is Taking Over {platform.title()}",
        f"Everyone is Talking About {content_analysis['main_topic']} - Here's Why",
        f"The {content_analysis['main_topic']} Trend That's Breaking the Internet"
    ]
    ai_titles.extend(random.sample(trend_templates, min(2, len(trend_templates))))
    
    # Clean and format titles
    cleaned_titles = []
    for title in ai_titles:
        cleaned_title = clean_generated_title(title, platform)
        if cleaned_title and len(cleaned_title) > 10:
            cleaned_titles.append(cleaned_title)
    
    return cleaned_titles[:5]

def analyze_content_for_titles(transcription):
    """Analyze content to extract meaningful patterns for title generation"""
    import re
    
    # Extract key elements
    words = re.findall(r'\b\w+\b', transcription.lower())
    sentences = re.split(r'[.!?]+', transcription)
    
    # Find main topic (most frequent meaningful word)
    meaningful_words = [word for word in words if len(word) > 4 and word not in {
        'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 
        'said', 'each', 'which', 'their', 'time', 'about', 'would', 'there', 
        'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first'
    }]
    
    word_freq = {}
    for word in meaningful_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    main_topic = max(word_freq.items(), key=lambda x: x[1])[0].title() if word_freq else "This"
    
    # Detect emotional words
    emotional_words = []
    emotion_patterns = {
        'amazing': ['amazing', 'incredible', 'awesome', 'fantastic'],
        'shocking': ['shocking', 'unbelievable', 'mind-blowing', 'crazy'],
        'surprised': ['surprised', 'stunned', 'amazed', 'blown away'],
        'excited': ['excited', 'thrilled', 'pumped', 'energized']
    }
    
    for emotion, patterns in emotion_patterns.items():
        if any(pattern in transcription.lower() for pattern in patterns):
            emotional_words.append(emotion)
    
    # Extract numbers
    numbers = re.findall(r'\b\d+\b', transcription)
    
    # Detect questions
    has_questions = '?' in transcription
    
    # Determine sentiment
    positive_words = ['good', 'great', 'amazing', 'awesome', 'fantastic', 'excellent', 'perfect', 'love', 'best']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing']
    
    pos_count = sum(1 for word in positive_words if word in transcription.lower())
    neg_count = sum(1 for word in negative_words if word in transcription.lower())
    
    if pos_count > neg_count:
        sentiment = "Revolutionary"
    elif neg_count > pos_count:
        sentiment = "Controversial"
    else:
        sentiment = "Game-Changing"
    
    # Detect action words
    action_words = ['learn', 'discover', 'find', 'create', 'build', 'make', 'solve', 'fix', 'improve']
    detected_actions = [word for word in action_words if word in transcription.lower()]
    main_action = detected_actions[0].title() if detected_actions else "Transform"
    
    return {
        'main_topic': main_topic,
        'emotional_words': emotional_words,
        'numbers': numbers,
        'has_questions': has_questions,
        'sentiment': sentiment,
        'action': main_action
    }

def get_viral_title_patterns(platform):
    """Get platform-specific viral title patterns"""
    patterns = {
        'youtube': {
            'hooks': ['🔥', '💯', '🤯', '😱', '⚡'],
            'power_words': ['SECRET', 'SHOCKING', 'REVEALED', 'EXPOSED', 'HIDDEN'],
            'formats': ['How to', 'Why', 'What if', 'The truth about', 'Nobody talks about']
        },
        'instagram': {
            'hooks': ['✨', '💫', '🔥', '💯', '🌟'],
            'power_words': ['obsessed', 'vibes', 'energy', 'mood', 'aesthetic'],
            'formats': ['POV:', 'Tell me', 'Not me', 'This hits different', 'Main character']
        },
        'tiktok': {
            'hooks': ['🤯', '😭', '💀', '🔥', '✨'],
            'power_words': ['viral', 'trending', 'fyp', 'mindblown', 'facts'],
            'formats': ['POV:', 'Wait for it', 'Tell me why', 'This is why', 'Main character']
        },
        'twitter': {
            'hooks': ['🧵', '🔥', '💯', '🚨', '⚡'],
            'power_words': ['thread', 'unpopular opinion', 'breaking', 'plot twist', 'discourse'],
            'formats': ['Thread:', 'Unpopular opinion:', 'Breaking:', 'Plot twist:', 'Hot take:']
        }
    }
    
    return patterns.get(platform, patterns['youtube'])

def clean_generated_title(title, platform):
    """Clean and format the generated title"""
    # Remove common GPT-2 artifacts
    title = title.split('\n')[0]  # Take only first line
    title = title.split('.')[0]   # Remove trailing sentences
    title = title.strip('"\'')    # Remove quotes
    
    # Remove incomplete sentences
    if title.endswith(('and', 'or', 'but', 'the', 'a', 'an')):
        words = title.split()
        title = ' '.join(words[:-1])
    
    # Add platform-specific formatting
    if platform == 'youtube' and not any(emoji in title for emoji in ['🔥', '💯', '😱', '🤯']):
        if 'shocking' in title.lower() or 'amazing' in title.lower():
            title = f"🤯 {title}"
        elif 'secret' in title.lower() or 'hidden' in title.lower():
            title = f"🔥 {title}"
    
    elif platform == 'instagram' and not any(emoji in title for emoji in ['✨', '💫', '🔥', '💯']):
        title = f"✨ {title} 💫"
    
    elif platform == 'tiktok' and '#' not in title:
        title = f"{title} #viral"
    
    # Capitalize first letter
    if title:
        title = title[0].upper() + title[1:] if len(title) > 1 else title.upper()
    
    return title

def generate_enhanced_fallback_titles(transcription, platform):
    """Enhanced fallback title generation with content analysis"""
    import re
    import random
    
    # Extract meaningful content
    words = re.findall(r'\b\w{4,}\b', transcription.lower())
    
    # Common words to filter out
    common_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'about', 'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'can', 'still', 'should', 'after', 'being', 'now', 'made', 'before', 'here', 'through', 'when', 'where', 'much', 'some', 'these', 'many', 'then', 'them', 'well', 'were'}
    
    # Get meaningful words
    meaningful_words = [word for word in words if word not in common_words and len(word) > 3]
    
    # Count word frequency
    word_count = {}
    for word in meaningful_words:
        word_count[word] = word_count.get(word, 0) + 1
    
    # Get top keywords
    top_keywords = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:5]
    main_topic = top_keywords[0][0].title() if top_keywords else "This Topic"
    
    # Extract first meaningful sentence
    sentences = re.split(r'[.!?]+', transcription)
    first_sentence = next((s.strip() for s in sentences if len(s.strip()) > 20), "")
    
    # Platform-specific enhanced templates
    templates = {
        'youtube': [
            f"🔥 The {main_topic} Secret That Will Blow Your Mind!",
            f"Why {main_topic} Is Taking Over The Internet (MUST WATCH)",
            f"I Tried {main_topic} For 30 Days - Here's What Happened",
            f"The Shocking Truth About {main_topic} Nobody Talks About",
            f"How {main_topic} Changed Everything (You Won't Believe This)"
        ],
        'instagram': [
            f"✨ {main_topic} hits different when you know this 🤯",
            f"POV: You just discovered the {main_topic} secret 💫",
            f"Tell me you're obsessed with {main_topic} without telling me 😍",
            f"This {main_topic} hack will change your life... save this! 📌",
            f"Not me crying over how good this {main_topic} content is 🥺✨"
        ],
        'tiktok': [
            f"Wait for the {main_topic} plot twist... 🤯 #viral #fyp",
            f"{main_topic} is the reason I love this app 💯 #trending",
            f"Tell me why {main_topic} actually works like this 😭 #mindblown",
            f"POV: {main_topic} just made perfect sense 🧠✨ #facts",
            f"Main character energy: {main_topic} edition 💅 #confidence"
        ],
        'twitter': [
            f"This {main_topic} thread will change how you see everything 🧵",
            f"Unpopular opinion: {main_topic} is actually revolutionary",
            f"Breaking: {main_topic} just hit different and I'm here for it",
            f"Plot twist: {main_topic} was the answer all along",
            f"The {main_topic} discourse we needed in 2024"
        ]
    }
    
    # Add some randomization to make titles unique
    selected_templates = templates.get(platform, templates['youtube'])
    random.shuffle(selected_templates)
    
    return selected_templates

@app.get("/outputs/{filename}")
async def serve_output_file(filename: str):
    """Serve output video files"""
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        logger.error(f"Output file not found: {file_path}")
        return JSONResponse(
            status_code=404,
            content={"error": "File not found"}
        )
    
    # Determine media type based on file extension
    if filename.endswith('.mp4'):
        media_type = "video/mp4"
    elif filename.endswith('.json'):
        media_type = "application/json"
    else:
        media_type = "application/octet-stream"
    
    return FileResponse(file_path, media_type=media_type, filename=filename)

# API test endpoint
@app.get("/api/test")
async def api_test():
    """Simple API test endpoint"""
    return JSONResponse(content={"status": "ok", "message": "API is working"})

# Translation API Endpoints
# Translation endpoints removed

@app.get("/api/status")
async def get_status():
    """Get API status"""
    return {
        "status": "online",
        "version": "1.0.0",
        "whisper_model": "small",
        "cuda_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting GPU QuickCap Server on port 8080...")
    logger.info("Starting GPU QuickCap Server on port 8080")
    print("📝 Logs will be saved to: gpu_quickcap.log")
    print("🌐 Access the web interface at: http://localhost:8080")
    print("📊 Server logs and processing details will appear below...")
    print("="*60)
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)

