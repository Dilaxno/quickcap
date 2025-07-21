"""
Upload status module to check if the server is ready to accept uploads
"""

import time
import logging
import os
import shutil
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["upload"])

@router.get("/upload-status")
async def upload_status():
    """Check if the upload endpoint is ready to accept requests"""
    try:
        # Check if upload directory exists and is writable
        upload_dir = "uploads"  # Default upload directory
        dir_exists = os.path.isdir(upload_dir)
        
        # Create directory if it doesn't exist
        if not dir_exists:
            try:
                os.makedirs(upload_dir, exist_ok=True)
                dir_exists = True
                logger.info(f"Created upload directory: {upload_dir}")
            except Exception as e:
                logger.error(f"Failed to create upload directory: {e}")
        
        dir_writable = os.access(upload_dir, os.W_OK) if dir_exists else False
        
        # Check disk space
        try:
            disk_usage = shutil.disk_usage(upload_dir if dir_exists else ".")
            free_space_gb = disk_usage.free / (1024 * 1024 * 1024)  # Convert to GB
            has_enough_disk = free_space_gb > 1.0  # At least 1GB free
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            free_space_gb = 0
            has_enough_disk = False
        
        status = "ready" if (dir_exists and dir_writable and has_enough_disk) else "not_ready"
        
        return JSONResponse(content={
            "status": status,
            "timestamp": time.time(),
            "checks": {
                "upload_dir_exists": dir_exists,
                "upload_dir_writable": dir_writable,
                "free_disk_space_gb": round(free_space_gb, 2),
                "has_enough_disk": has_enough_disk
            }
        })
    except Exception as e:
        logger.error(f"Error checking upload status: {e}")
        # Return a more graceful error response
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error checking upload status: {str(e)}",
                "timestamp": time.time()
            }
        )