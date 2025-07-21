"""
Test upload module to diagnose upload issues
"""

import os
import time
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["test"])

@router.post("/test-upload")
async def test_upload(
    file: UploadFile = File(...),
    test_param: str = Form("test")
):
    """
    Test upload endpoint to diagnose upload issues
    """
    try:
        # Create test uploads directory if it doesn't exist
        test_dir = "test_uploads"
        os.makedirs(test_dir, exist_ok=True)
        
        # Generate a unique filename
        timestamp = int(time.time())
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(test_dir, filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Log success
        file_size = os.path.getsize(file_path)
        logger.info(f"Test upload successful: {file_path} ({file_size} bytes)")
        
        # Return success response
        return JSONResponse(content={
            "status": "success",
            "message": "Test upload successful",
            "filename": filename,
            "file_size": file_size,
            "test_param": test_param,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Test upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test upload failed: {str(e)}")