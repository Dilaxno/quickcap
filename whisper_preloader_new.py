"""
Simple and Reliable Whisper Model Preloader
Focuses on reliability over complexity
Lazy loading - only loads model when needed for transcription
"""
import os
import time
import threading
import logging
import torch
import whisper
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class WhisperPreloader:
    """
    Simple, reliable Whisper model preloader that actually works.
    Removes complexity and focuses on core functionality.
    Lazy loading - only loads model when needed for transcription.
    """
    
    def __init__(self):
        self.model: Optional[whisper.Whisper] = None
        self.model_name = os.getenv("WHISPER_MODEL", "small.en")
        self.model_loaded = False
        self.loading_lock = threading.Lock()
        self.load_error: Optional[str] = None
        
        # Simple CUDA check
        try:
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("CUDA not available, using CPU")
        except Exception as e:
            logger.warning(f"CUDA check failed: {e}")
            self.cuda_available = False
    
    def load_model(self) -> bool:
        """
        Load the Whisper model. Simple, reliable, no fancy tricks.
        """
        if self.model_loaded:
            return True
            
        with self.loading_lock:
            # Double-check after acquiring lock
            if self.model_loaded:
                return True
                
            try:
                logger.info(f"Loading Whisper model: {self.model_name}")
                start_time = time.time()
                
                # Clear any previous errors
                self.load_error = None
                
                # Load the model
                self.model = whisper.load_model(self.model_name)
                
                # Move to GPU if available
                if self.cuda_available:
                    logger.info("Moving model to CUDA")
                    self.model = self.model.cuda()
                
                # Set to eval mode
                self.model.eval()
                
                # Simple test to ensure model is working
                logger.info("Testing model...")
                device = next(self.model.parameters()).device
                param_count = sum(p.numel() for p in self.model.parameters())
                
                load_time = time.time() - start_time
                logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
                logger.info(f"📍 Model on device: {device}")
                logger.info(f"📊 Parameters: {param_count:,}")
                
                self.model_loaded = True
                return True
                
            except Exception as e:
                error_msg = f"Failed to load model {self.model_name}: {str(e)}"
                logger.error(error_msg)
                self.load_error = error_msg
                self.model_loaded = False
                self.model = None
                return False
    
    def transcribe(self, audio_path: str, language: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio file. Loads model if not already loaded.
        """
        # Ensure model is loaded
        if not self.model_loaded:
            logger.info("Model not loaded, loading now...")
            if not self.load_model():
                raise RuntimeError(f"Failed to load model: {self.load_error}")
        
        if self.model is None:
            raise RuntimeError("Model is None after loading")
        
        try:
            logger.info(f"Transcribing: {audio_path}")
            start_time = time.time()
            
            # Basic transcription with word timestamps
            result = self.model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                fp16=self.cuda_available,
                **kwargs
            )
            
            transcribe_time = time.time() - start_time
            logger.info(f"✅ Transcription completed in {transcribe_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the preloader"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "cuda_available": self.cuda_available,
            "load_error": self.load_error,
            "device": str(next(self.model.parameters()).device) if self.model and self.model_loaded else None
        }
    
    def unload_model(self):
        """Unload the model to free memory"""
        with self.loading_lock:
            if self.model is not None:
                logger.info("Unloading model...")
                del self.model
                self.model = None
                self.model_loaded = False
                
                # Clear GPU memory if using CUDA
                if self.cuda_available:
                    torch.cuda.empty_cache()
                
                logger.info("Model unloaded")

# Create global instance
whisper_preloader = WhisperPreloader()

# No automatic preloading - model will be loaded on first transcription request