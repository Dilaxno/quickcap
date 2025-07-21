#!/usr/bin/env python3
"""
Script to fix Phi-3 Mini model loading issues
This script ensures the model is properly downloaded and cached
"""

import os
import logging
import sys
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def fix_phi3_model():
    """Fix Phi-3 Mini model loading issues"""
    try:
        logger.info("Starting Phi-3 Mini model fix...")
        
        # Check if transformers is installed
        try:
            import transformers
            logger.info(f"Transformers version: {transformers.__version__}")
        except ImportError:
            logger.error("Transformers not installed. Installing required packages...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "huggingface_hub"])
            logger.info("Packages installed successfully")
        
        # Check if torch is installed
        try:
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except ImportError:
            logger.error("PyTorch not installed. Please install with: pip install torch")
            return False
        
        # Set environment variables to ensure model downloads properly
        os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Ensure online mode
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Disable symlink warnings
        
        # Model ID for Phi-3 Mini
        model_id = "microsoft/phi-3-mini-128k-instruct"
        
        # Get the cache directory
        from huggingface_hub import constants
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        model_cache_dir = os.path.join(cache_dir, "hub", model_id.replace("/", "--"))
        
        logger.info(f"Cache directory: {cache_dir}")
        logger.info(f"Model cache directory: {model_cache_dir}")
        
        # Clear existing cache for this model if it exists
        if os.path.exists(model_cache_dir):
            logger.info(f"Removing existing model cache: {model_cache_dir}")
            try:
                shutil.rmtree(model_cache_dir)
                logger.info("Existing cache removed successfully")
            except Exception as e:
                logger.warning(f"Could not remove cache directory: {e}")
        
        # Force download the model
        logger.info("Downloading Phi-3 Mini model...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # First download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            use_fast=True,
            token=None  # No token needed for public models
        )
        
        # Then download model
        logger.info("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=None  # No token needed for public models
        )
        
        # Test the model
        logger.info("Testing model...")
        test_prompt = "<|user|>\nGenerate a short title for a video about cooking.\n<|assistant|>"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7
            )
        
        test_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Model test response: {test_response}")
        
        # Create a marker file to indicate successful installation
        success_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".phi3_model_ready")
        with open(success_marker, "w") as f:
            f.write(f"Model installed successfully at {model_cache_dir}")
        
        logger.info("✅ Phi-3 Mini model fix completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing Phi-3 Mini model: {str(e)}")
        return False

if __name__ == "__main__":
    success = fix_phi3_model()
    if success:
        print("\n✅ Phi-3 Mini model is now ready to use!")
    else:
        print("\n❌ Failed to fix Phi-3 Mini model. Please check the logs.")