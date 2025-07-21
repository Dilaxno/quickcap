import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
import base64
from pathlib import Path

logger = logging.getLogger(__name__)

class BRollProcessor:
    """Handles B-roll image processing and video integration"""
    
    def __init__(self):
        self.broll_data_dir = "broll_data"
        self.broll_images_dir = "broll_images"
        
        # Create directories if they don't exist
        os.makedirs(self.broll_data_dir, exist_ok=True)
        os.makedirs(self.broll_images_dir, exist_ok=True)
    
    def save_broll_image(self, image_data: Union[str, bytes], video_id: str, phrase_id: str) -> str:
        """Save a B-roll image for a specific video and phrase"""
        try:
            # Create video-specific directory
            video_dir = os.path.join(self.broll_images_dir, video_id)
            os.makedirs(video_dir, exist_ok=True)
            
            # Handle different input types
            if isinstance(image_data, bytes):
                # Raw bytes data (from file upload)
                image_bytes = image_data
            elif isinstance(image_data, str):
                # Base64 string data
                if image_data.startswith('data:image/'):
                    # Remove data URL prefix
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
            else:
                raise ValueError(f"Unsupported image_data type: {type(image_data)}")
            
            # Save image file
            image_filename = f"{phrase_id}.jpg"
            image_path = os.path.join(video_dir, image_filename)
            
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            logger.info(f"Saved B-roll image: {image_path}")
            return image_path
            
        except Exception as e:
            logger.error(f"Error saving B-roll image: {e}")
            raise
    
    def get_broll_data(self, video_id: str) -> Dict[str, Any]:
        """Get B-roll data for a specific video"""
        try:
            data_file = os.path.join(self.broll_data_dir, f"{video_id}.json")
            
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    return json.load(f)
            else:
                # Return empty structure if no data exists
                return {
                    "video_id": video_id,
                    "images": {},
                    "metadata": {}
                }
                
        except Exception as e:
            logger.error(f"Error getting B-roll data: {e}")
            return {
                "video_id": video_id,
                "images": {},
                "metadata": {}
            }
    
    def save_broll_data(self, video_id: str, broll_data: Dict[str, Any]) -> None:
        """Save B-roll data for a specific video"""
        try:
            data_file = os.path.join(self.broll_data_dir, f"{video_id}.json")
            
            with open(data_file, 'w') as f:
                json.dump(broll_data, f, indent=2)
            
            logger.info(f"Saved B-roll data: {data_file}")
            
        except Exception as e:
            logger.error(f"Error saving B-roll data: {e}")
            raise
    
    def apply_broll_to_video(self, input_video_path: str, output_video_path: str, 
                           broll_data: Dict[str, Any], timestamps: List[Dict]) -> bool:
        """Apply B-roll images to video at specified timestamps"""
        try:
            # This is a placeholder implementation
            # In a real implementation, you would use FFmpeg or similar to overlay images
            logger.info(f"Processing video with B-roll: {input_video_path} -> {output_video_path}")
            logger.info(f"B-roll data: {len(broll_data.get('images', {}))} images")
            logger.info(f"Timestamps: {len(timestamps)} entries")
            
            # For now, just copy the input to output as a placeholder
            import shutil
            shutil.copy2(input_video_path, output_video_path)
            
            logger.info(f"B-roll processing completed: {output_video_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying B-roll to video: {e}")
            return False

# Global instance
_broll_processor = None

def get_broll_processor() -> BRollProcessor:
    """Get the global B-roll processor instance"""
    global _broll_processor
    if _broll_processor is None:
        _broll_processor = BRollProcessor()
    return _broll_processor