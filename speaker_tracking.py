"""
Speaker tracking module for GPU QuickCap.
Combines pyannote.audio for speaker diarization and YOLOv8 for visual person detection.
"""

import os
import numpy as np
import torch
import logging
from pyannote.audio import Pipeline
from ultralytics import YOLO
import cv2
from typing import Dict, List, Tuple, Optional
import tempfile
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import ultralytics.nn.tasks

# Add YOLOv8 model to PyTorch's safe globals to avoid serialization issues
try:
    torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
except (AttributeError, ImportError):
    # Fallback for older PyTorch versions
    pass

# Configure logging
logger = logging.getLogger("speaker_tracking")

# Hugging Face token for pyannote.audio
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required for Hugging Face authentication")

class SpeakerTracker:
    """
    Tracks speakers in a video using audio diarization and visual detection.
    """
    
    def __init__(self, ffmpeg_cmd="/usr/bin/ffmpeg"):
        """
        Initialize the speaker tracker.
        
        Args:
            ffmpeg_cmd: Path to ffmpeg binary
        """
        self.ffmpeg_cmd = ffmpeg_cmd
        self.diarization_pipeline = None
        self.object_detector = None
        self.speakers = {}  # Speaker ID to bounding box mapping
        self.current_speaker = None
        self.speaker_history = []  # Track speaker changes for smooth transitions
        
    def load_models(self):
        """Load the required models for speaker tracking."""
        try:
            # Load pyannote.audio diarization model
            logger.info("Loading pyannote.audio diarization model...")
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", 
                    use_auth_token=HF_TOKEN
                )
            except Exception as e:
                logger.error(f"Error loading diarization model: {e}")
                return False
            
            # Load YOLOv8 model for person detection
            logger.info("Loading YOLOv8 model for person detection...")
            try:
                # Monkey patch YOLO to use weights_only=False for loading
                original_torch_load = torch.load
                
                def safe_torch_load(f, *args, **kwargs):
                    # Force weights_only=False for YOLO model loading
                    kwargs['weights_only'] = False
                    return original_torch_load(f, *args, **kwargs)
                
                # Temporarily replace torch.load with our safe version
                torch.load = safe_torch_load
                
                try:
                    self.object_detector = YOLO("yolov8n.pt")  # Use the smallest model for speed
                finally:
                    # Restore original torch.load
                    torch.load = original_torch_load
            except Exception as e:
                logger.error(f"Error loading YOLOv8 model: {e}")
                return False
            
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video for diarization.
        
        Args:
            video_path: Path to the input video
            
        Returns:
            Path to the extracted audio file
        """
        try:
            # Create a temporary file for the audio
            audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_path = audio_file.name
            audio_file.close()
            
            # Extract audio using ffmpeg
            cmd = [
                self.ffmpeg_cmd,
                "-y",  # Overwrite output files without asking
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit little-endian format
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                audio_path
            ]
            
            subprocess.run(cmd, check=True)
            logger.info(f"Audio extracted to {audio_path}")
            return audio_path
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise
    
    def perform_diarization(self, audio_path: str) -> Dict:
        """
        Perform speaker diarization on the audio.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary mapping timestamps to speaker IDs
        """
        try:
            if self.diarization_pipeline is None:
                self.load_models()
                
            # Run diarization
            diarization = self.diarization_pipeline(audio_path)
            
            # Process the results
            speaker_segments = {}
            
            # Convert diarization result to a dictionary of time segments
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time = turn.start
                end_time = turn.end
                
                # Store segments with 0.1 second resolution
                for t in np.arange(start_time, end_time, 0.1):
                    t = round(t, 1)  # Round to 1 decimal place
                    speaker_segments[t] = speaker
            
            logger.info(f"Diarization completed: {len(speaker_segments)} segments, {len(set(speaker_segments.values()))} speakers")
            return speaker_segments
        except Exception as e:
            logger.error(f"Error in diarization: {e}")
            return {}
    
    def detect_people(self, frame) -> List[Dict]:
        """
        Detect people in a video frame.
        
        Args:
            frame: Video frame (numpy array)
            
        Returns:
            List of detected people with bounding boxes
        """
        if self.object_detector is None:
            self.load_models()
            
        # Run detection
        results = self.object_detector(frame, classes=0)  # Class 0 is person
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls[0] == 0:  # Person class
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    
                    # Only include high-confidence detections
                    if confidence > 0.5:
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "center": [(x1 + x2) / 2, (y1 + y2) / 2]
                        })
        
        return detections
    
    def match_speakers_to_people(self, 
                                frame, 
                                current_time: float, 
                                speaker_segments: Dict, 
                                previous_matches: Dict = None) -> Dict:
        """
        Match detected speakers to people in the frame.
        
        Args:
            frame: Video frame
            current_time: Current timestamp in the video
            speaker_segments: Speaker diarization results
            previous_matches: Previous speaker-person matches
            
        Returns:
            Dictionary mapping speaker IDs to person bounding boxes
        """
        # Get current speaker
        current_speaker = speaker_segments.get(round(current_time, 1))
        if not current_speaker:
            return previous_matches or {}
            
        # Detect people in the frame
        people = self.detect_people(frame)
        if not people:
            return previous_matches or {}
            
        # If we have previous matches, try to maintain consistency
        if previous_matches and current_speaker in previous_matches:
            prev_bbox = previous_matches[current_speaker]
            
            # Find the closest person to the previous position
            min_dist = float('inf')
            best_match = None
            
            prev_center = [(prev_bbox[0] + prev_bbox[2]) / 2, 
                          (prev_bbox[1] + prev_bbox[3]) / 2]
            
            for person in people:
                center = person["center"]
                dist = ((center[0] - prev_center[0]) ** 2 + 
                        (center[1] - prev_center[1]) ** 2) ** 0.5
                
                if dist < min_dist:
                    min_dist = dist
                    best_match = person
            
            # If we found a close match, update the position
            if best_match and min_dist < 100:  # Threshold for considering it the same person
                previous_matches[current_speaker] = best_match["bbox"]
                return previous_matches
        
        # If no previous match or couldn't maintain consistency, assign based on position
        # Assume the person closest to the center of the frame is speaking
        frame_center = [frame.shape[1] / 2, frame.shape[0] / 2]
        
        min_dist = float('inf')
        best_match = None
        
        for person in people:
            center = person["center"]
            dist = ((center[0] - frame_center[0]) ** 2 + 
                    (center[1] - frame_center[1]) ** 2) ** 0.5
            
            if dist < min_dist:
                min_dist = dist
                best_match = person
        
        if best_match:
            # Create a new matches dictionary if needed
            matches = previous_matches or {}
            matches[current_speaker] = best_match["bbox"]
            return matches
        
        return previous_matches or {}
    
    def process_video(self, input_path: str, output_path: str, highlight_speaker: bool = True):
        """
        Process a video to track and highlight speakers.
        
        Args:
            input_path: Path to the input video
            output_path: Path to save the output video
            highlight_speaker: Whether to highlight the active speaker
            
        Returns:
            bool: True if successful, False otherwise
        """
        audio_path = None
        cap = None
        out = None
        
        try:
            # Check if input file exists
            if not os.path.exists(input_path):
                logger.error(f"Input file does not exist: {input_path}")
                return False
                
            # Make sure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Make sure models are loaded
            if not self.load_models():
                logger.error("Failed to load required models")
                return False
            
            try:
                # Extract audio for diarization
                audio_path = self.extract_audio(input_path)
                
                # Perform diarization
                speaker_segments = self.perform_diarization(audio_path)
                
                if not speaker_segments:
                    logger.warning("No speaker segments found, skipping speaker tracking")
                    # Copy the input file to the output path
                    import shutil
                    shutil.copy2(input_path, output_path)
                    return True
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")
                # Copy the input file to the output path
                import shutil
                shutil.copy2(input_path, output_path)
                return True
            
            # Open the input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {input_path}")
                return False
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output video writer with H.264 codec for better compatibility
            # First create a temporary file
            temp_output = os.path.join(os.path.dirname(output_path), f"temp_{os.path.basename(output_path)}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"Could not create output video: {output_path}")
                return False
            
            # Process the video frame by frame
            frame_count = 0
            speaker_matches = {}
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Calculate current time in the video
                current_time = frame_count / fps
                
                try:
                    # Get current speaker and match to a person in the frame
                    speaker_matches = self.match_speakers_to_people(
                        frame, current_time, speaker_segments, speaker_matches
                    )
                    
                    # Get current speaker
                    current_speaker = speaker_segments.get(round(current_time, 1))
                    
                    # We're tracking speakers but not highlighting them visually
                    # This is just for internal tracking purposes
                    
                    # If we have a current speaker and a match, we can center the frame on them
                    # This is handled by the FFmpeg crop filter in app.py
                except Exception as e:
                    logger.warning(f"Error processing frame {frame_count}: {e}")
                
                # Write the frame to the output video
                out.write(frame)
                
                # Update frame count
                frame_count += 1
                
                # Log progress
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
            
            logger.info(f"Video processing completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False
        finally:
            # Release resources
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            
            # Convert the video to a web-compatible format using FFmpeg
            if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                try:
                    logger.info(f"Converting video to web-compatible format: {output_path}")
                    cmd = [
                        self.ffmpeg_cmd,
                        "-y",  # Overwrite output files without asking
                        "-i", temp_output,
                        "-c:v", "libx264",  # H.264 codec
                        "-preset", "fast",  # Encoding speed/compression ratio
                        "-crf", "23",  # Quality level (lower is better)
                        "-pix_fmt", "yuv420p",  # Pixel format for maximum compatibility
                        "-movflags", "+faststart",  # Optimize for web streaming
                        output_path
                    ]
                    
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    # Remove the temporary file
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                except Exception as e:
                    logger.error(f"Error converting video: {e}")
                    # If conversion fails, try to use the original file
                    if os.path.exists(temp_output):
                        try:
                            import shutil
                            shutil.copy2(temp_output, output_path)
                        except Exception as copy_error:
                            logger.error(f"Error copying temporary file: {copy_error}")
            
            # Clean up temporary files
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.warning(f"Could not remove temporary audio file: {e}")
    
    def get_speaker_data(self, input_path: str) -> Dict:
        """
        Extract speaker data from a video without creating an output.
        Useful for getting diarization data to use in other processing.
        
        Args:
            input_path: Path to the input video
            
        Returns:
            Dictionary with speaker diarization data
        """
        audio_path = None
        try:
            # Check if input file exists
            if not os.path.exists(input_path):
                logger.error(f"Input file does not exist: {input_path}")
                return {}
            
            # Only load the diarization model, not the object detection model
            try:
                logger.info("Loading pyannote.audio diarization model...")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", 
                    use_auth_token=HF_TOKEN
                )
            except Exception as e:
                logger.error(f"Error loading diarization model: {e}")
                return {}
                
            # Extract audio for diarization
            try:
                audio_path = self.extract_audio(input_path)
            except Exception as e:
                logger.error(f"Error extracting audio: {e}")
                return {}
            
            # Perform diarization
            try:
                speaker_segments = self.perform_diarization(audio_path)
                return speaker_segments
            except Exception as e:
                logger.error(f"Error performing diarization: {e}")
                return {}
            
        except Exception as e:
            logger.error(f"Error getting speaker data: {e}")
            return {}
        finally:
            # Clean up temporary files
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.warning(f"Could not remove temporary audio file: {e}")