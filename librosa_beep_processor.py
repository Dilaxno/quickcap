"""
Librosa-based beep sound processor for censoring profanity with precise timing
"""

import os
import numpy as np
import librosa
import soundfile as sf
import tempfile
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class LibrosaBeepProcessor:
    def __init__(self, sample_rate: int = 44100, beep_frequency: float = 1000.0):
        """
        Initialize the librosa-based beep processor
        
        Args:
            sample_rate: Audio sample rate (default: 44100)
            beep_frequency: Frequency of the beep sound in Hz (default: 1000.0)
        """
        self.sample_rate = sample_rate
        self.beep_frequency = beep_frequency
        logger.info(f"Initialized LibrosaBeepProcessor with sample_rate={sample_rate}, beep_frequency={beep_frequency}")
    
    def generate_beep_tone(self, duration: float, amplitude: float = 0.5) -> np.ndarray:
        """
        Generate a beep tone using librosa
        
        Args:
            duration: Duration of the beep in seconds
            amplitude: Amplitude of the beep (0.0 to 1.0)
            
        Returns:
            np.ndarray: Generated beep audio data
        """
        # Generate time array
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Generate sine wave
        beep = amplitude * np.sin(2 * np.pi * self.beep_frequency * t)
        
        # Apply fade in/out to avoid clicks
        fade_samples = int(0.01 * self.sample_rate)  # 10ms fade
        if len(beep) > 2 * fade_samples:
            # Fade in
            beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
            # Fade out
            beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return beep.astype(np.float32)
    
    def load_audio_from_video(self, video_path: str) -> Tuple[np.ndarray, int]:
        """
        Extract audio from video file using librosa
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio from video file
            audio_data, sr = librosa.load(video_path, sr=self.sample_rate, mono=True)
            logger.info(f"Loaded audio from video: {len(audio_data)} samples at {sr} Hz")
            return audio_data, sr
        except Exception as e:
            logger.error(f"Failed to load audio from video {video_path}: {str(e)}")
            raise
    
    def apply_precise_muting_and_beeps(self, 
                                     audio_data: np.ndarray, 
                                     profanity_timestamps: List[Tuple[float, float, str]],
                                     beep_amplitude: float = 0.6,
                                     crossfade_duration: float = 0.05) -> np.ndarray:
        """
        Apply precise muting and beep sounds to audio data
        
        Args:
            audio_data: Original audio data
            profanity_timestamps: List of (start_time, end_time, word) tuples
            beep_amplitude: Amplitude of the beep sound (0.0 to 1.0)
            crossfade_duration: Duration of crossfade in seconds for smooth transitions
            
        Returns:
            np.ndarray: Processed audio data with muted profanity and beep sounds
        """
        if not profanity_timestamps:
            return audio_data
        
        # Create a copy of the original audio
        processed_audio = audio_data.copy()
        
        # Convert crossfade duration to samples
        crossfade_samples = int(crossfade_duration * self.sample_rate)
        
        logger.info(f"Processing {len(profanity_timestamps)} profanity instances")
        
        for start_time, end_time, word in profanity_timestamps:
            # Convert timestamps to sample indices
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Ensure indices are within bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(processed_audio), end_sample)
            
            if start_sample >= end_sample:
                logger.warning(f"Invalid timestamp range for word '{word}': {start_time}-{end_time}")
                continue
            
            # Calculate duration and generate beep
            duration = (end_sample - start_sample) / self.sample_rate
            beep_tone = self.generate_beep_tone(duration, beep_amplitude)
            
            # Ensure beep length matches the segment length
            if len(beep_tone) != (end_sample - start_sample):
                # Resize beep to match exact segment length
                beep_tone = np.resize(beep_tone, end_sample - start_sample)
            
            # Apply crossfade for smooth transitions
            if crossfade_samples > 0:
                # Crossfade at the beginning
                fade_start = max(0, start_sample - crossfade_samples)
                fade_end = min(len(processed_audio), start_sample + crossfade_samples)
                
                if fade_start < start_sample:
                    # Create fade-out for original audio before the beep
                    fade_out_length = start_sample - fade_start
                    fade_out_curve = np.linspace(1, 0, fade_out_length)
                    processed_audio[fade_start:start_sample] *= fade_out_curve
                
                if fade_end > start_sample and fade_end <= end_sample:
                    # Create fade-in for beep at the beginning
                    fade_in_length = fade_end - start_sample
                    fade_in_curve = np.linspace(0, 1, fade_in_length)
                    beep_tone[:fade_in_length] *= fade_in_curve
                
                # Crossfade at the end
                fade_start = max(start_sample, end_sample - crossfade_samples)
                fade_end = min(len(processed_audio), end_sample + crossfade_samples)
                
                if fade_start < end_sample:
                    # Create fade-out for beep at the end
                    fade_out_length = end_sample - fade_start
                    fade_out_curve = np.linspace(1, 0, fade_out_length)
                    beep_tone[-fade_out_length:] *= fade_out_curve
                
                if fade_end > end_sample:
                    # Create fade-in for original audio after the beep
                    fade_in_length = fade_end - end_sample
                    fade_in_curve = np.linspace(0, 1, fade_in_length)
                    processed_audio[end_sample:fade_end] *= fade_in_curve
            
            # Replace the audio segment with the beep
            processed_audio[start_sample:end_sample] = beep_tone
            
            logger.info(f"Applied beep for '{word}' at {start_time:.3f}s-{end_time:.3f}s "
                       f"(samples {start_sample}-{end_sample})")
        
        return processed_audio
    
    def save_audio_to_temp_file(self, audio_data: np.ndarray, temp_dir: str) -> str:
        """
        Save processed audio data to a temporary file
        
        Args:
            audio_data: Audio data to save
            temp_dir: Temporary directory path
            
        Returns:
            str: Path to the saved temporary audio file
        """
        temp_audio_path = os.path.join(temp_dir, "processed_audio.wav")
        
        try:
            # Save audio using soundfile
            sf.write(temp_audio_path, audio_data, self.sample_rate)
            logger.info(f"Saved processed audio to {temp_audio_path}")
            return temp_audio_path
        except Exception as e:
            logger.error(f"Failed to save processed audio: {str(e)}")
            raise
    
    def process_video_with_beeps(self, 
                                input_video_path: str, 
                                output_video_path: str,
                                profanity_timestamps: List[Tuple[float, float, str]],
                                ffmpeg_cmd: str = "/usr/bin/ffmpeg",
                                beep_amplitude: float = 0.6,
                                crossfade_duration: float = 0.05) -> bool:
        """
        Process video with precise profanity beeping using librosa
        
        Args:
            input_video_path: Path to input video file
            output_video_path: Path to output video file
            profanity_timestamps: List of (start_time, end_time, word) tuples
            ffmpeg_cmd: FFmpeg command for video processing
            beep_amplitude: Amplitude of beep sounds
            crossfade_duration: Duration of crossfade for smooth transitions
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not profanity_timestamps:
            # No profanity to process, just copy the file
            try:
                import subprocess
                subprocess.run([ffmpeg_cmd, "-i", input_video_path, "-c", "copy", "-y", output_video_path], 
                             check=True, capture_output=True)
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to copy video file: {str(e)}")
                return False
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Load audio from video
                logger.info("Loading audio from video...")
                audio_data, sr = self.load_audio_from_video(input_video_path)
                
                # Apply precise muting and beep sounds
                logger.info("Applying precise muting and beep sounds...")
                processed_audio = self.apply_precise_muting_and_beeps(
                    audio_data, profanity_timestamps, beep_amplitude, crossfade_duration
                )
                
                # Save processed audio to temporary file
                temp_audio_path = self.save_audio_to_temp_file(processed_audio, temp_dir)
                
                # Use FFmpeg to combine processed audio with original video
                logger.info("Combining processed audio with video...")
                import subprocess
                
                cmd = [
                    ffmpeg_cmd,
                    "-i", input_video_path,  # Input video
                    "-i", temp_audio_path,   # Processed audio
                    "-c:v", "copy",          # Copy video stream
                    "-c:a", "aac",           # Encode audio as AAC
                    "-map", "0:v:0",         # Map video from first input
                    "-map", "1:a:0",         # Map audio from second input
                    "-y",                    # Overwrite output
                    output_video_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    logger.info(f"Successfully processed video with {len(profanity_timestamps)} beeps")
                    return True
                else:
                    logger.error(f"FFmpeg video processing failed: {result.stderr}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error processing video with beeps: {str(e)}")
            return False
    
    def analyze_audio_precision(self, 
                               audio_data: np.ndarray, 
                               profanity_timestamps: List[Tuple[float, float, str]]) -> dict:
        """
        Analyze audio data to provide precision metrics for profanity detection
        
        Args:
            audio_data: Audio data to analyze
            profanity_timestamps: List of profanity timestamps
            
        Returns:
            dict: Analysis results with precision metrics
        """
        analysis = {
            "total_duration": len(audio_data) / self.sample_rate,
            "sample_rate": self.sample_rate,
            "total_profanity_instances": len(profanity_timestamps),
            "profanity_details": []
        }
        
        for start_time, end_time, word in profanity_timestamps:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Ensure indices are within bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            
            if start_sample < end_sample:
                segment = audio_data[start_sample:end_sample]
                
                # Calculate RMS energy of the segment
                rms_energy = np.sqrt(np.mean(segment**2))
                
                # Calculate zero crossing rate
                zero_crossings = np.sum(np.diff(np.signbit(segment)))
                zcr = zero_crossings / len(segment)
                
                analysis["profanity_details"].append({
                    "word": word,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "sample_precision": {
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "sample_count": end_sample - start_sample
                    },
                    "audio_features": {
                        "rms_energy": float(rms_energy),
                        "zero_crossing_rate": float(zcr)
                    }
                })
        
        return analysis