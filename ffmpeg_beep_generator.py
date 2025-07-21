"""
FFmpeg-based beep sound generator for censoring profanity
"""

import os
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)

def generate_beep_with_ffmpeg(duration, output_path, frequency=1000, sample_rate=44100, ffmpeg_cmd="/usr/bin/ffmpeg"):
    """
    Generate a beep sound using FFmpeg
    
    Args:
        duration: Duration of the beep in seconds
        output_path: Path where to save the beep audio file
        frequency: Frequency of the beep in Hz (default: 1000)
        sample_rate: Sample rate of the audio (default: 44100)
        ffmpeg_cmd: FFmpeg command/path to use
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure duration is a positive number
        duration = max(0.1, float(duration))
        
        # FFmpeg command to generate a sine wave beep
        cmd = [
            ffmpeg_cmd,
            "-f", "lavfi",
            "-i", f"sine=frequency={frequency}:duration={duration}:sample_rate={sample_rate}",
            "-af", "volume=0.8",  # Set volume to 80% to avoid clipping
            "-y",  # Overwrite output file
            output_path
        ]
        
        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.debug(f"Beep sound generated successfully: {output_path} ({duration}s)")
            return True
        else:
            logger.error(f"FFmpeg beep generation failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg beep generation timed out for duration {duration}s")
        return False
    except Exception as e:
        logger.error(f"Error generating beep sound: {str(e)}")
        return False

def create_beep_segments(profanity_timestamps, temp_dir, ffmpeg_cmd="/usr/bin/ffmpeg"):
    """
    Create individual beep audio files for each profanity timestamp
    
    Args:
        profanity_timestamps: List of tuples (start_time, end_time, word)
        temp_dir: Temporary directory to store beep files
        ffmpeg_cmd: FFmpeg command/path to use
        
    Returns:
        list: List of tuples (beep_file_path, start_time, end_time)
    """
    beep_files = []
    
    for i, (start_time, end_time, word) in enumerate(profanity_timestamps):
        duration = end_time - start_time
        beep_filename = f"beep_{i:03d}.wav"
        beep_path = os.path.join(temp_dir, beep_filename)
        
        if generate_beep_with_ffmpeg(duration, beep_path, ffmpeg_cmd=ffmpeg_cmd):
            beep_files.append((beep_path, start_time, end_time))
            logger.info(f"Created beep for '{word}' at {start_time:.2f}s-{end_time:.2f}s")
        else:
            logger.warning(f"Failed to create beep for '{word}' at {start_time:.2f}s-{end_time:.2f}s")
    
    return beep_files

def apply_beeps_to_video(input_video_path, output_video_path, beep_files, ffmpeg_cmd="/usr/bin/ffmpeg"):
    """
    Apply beep sounds to video at specified timestamps using FFmpeg
    
    Args:
        input_video_path: Path to input video file
        output_video_path: Path to output video file
        beep_files: List of tuples (beep_file_path, start_time, end_time)
        ffmpeg_cmd: FFmpeg command/path to use
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not beep_files:
        # No beeps to apply, just copy the file
        try:
            subprocess.run([ffmpeg_cmd, "-i", input_video_path, "-c", "copy", "-y", output_video_path], 
                         check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    try:
        # Build FFmpeg command for mixing beeps with original audio
        cmd = [ffmpeg_cmd, "-i", input_video_path]
        
        # Add all beep files as inputs
        for beep_file, _, _ in beep_files:
            cmd.extend(["-i", beep_file])
        
        # Build filter complex for muting original audio and adding beeps
        filter_parts = []
        
        # Start with the original audio
        original_audio = "[0:a]"
        
        # Create a complex filter to mute the original audio at specific timestamps
        # and add beep sounds at those exact timestamps
        
        # First, split the original audio for processing
        split_audio = "asplit"
        filter_parts.append(f"{original_audio}{split_audio}=1[original]")
        
        # Process the original audio to mute at profanity timestamps
        mute_filter = "[original]"
        
        # Add volume filter with timeline editing to mute at specific timestamps
        volume_timeline = []
        for _, start_time, end_time in beep_files:
            # Format: enable='between(t,start_time,end_time)'
            volume_timeline.append(f"volume=enable='between(t,{start_time},{end_time})':volume=0")
        
        # Apply all muting in sequence
        for i, volume_expr in enumerate(volume_timeline):
            muted_label = f"muted{i}"
            filter_parts.append(f"{mute_filter}{volume_expr}[{muted_label}]")
            mute_filter = f"[{muted_label}]"
        
        # Process each beep separately
        for i, (_, start_time, end_time) in enumerate(beep_files):
            beep_input = f"[{i+1}:a]"
            delayed_beep = f"beep{i}"
            
            # Delay the beep to start at the correct time
            filter_parts.append(f"{beep_input}adelay={int(start_time * 1000)}|{int(start_time * 1000)}[{delayed_beep}]")
        
        # Mix the muted original audio with all beeps
        mix_inputs = [mute_filter]  # Start with the muted original audio
        for i in range(len(beep_files)):
            mix_inputs.append(f"[beep{i}]")
        
        # Create the final mix
        final_mix = "finalmix"
        filter_parts.append(f"{','.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest[{final_mix}]")
        
        # Combine all filter parts
        filter_complex = ";".join(filter_parts)
        
        # Fix for FFmpeg version >4: Ensure there's no trailing empty filterchain
        if filter_complex.endswith(';'):
            filter_complex = filter_complex[:-1]
            print(f"Removed trailing semicolon for FFmpeg >4 compatibility")
        
        # Add filter complex to command
        cmd.extend(["-filter_complex", filter_complex])
        
        # Map video and final mixed audio
        cmd.extend(["-map", "0:v", "-map", f"[{final_mix}]"])
        
        # Output settings
        cmd.extend(["-c:v", "copy", "-c:a", "aac", "-y", output_video_path])
        
        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"Successfully applied {len(beep_files)} beeps to video")
            return True
        else:
            logger.error(f"FFmpeg beep mixing failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg beep mixing timed out")
        return False
    except Exception as e:
        logger.error(f"Error applying beeps to video: {str(e)}")
        return False