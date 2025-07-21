"""
Beep sound generator for censoring profanity
"""

import numpy as np
import os
import tempfile
from moviepy.audio.AudioClip import AudioArrayClip

def generate_beep_sound(duration=0.5, frequency=1000, sample_rate=44100):
    """
    Generate a beep sound with the specified duration and frequency
    
    Args:
        duration: Duration of the beep in seconds
        frequency: Frequency of the beep in Hz
        sample_rate: Sample rate of the audio
        
    Returns:
        AudioArrayClip: A MoviePy AudioClip containing the beep sound
    """
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate sine wave
    beep = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Apply fade in/out to avoid clicks
    fade_duration = min(0.05, duration / 4)  # 50ms fade or 1/4 of duration, whichever is smaller
    fade_samples = int(fade_duration * sample_rate)
    
    if fade_samples > 0:
        # Apply fade in
        beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
        # Apply fade out
        beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # Convert to stereo
    beep_stereo = np.column_stack((beep, beep))
    
    # Create AudioArrayClip
    return AudioArrayClip(beep_stereo, fps=sample_rate)

def get_beep_for_duration(duration, volume_factor=1.5):
    """
    Get a beep sound clip with the exact duration needed

    Args:
        duration: Duration of the beep in seconds
        volume_factor: Volume multiplier for the beep (default: 1.5 for 50% louder)

    Returns:
        AudioArrayClip: A MoviePy AudioClip containing the beep sound
    """
    try:
        # Ensure duration is a positive number
        duration = max(0.1, float(duration))
        
        # Generate the beep sound
        beep_sound = generate_beep_sound(duration=duration)
        
        # Increase the volume to make it more noticeable
        if volume_factor != 1.0:
            beep_sound = beep_sound.volumex(volume_factor)
            
        return beep_sound
    except Exception as e:
        import traceback
        print(f"Error generating beep sound: {str(e)}")
        traceback.print_exc()
        # Return a default beep if there's an error
        return generate_beep_sound(duration=0.5)