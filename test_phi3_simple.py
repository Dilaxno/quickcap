#!/usr/bin/env python3
"""
Simple test script for Phi-3 Mini title generation
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_phi3_title_generation():
    """Test Phi-3 Mini title generation"""
    try:
        from phi3_title_generator import generate_titles
        
        # Test transcriptions
        test_transcriptions = [
            "Today I'm going to show you how to make the perfect pasta carbonara. This is a classic Italian dish that's actually quite simple to make.",
            "In this video, we're exploring the beautiful landscapes of New Zealand. The mountains and lakes here are absolutely breathtaking.",
            "Let's take a look at the latest smartphone from Apple. The camera quality is impressive and the battery life has been significantly improved."
        ]
        
        # Test each transcription
        for i, transcription in enumerate(test_transcriptions, 1):
            print(f"\n=== Test {i}: {transcription[:50]}... ===")
            
            # Generate titles
            titles = generate_titles(transcription, "youtube")
            
            # Print results
            print(f"\nGenerated {len(titles)} titles:")
            for j, title in enumerate(titles, 1):
                print(f"{j}. {title}")
            
            print("\n" + "-" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Phi-3 title generation: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n=== Phi-3 Mini Title Generation Test ===\n")
    success = test_phi3_title_generation()
    
    if success:
        print("\n✅ Phi-3 Mini title generation test completed successfully!")
    else:
        print("\n❌ Phi-3 Mini title generation test failed. Please run setup_phi3_model.bat first.")