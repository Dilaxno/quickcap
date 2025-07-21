#!/usr/bin/env python3
"""
Test script for Hugging Face GPT-2 title generation
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from title_generator import TitleGenerator

def test_huggingface_title_generation():
    """Test the Hugging Face GPT-2 title generation functionality"""
    
    # Sample transcription for testing
    test_transcription = """
    Today I'm going to show you how to make the perfect pasta carbonara. 
    This is a classic Italian dish that's surprisingly simple but requires 
    the right technique. We'll start with guanciale, which is the traditional 
    pork used in carbonara, and I'll show you how to get that perfect creamy 
    texture without scrambling the eggs.
    """
    
    print("Testing Hugging Face GPT-2 Title Generator...")
    print("=" * 50)
    
    # Initialize the title generator
    generator = TitleGenerator()
    
    # Check if Hugging Face is available
    if not generator.hf_available:
        print("❌ Hugging Face GPT-2 is not available. Please check your HF_TOKEN in .env file.")
        print("Current HF_TOKEN:", os.getenv('HF_TOKEN', 'Not set'))
        print("Note: Falling back to template generation.")
        return False
    
    print("✅ Hugging Face GPT-2 model initialized successfully!")
    
    # Test title generation for different platforms
    platforms = ['youtube', 'instagram', 'tiktok', 'twitter']
    
    for platform in platforms:
        print(f"\n🎯 Generating titles for {platform.upper()}:")
        print("-" * 30)
        
        try:
            titles = generator.generate_titles(test_transcription, platform)
            
            if titles:
                for i, title in enumerate(titles, 1):
                    print(f"{i}. {title}")
            else:
                print("❌ No titles generated")
                
        except Exception as e:
            print(f"❌ Error generating titles for {platform}: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    return True

if __name__ == "__main__":
    test_huggingface_title_generation()