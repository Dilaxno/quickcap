#!/usr/bin/env python3
"""
Debug script to test TusharJoshi89/title-generator model loading
"""

import traceback

def test_model_loading():
    """Test different ways to load the model"""
    
    print("=== Testing TusharJoshi89/title-generator Model Loading ===\n")
    
    # Test 1: Basic pipeline loading
    print("Test 1: Basic pipeline loading")
    try:
        from transformers import pipeline
        
        print("  - Importing pipeline: SUCCESS")
        
        # Try to load the model
        summarizer = pipeline("summarization", model="TusharJoshi89/title-generator")
        print("  - Loading model: SUCCESS")
        
        # Test with simple text
        test_text = "This is a simple test text for generating a title."
        result = summarizer(test_text, max_length=50, min_length=10)
        print(f"  - Generation result: {result}")
        print("  - Test 1: PASSED\n")
        
    except Exception as e:
        print(f"  - Test 1 FAILED: {e}")
        print(f"  - Traceback: {traceback.format_exc()}")
        print()
    
    # Test 2: Try with explicit tokenizer and model
    print("Test 2: Explicit tokenizer and model loading")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        
        print("  - Importing classes: SUCCESS")
        
        tokenizer = AutoTokenizer.from_pretrained("TusharJoshi89/title-generator")
        model = AutoModelForSeq2SeqLM.from_pretrained("TusharJoshi89/title-generator")
        
        print("  - Loading tokenizer and model: SUCCESS")
        
        # Create pipeline with explicit model and tokenizer
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
        
        print("  - Creating pipeline: SUCCESS")
        
        # Test with simple text
        test_text = "This is a simple test text for generating a title."
        result = summarizer(test_text, max_length=50, min_length=10)
        print(f"  - Generation result: {result}")
        print("  - Test 2: PASSED\n")
        
    except Exception as e:
        print(f"  - Test 2 FAILED: {e}")
        print(f"  - Traceback: {traceback.format_exc()}")
        print()
    
    # Test 3: Try with different parameters
    print("Test 3: Different parameter approach")
    try:
        from transformers import pipeline
        
        # Try with different parameters
        summarizer = pipeline(
            "summarization", 
            model="TusharJoshi89/title-generator",
            device=-1,  # Force CPU
            framework="pt"  # Force PyTorch
        )
        
        print("  - Loading model with specific parameters: SUCCESS")
        
        # Test with simple text
        test_text = "This is a simple test text for generating a title."
        result = summarizer(test_text, max_length=50, min_length=10, do_sample=False)
        print(f"  - Generation result: {result}")
        print("  - Test 3: PASSED\n")
        
    except Exception as e:
        print(f"  - Test 3 FAILED: {e}")
        print(f"  - Traceback: {traceback.format_exc()}")
        print()

if __name__ == "__main__":
    test_model_loading()