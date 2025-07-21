#!/usr/bin/env python3
"""
Simple title generator that works without transformers pipeline
Uses basic HuggingFace models with direct API calls
"""

import logging
import re
import random
from typing import List, Optional

logger = logging.getLogger(__name__)

class SimpleTitleGenerator:
    """Simple title generator that avoids pipeline issues"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model without using pipeline"""
        try:
            logger.info("Initializing simple title generator...")
            
            # Try to load model components directly
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            self.tokenizer = AutoTokenizer.from_pretrained("TusharJoshi89/title-generator")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("TusharJoshi89/title-generator")
            
            self.model_loaded = True
            logger.info("Simple title generator loaded successfully!")
            
        except Exception as e:
            logger.warning(f"Failed to load simple model: {str(e)}")
            self.model_loaded = False
    
    def generate_title(self, text: str, max_length: int = 50, min_length: int = 10) -> Optional[str]:
        """Generate a single title from text"""
        if not self.model_loaded:
            return None
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Generate title
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1
            )
            
            # Decode output
            title = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return title.strip()
            
        except Exception as e:
            logger.error(f"Error generating title: {str(e)}")
            return None
    
    def generate_titles(self, text: str, platform: str = "youtube", count: int = 5) -> List[str]:
        """Generate multiple titles"""
        titles = []
        
        if self.model_loaded:
            # Generate multiple titles with different parameters
            for i in range(count):
                title = self.generate_title(
                    text,
                    max_length=45 + (i * 5),  # Vary length
                    min_length=10 + (i * 2)
                )
                if title and title not in titles:
                    titles.append(title)
        
        # Fill remaining with template-based titles if needed
        if len(titles) < count:
            template_titles = self._generate_template_titles(text, platform)
            for title in template_titles:
                if title not in titles and len(titles) < count:
                    titles.append(title)
        
        return titles
    
    def _generate_template_titles(self, text: str, platform: str) -> List[str]:
        """Generate template-based titles as fallback"""
        # Extract key topic from text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Find the most meaningful word
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'you', 'your', 'yours', 'we', 'our', 'ours', 'they', 'them', 'their', 'theirs', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'i', 'me', 'my', 'mine'}
        
        meaningful_words = [word for word in words if len(word) > 3 and word not in stopwords]
        
        if meaningful_words:
            # Use most common meaningful word
            word_count = {word: words.count(word) for word in meaningful_words}
            main_topic = max(word_count, key=word_count.get).title()
        else:
            main_topic = "Content"
        
        # Platform-specific templates
        if platform == "youtube":
            templates = [
                f"The Ultimate {main_topic} Guide",
                f"Everything You Need to Know About {main_topic}",
                f"How {main_topic} Will Change Your Life",
                f"The Secret to {main_topic} Success",
                f"Why {main_topic} Is Trending Right Now"
            ]
        elif platform == "instagram":
            templates = [
                f"✨ {main_topic} vibes are everything",
                f"Obsessed with this {main_topic} content 😍",
                f"Why {main_topic} is my new favorite thing",
                f"This {main_topic} post hits different 💫",
                f"Living for this {main_topic} energy 🔥"
            ]
        elif platform == "tiktok":
            templates = [
                f"This {main_topic} hack will change your life",
                f"Why everyone is talking about {main_topic}",
                f"The {main_topic} trend you need to know",
                f"This {main_topic} content is viral for a reason",
                f"How {main_topic} became everyone's obsession"
            ]
        else:  # twitter
            templates = [
                f"Thread: Everything about {main_topic} 🧵",
                f"The {main_topic} conversation we need to have",
                f"Why {main_topic} is trending right now",
                f"Hot take: {main_topic} is underrated",
                f"Breaking down the {main_topic} phenomenon"
            ]
        
        return templates

# Global instance
_simple_generator = None

def get_simple_generator():
    """Get the global simple generator instance"""
    global _simple_generator
    if _simple_generator is None:
        _simple_generator = SimpleTitleGenerator()
    return _simple_generator

def generate_simple_titles(text: str, platform: str = "youtube", count: int = 5) -> List[str]:
    """Generate titles using the simple generator"""
    generator = get_simple_generator()
    return generator.generate_titles(text, platform, count)

if __name__ == "__main__":
    # Test the simple generator
    test_text = "Today I'm going to show you how to make the perfect pasta carbonara. This is a classic Italian dish that's actually quite simple to make."
    
    print("=== Testing Simple Title Generator ===")
    titles = generate_simple_titles(test_text, "youtube")
    for i, title in enumerate(titles, 1):
        print(f"{i}. {title}")