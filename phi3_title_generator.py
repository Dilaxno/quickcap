#!/usr/bin/env python3
"""
Title generation module using Microsoft's Phi-3 Mini 1.8B model
This module provides a more efficient and creative approach to title generation
"""

import logging
import re
import random
import os
import time
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phi3TitleGenerator:
    """
    A title generator using Microsoft's Phi-3 Mini 1.8B model
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Phi-3 Mini model with robust error handling"""
        logger.info("Initializing Microsoft Phi-3 Mini title generator...")
        
        # Check if model was previously successfully loaded
        success_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".phi3_model_ready")
        if os.path.exists(success_marker):
            logger.info("Found marker file indicating model was previously loaded successfully")
        
        try:
            # Try to import and initialize the model
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Model ID for Phi-3 Mini
            model_id = "microsoft/phi-3-mini-128k-instruct"
            
            # Set environment variables to ensure model downloads properly
            os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Ensure online mode
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Disable symlink warnings
            
            # Download tokenizer first
            logger.info("Loading tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id, 
                    trust_remote_code=True
                )
                logger.info("Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                raise
            
            # Then download model
            logger.info("Loading model (this may take a few minutes)...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
            
            # Test the model with a simple request to ensure it's working
            logger.info("Testing Phi-3 Mini model...")
            test_prompt = "<|user|>\nGenerate a short title for a video about cooking.\n<|assistant|>"
            
            try:
                inputs = self.tokenizer(test_prompt, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=0.7
                    )
                
                test_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Model test response: {test_response}")
                
                # Create a marker file to indicate successful initialization
                with open(success_marker, "w") as f:
                    f.write(f"Model initialized successfully at {time.ctime()}")
                
                self.model_loaded = True
                logger.info("✅ Phi-3 Mini model initialized successfully!")
                
            except Exception as e:
                logger.error(f"Model test failed: {e}")
                raise
                
        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                logger.warning("GPU memory insufficient for model. Falling back to template generation.")
            else:
                logger.warning(f"Failed to initialize Phi-3 Mini model: {error_msg}")
            self.model_loaded = False
        
        logger.info(f"Phi-3 Mini title generator initialization status: {self.model_loaded}")
    
    def generate_titles(self, transcription: str, platform: str = "youtube") -> List[str]:
        """
        Generate viral titles for the given transcription and platform
        
        Args:
            transcription: The video transcript text
            platform: Target platform (youtube, instagram, tiktok, twitter)
            
        Returns:
            List of generated titles
        """
        try:
            titles = []
            
            # Extract key content for shorter input
            summary = self._extract_key_content(transcription)
            
            # Method 1: Try Phi-3 Mini if available
            if self.model_loaded:
                logger.info("Generating titles with Phi-3 Mini model...")
                phi3_titles = self._generate_with_phi3(summary, platform)
                titles.extend(phi3_titles)
                logger.info(f"Generated {len(phi3_titles)} titles with Phi-3 Mini")
            else:
                logger.info("Phi-3 Mini model not loaded, skipping model-based generation")
            
            # Method 2: Fallback to template generation if needed
            if len(titles) < 3:
                logger.info("Using template-based title generation as fallback")
                template_titles = self._generate_template_titles(transcription, platform)
                titles.extend(template_titles)
            
            # Clean and deduplicate
            unique_titles = []
            seen = set()
            for title in titles:
                cleaned = self._clean_title(title, platform)
                if cleaned and len(cleaned) > 10 and cleaned not in seen:
                    unique_titles.append(cleaned)
                    seen.add(cleaned)
            
            logger.info(f"Returning {len(unique_titles)} unique titles")
            return unique_titles[:5]  # Return top 5 titles
            
        except Exception as e:
            logger.error(f"Error generating titles: {str(e)}")
            return self._generate_template_titles(transcription, platform)
    
    def _generate_with_phi3(self, transcription: str, platform: str) -> List[str]:
        """Generate titles using Phi-3 Mini model"""
        titles = []
        
        if not self.model_loaded:
            logger.warning("Model not loaded, cannot generate titles with Phi-3")
            return titles
        
        try:
            import torch
            
            # Viral content strategist prompt
            prompt_template = """<|user|>
You are a viral content strategist for social media platforms like YouTube Shorts, TikTok, and Instagram Reels.
Your task is to generate 5 creative, attention-grabbing, and emotionally compelling video titles from the transcription below. The titles must be short (under 60 characters), optimized for virality and curiosity, and written in a tone that matches the content's mood (funny, shocking, emotional, motivational, etc).

The titles should:
- Be written like human viral headlines
- Spark curiosity or emotion in the first 5 words
- Use powerful, punchy language
- Avoid generic or boring phrasing
- Avoid repeating words from the transcript verbatim unless they are dramatic or shocking
- NOT mention "transcript", "summary", or metadata

Output the titles as a numbered list. Include a variety of styles: one clickbait, one emotional, one mysterious, and one that could go viral based on a trending tone.

Here is the transcription:
"{}"
<|assistant|>"""
            
            # Truncate transcription if too long (Phi-3 Mini has context window limitations)
            max_chars = 1500
            if len(transcription) > max_chars:
                transcription = transcription[:max_chars] + "..."
            
            # Format the prompt with the transcription
            prompt = prompt_template.format(transcription)
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate titles
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=250,  # Enough for 5 titles
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    repetition_penalty=1.2
                )
            
            # Decode the output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the assistant's response (after the prompt)
            response = generated_text.split("<|assistant|>")[-1].strip()
            
            # Extract numbered titles from the response
            title_pattern = r"\d+\.\s*(.*?)(?=\d+\.|$)"
            extracted_titles = re.findall(title_pattern, response, re.DOTALL)
            
            # Clean up the extracted titles
            for title in extracted_titles:
                clean_title = title.strip()
                if clean_title and len(clean_title) > 5:
                    titles.append(clean_title)
            
            logger.info(f"Generated {len(titles)} titles with Phi-3 Mini")
            
        except Exception as e:
            logger.warning(f"Phi-3 Mini title generation failed: {str(e)}")
        
        return titles
    
    def _extract_key_content(self, transcription: str) -> str:
        """Extract key content from transcription"""
        # Clean up the transcription
        cleaned = re.sub(r'\s+', ' ', transcription.strip())
        
        # Take first 1000 characters for Phi-3 Mini context
        summary = cleaned[:1000]
        if not summary.endswith(('.', '!', '?')):
            # Find the last complete sentence
            last_period = max(summary.rfind('.'), summary.rfind('!'), summary.rfind('?'))
            if last_period > 50:
                summary = summary[:last_period + 1]
            else:
                summary += '.'
        
        return summary
    
    def _generate_template_titles(self, transcription: str, platform: str) -> List[str]:
        """Generate template-based titles as fallback"""
        # Extract key topic from text
        content_analysis = self._analyze_content(transcription)
        main_topic = content_analysis['main_topic']
        
        # Platform-specific templates
        if platform == "youtube":
            templates = [
                f"The Ultimate {main_topic} Guide You Need to See",
                f"I Tried {main_topic} for 24 Hours - What Happened Next",
                f"The {main_topic} Secret Nobody Talks About",
                f"How {main_topic} Changed Everything Overnight",
                f"Why {main_topic} Is Breaking The Internet Right Now"
            ]
        elif platform == "instagram":
            templates = [
                f"✨ {main_topic} vibes that hit different",
                f"This {main_topic} moment lives rent free in my head 😍",
                f"POV: You discover the {main_topic} hack everyone needs",
                f"The {main_topic} energy we all need right now 💫",
                f"Main character energy: {main_topic} edition 🔥"
            ]
        elif platform == "tiktok":
            templates = [
                f"This {main_topic} hack changed my life overnight",
                f"POV: When {main_topic} hits just right #fyp",
                f"No one's talking about this {main_topic} secret",
                f"Wait for it... {main_topic} plot twist 🤯",
                f"Tell me you love {main_topic} without telling me"
            ]
        else:  # twitter
            templates = [
                f"I can't believe this {main_topic} moment actually happened",
                f"The {main_topic} discourse we all needed right now",
                f"Hot take: {main_topic} is actually underrated",
                f"This {main_topic} revelation changed everything",
                f"Unpopular opinion: {main_topic} deserves more attention"
            ]
        
        return templates
    
    def _analyze_content(self, transcription: str) -> Dict[str, Any]:
        """Analyze content to extract meaningful patterns"""
        words = re.findall(r'\b\w+\b', transcription.lower())
        
        # Enhanced stopwords list
        stopwords = {
            'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 
            'were', 'said', 'each', 'which', 'their', 'time', 'about', 'would',
            'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just',
            'first', 'also', 'after', 'back', 'only', 'come', 'work', 'life',
            'where', 'much', 'before', 'move', 'right', 'think', 'even', 'through',
            'these', 'good', 'most', 'well', 'way', 'down', 'should', 'because',
            'each', 'those', 'people', 'take', 'year', 'your', 'some', 'them',
            'see', 'him', 'long', 'make', 'thing', 'look', 'two', 'how', 'its',
            'our', 'out', 'day', 'get', 'use', 'man', 'new', 'now', 'way',
            'may', 'say', 'each', 'which', 'she', 'do', 'his', 'but', 'from',
            'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my',
            'one', 'all', 'would', 'there', 'their', 'going', 'show', 'here',
            'really', 'today', 'welcome', 'video', 'going', 'gonna', 'want',
            'like', 'need', 'into', 'over', 'think', 'than', 'find', 'many',
            'then', 'them', 'these', 'come', 'made', 'both', 'little', 'being'
        }
        
        # Find meaningful words (nouns, verbs, adjectives that describe the content)
        meaningful_words = []
        for word in words:
            if (len(word) > 3 and 
                word not in stopwords and 
                not word.isdigit() and 
                word.isalpha()):
                meaningful_words.append(word)
        
        # Look for compound topics and key phrases
        text_lower = transcription.lower()
        
        # Common topic patterns
        topic_patterns = [
            r'\b(pasta|carbonara|recipe|cooking|food|kitchen|chef|ingredients)\b',
            r'\b(travel|hiking|trails|zealand|landscape|mountain|adventure|explore)\b',
            r'\b(tech|smartphone|phone|camera|review|technology|gadget|device)\b',
            r'\b(productivity|workspace|workflow|setup|office|work|efficiency)\b',
            r'\b(fitness|workout|exercise|gym|training|health|body)\b',
            r'\b(gaming|game|play|console|pc|xbox|playstation)\b',
            r'\b(tutorial|guide|learn|teach|education|instruction)\b',
            r'\b(business|money|finance|investment|entrepreneur)\b',
            r'\b(art|design|creative|painting|drawing|music)\b',
            r'\b(science|research|study|experiment|discovery)\b'
        ]
        
        detected_topics = []
        for pattern in topic_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                detected_topics.extend(matches)
        
        # If we found topic-specific words, use the most common one
        if detected_topics:
            topic_freq = {}
            for topic in detected_topics:
                topic_freq[topic] = topic_freq.get(topic, 0) + 1
            main_topic = max(topic_freq.items(), key=lambda x: x[1])[0].title()
        elif meaningful_words:
            # Fallback to general word frequency analysis
            word_freq = {}
            for word in meaningful_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get the most frequent meaningful word
            if word_freq:
                main_topic = max(word_freq.items(), key=lambda x: x[1])[0].title()
            else:
                main_topic = "This Amazing Content"
        else:
            main_topic = "This Amazing Content"
        
        return {
            'main_topic': main_topic
        }
    
    def _clean_title(self, title: str, platform: str) -> str:
        """Clean and format the generated title"""
        # Remove common artifacts
        title = title.split('\n')[0].strip()
        title = title.strip('"\'')
        
        # Remove incomplete sentences
        if title.endswith(('and', 'or', 'but', 'the', 'a', 'an')):
            words = title.split()
            title = ' '.join(words[:-1])
        
        # Ensure proper capitalization
        if title and len(title) > 1:
            title = title[0].upper() + title[1:]
        
        # Ensure title isn't too long
        if len(title) > 60:
            title = title[:57] + "..."
        
        return title

# Global instance
_phi3_title_generator = None

def get_phi3_title_generator():
    """Get the global Phi-3 title generator instance"""
    global _phi3_title_generator
    if _phi3_title_generator is None:
        _phi3_title_generator = Phi3TitleGenerator()
    return _phi3_title_generator

def generate_titles(transcription: str, platform: str = "youtube") -> List[str]:
    """
    Generate viral titles for the given transcription and platform using Phi-3 Mini
    
    Args:
        transcription: The video transcript text
        platform: Target platform (youtube, instagram, tiktok, twitter)
        
    Returns:
        List of generated titles
    """
    try:
        # Try the Phi-3 generator
        generator = get_phi3_title_generator()
        titles = generator.generate_titles(transcription, platform)
        
        # If we got good results, return them
        if titles and len(titles) >= 3:
            return titles
        
        # If Phi-3 generator failed or produced few titles, try simple generator
        logger.info("Trying simple generator as fallback...")
        from simple_title_generator import generate_simple_titles
        simple_titles = generate_simple_titles(transcription, platform)
        
        # Combine results
        all_titles = titles + simple_titles
        
        # Remove duplicates while preserving order
        unique_titles = []
        seen = set()
        for title in all_titles:
            if title not in seen:
                unique_titles.append(title)
                seen.add(title)
        
        return unique_titles[:5]  # Return top 5
        
    except Exception as e:
        logger.error(f"Error in generate_titles: {str(e)}")
        # Final fallback - return basic templates
        return _generate_basic_fallback(transcription, platform)

def _generate_basic_fallback(transcription: str, platform: str) -> List[str]:
    """Generate basic fallback titles when all else fails"""
    # Extract a key word from transcription
    words = re.findall(r'\b\w{4,}\b', transcription.lower())
    stopwords = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'about', 'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'also', 'after', 'back', 'only', 'come', 'work', 'life', 'where', 'much', 'before', 'move', 'right', 'think', 'even', 'through', 'these', 'good', 'most', 'well', 'down', 'should', 'because', 'those', 'people', 'take', 'year', 'your', 'some', 'them', 'look', 'make', 'thing', 'going', 'gonna', 'want', 'like', 'need', 'into', 'over', 'than', 'find', 'many', 'then', 'come', 'made', 'both', 'little', 'being'}
    
    meaningful_words = [word for word in words if word not in stopwords]
    
    if meaningful_words:
        # Use most common word
        word_count = {word: words.count(word) for word in meaningful_words}
        main_topic = max(word_count, key=word_count.get).title()
    else:
        main_topic = "Amazing Content"
    
    # Basic templates
    if platform == "youtube":
        return [
            f"The Ultimate {main_topic} Guide",
            f"Everything About {main_topic}",
            f"How {main_topic} Works",
            f"The {main_topic} Secret",
            f"Why {main_topic} Matters"
        ]
    elif platform == "instagram":
        return [
            f"✨ {main_topic} vibes",
            f"{main_topic} content 😍",
            f"All about {main_topic}",
            f"{main_topic} energy 🔥",
            f"Living for {main_topic} 💫"
        ]
    elif platform == "tiktok":
        return [
            f"{main_topic} hack!",
            f"Why {main_topic} is trending",
            f"The {main_topic} trend",
            f"{main_topic} is everything",
            f"This {main_topic} content!"
        ]
    else:  # twitter
        return [
            f"Thread: {main_topic} 🧵",
            f"The {main_topic} discussion",
            f"{main_topic} is trending",
            f"Hot take: {main_topic}",
            f"Breaking: {main_topic}"
        ]

if __name__ == "__main__":
    # Test the Phi-3 generator
    test_text = "Today I'm going to show you how to make the perfect pasta carbonara. This is a classic Italian dish that's actually quite simple to make."
    
    print("=== Testing Phi-3 Mini Title Generator ===")
    titles = generate_titles(test_text, "youtube")
    for i, title in enumerate(titles, 1):
        print(f"{i}. {title}")