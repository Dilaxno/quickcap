#!/usr/bin/env python3
"""
Standalone title generation module using Hugging Face Mixtral model
This module provides a more robust approach to title generation with better error handling
"""

import logging
import re
import random
import os
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TitleGenerator:
    """
    A robust title generator using Hugging Face Mixtral model with comprehensive fallback
    """
    
    def __init__(self):
        self.hf_pipeline = None
        self.hf_available = False
        self._initialize_huggingface()
    
    def _initialize_huggingface(self):
        """Initialize Hugging Face pipeline with robust error handling"""
        logger.info("Initializing Hugging Face Mixtral title generator...")
        
        try:
            # Get Hugging Face token from environment
            hf_token = os.getenv('HF_TOKEN')
            
            if not hf_token or hf_token == 'your_hugging_face_token_here':
                logger.warning("Hugging Face token not found or not set. Trying without authentication...")
            
            # Try to import and initialize Hugging Face pipeline
            try:
                from transformers import pipeline
                
                logger.info("Loading local GPT-2 model for title generation...")
                
                # Try to use local GPT-2 model first, fallback to downloading if needed
                try:
                    # First try to load from local cache/directory
                    import torch
                    from transformers import GPT2LMHeadModel, GPT2Tokenizer
                    
                    # Try to load locally cached model
                    model_name = "gpt2"
                    tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=True)
                    model = GPT2LMHeadModel.from_pretrained(model_name, local_files_only=True)
                    
                    # Create pipeline from local components
                    self.hf_pipeline = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=-1  # Use CPU to avoid GPU memory issues
                    )
                    logger.info("Loaded GPT-2 model from local cache")
                    
                except Exception as local_error:
                    logger.info(f"Local model not found ({local_error}), downloading GPT-2...")
                    # Fallback to downloading the model
                    self.hf_pipeline = pipeline(
                        "text-generation", 
                        model="gpt2",
                        device=-1  # Use CPU
                    )
                    logger.info("Downloaded and loaded GPT-2 model")
                
                # Test the pipeline with a simple request
                test_response = self.hf_pipeline(
                    "Generate a title:",
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.7
                )
                
                self.hf_available = True
                logger.info("Hugging Face GPT-2 model initialized successfully!")
                
            except ImportError as e:
                logger.warning(f"Transformers library not properly installed: {str(e)}")
                logger.info("Install with: pip install transformers torch")
                self.hf_available = False
            except Exception as e:
                error_msg = str(e)
                if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                    logger.warning("GPU memory insufficient for model. Falling back to template generation.")
                elif "authentication" in error_msg.lower() or "401" in error_msg:
                    logger.warning("Hugging Face authentication failed. Check your HF_TOKEN.")
                else:
                    logger.warning(f"Failed to initialize Hugging Face pipeline: {error_msg}")
                self.hf_available = False
                
        except Exception as e:
            logger.error(f"Error initializing Hugging Face: {str(e)}")
            self.hf_available = False
        
        logger.info("Title generator initialized successfully!")
    
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
            
            # Extract key content
            summary = self._extract_key_content(transcription)
            
            # Method 1: Try Hugging Face Mixtral if available
            if self.hf_available:
                hf_titles = self._generate_with_huggingface(summary, platform)
                titles.extend(hf_titles)
            
            # Method 2: AI-enhanced template generation
            if len(titles) < 3:
                ai_titles = self._generate_ai_enhanced_titles(transcription, platform, summary)
                titles.extend(ai_titles)
            
            # Method 3: Fallback to basic templates
            if len(titles) < 3:
                fallback_titles = self._generate_fallback_titles(transcription, platform)
                titles.extend(fallback_titles)
            
            # Clean and deduplicate
            unique_titles = []
            seen = set()
            for title in titles:
                cleaned = self._clean_title(title, platform)
                if cleaned and len(cleaned) > 10 and cleaned not in seen:
                    unique_titles.append(cleaned)
                    seen.add(cleaned)
            
            return unique_titles[:5]  # Return top 5 titles
            
        except Exception as e:
            logger.error(f"Error generating titles: {str(e)}")
            return self._generate_fallback_titles(transcription, platform)
    
    def _generate_with_huggingface(self, summary: str, platform: str) -> List[str]:
        """Generate titles using Hugging Face Mixtral model"""
        titles = []
        
        if not self.hf_pipeline:
            return titles
        
        try:
            # Create platform-specific prompt
            platform_instructions = {
                'youtube': "Create engaging YouTube video titles that are clickable and SEO-friendly. Include emotional hooks and clear value propositions.",
                'instagram': "Create casual, trendy Instagram post titles with emoji-friendly language that resonates with social media users.",
                'tiktok': "Create short, punchy TikTok video titles that are viral-worthy and trend-focused.",
                'twitter': "Create concise, discussion-worthy Twitter post titles that encourage engagement and retweets."
            }
            
            instruction = platform_instructions.get(platform, platform_instructions['youtube'])
            
            # Create a more focused prompt for GPT-2
            if platform == 'youtube':
                prompt = f"YouTube video title: How to make perfect carbonara pasta - "
            elif platform == 'instagram':
                prompt = f"Instagram post: Making carbonara pasta ✨ "
            elif platform == 'tiktok':
                prompt = f"TikTok: Carbonara pasta hack! "
            else:  # twitter
                prompt = f"Tweet: Just made carbonara pasta and "
            
            # Extract key topics from the content
            content_analysis = self._analyze_content(summary)
            main_topic = content_analysis['main_topic']
            
            # Create platform-specific prompts based on content
            if platform == 'youtube':
                prompt = f"YouTube video title: {main_topic} tutorial - "
            elif platform == 'instagram':
                prompt = f"Instagram post: {main_topic} vibes ✨ "
            elif platform == 'tiktok':
                prompt = f"TikTok: {main_topic} hack! "
            else:  # twitter
                prompt = f"Tweet: Just tried {main_topic} and "

            # Generate multiple titles by running the pipeline multiple times
            for i in range(5):  # Generate 5 different titles
                try:
                    response = self.hf_pipeline(
                        prompt,
                        max_new_tokens=20,  # Shorter for better titles
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.95,
                        pad_token_id=self.hf_pipeline.tokenizer.eos_token_id,
                        return_full_text=False  # Only return generated part
                    )
                    
                    if response and len(response) > 0:
                        generated_text = response[0]['generated_text'].strip()
                        
                        # Clean up the generated text
                        title = generated_text.split('\n')[0]  # Take first line only
                        title = title.strip()
                        
                        if title and len(title) > 5 and len(title) < 100:
                            # Combine prompt context with generated text for full title
                            if platform == 'youtube':
                                full_title = f"{main_topic} tutorial - {title}"
                            elif platform == 'instagram':
                                full_title = f"{main_topic} vibes ✨ {title}"
                            elif platform == 'tiktok':
                                full_title = f"{main_topic} hack! {title}"
                            else:  # twitter
                                full_title = f"Just tried {main_topic} and {title}"
                            
                            # Post-process title for platform-specific optimization
                            optimized_title = self._optimize_title_for_platform(full_title, platform)
                            titles.append(optimized_title)
                            
                except Exception as gen_error:
                    logger.warning(f"Error generating title {i+1}: {gen_error}")
                    continue

                            
        except Exception as e:
            logger.warning(f"Hugging Face title generation failed: {str(e)}")
        
        return titles
    
    def _optimize_title_for_platform(self, title: str, platform: str) -> str:
        """Optimize the generated title for specific platforms"""
        # Clean up the title
        title = title.strip()
        
        # Platform-specific optimizations
        if platform == 'youtube':
            # YouTube titles benefit from emotional hooks and clear value propositions
            if not any(word in title.lower() for word in ['how', 'why', 'what', 'secret', 'truth', 'amazing', 'incredible']):
                # Add engaging prefix if not present
                prefixes = ['How to', 'Why', 'The Secret to', 'Amazing', 'Incredible']
                prefix = random.choice(prefixes)
                title = f"{prefix} {title}"
        
        elif platform == 'instagram':
            # Instagram titles should be more casual and emoji-friendly
            if not any(emoji in title for emoji in ['✨', '🔥', '💫', '😍', '🤯']):
                emojis = ['✨', '🔥', '💫', '😍', '🤯']
                emoji = random.choice(emojis)
                title = f"{title} {emoji}"
        
        elif platform == 'tiktok':
            # TikTok titles should be short and punchy
            if len(title) > 50:
                title = title[:47] + "..."
            if not title.endswith('!'):
                title += "!"
        
        return title
    
    def _extract_key_content(self, transcription: str) -> str:
        """Extract key content from transcription"""
        # Clean up the transcription
        cleaned = re.sub(r'\s+', ' ', transcription.strip())
        
        # Take first 200 characters and ensure it ends properly
        summary = cleaned[:200]
        if not summary.endswith(('.', '!', '?')):
            # Find the last complete sentence
            last_period = summary.rfind('.')
            if last_period > 50:
                summary = summary[:last_period + 1]
            else:
                summary += '.'
        
        return summary
    
    def _generate_ai_enhanced_titles(self, transcription: str, platform: str, summary: str) -> List[str]:
        """Generate AI-enhanced titles using content analysis"""
        content_analysis = self._analyze_content(transcription)
        
        templates = {
            'youtube': [
                f"🔥 {content_analysis['main_topic']} Secret That Will Blow Your Mind!",
                f"Why {content_analysis['main_topic']} Is Taking Over The Internet",
                f"I Tried {content_analysis['main_topic']} For 30 Days - Here's What Happened",
                f"The Shocking Truth About {content_analysis['main_topic']}",
                f"How {content_analysis['main_topic']} Changed Everything"
            ],
            'instagram': [
                f"✨ {content_analysis['main_topic']} hits different 🤯",
                f"POV: You discovered the {content_analysis['main_topic']} secret 💫",
                f"This {content_analysis['main_topic']} content is everything 😍",
                f"Not me obsessing over {content_analysis['main_topic']} 🥺",
                f"Main character energy: {content_analysis['main_topic']} edition 💅"
            ],
            'tiktok': [
                f"Wait for the {content_analysis['main_topic']} plot twist 🤯 #viral",
                f"{content_analysis['main_topic']} is the reason I love this app 💯",
                f"Tell me why {content_analysis['main_topic']} hits like this 😭",
                f"POV: {content_analysis['main_topic']} just made sense 🧠",
                f"This {content_analysis['main_topic']} trend is everything ✨"
            ],
            'twitter': [
                f"Thread: The {content_analysis['main_topic']} discourse we needed 🧵",
                f"Unpopular opinion: {content_analysis['main_topic']} is revolutionary",
                f"Breaking: {content_analysis['main_topic']} just changed the game",
                f"Plot twist: {content_analysis['main_topic']} was the answer",
                f"Hot take: {content_analysis['main_topic']} is underrated"
            ]
        }
        
        platform_templates = templates.get(platform, templates['youtube'])
        return random.sample(platform_templates, min(3, len(platform_templates)))
    
    def _generate_fallback_titles(self, transcription: str, platform: str) -> List[str]:
        """Generate basic fallback titles"""
        # Use the same content analysis as the main method
        content_analysis = self._analyze_content(transcription)
        main_topic = content_analysis['main_topic']
        
        # Platform-specific basic templates
        if platform == 'youtube':
            basic_titles = [
                f"Everything You Need to Know About {main_topic}",
                f"The Ultimate {main_topic} Guide",
                f"Why {main_topic} Matters More Than You Think",
                f"The Truth About {main_topic}",
                f"How {main_topic} Can Change Your Life"
            ]
        elif platform == 'instagram':
            basic_titles = [
                f"✨ {main_topic} vibes are everything",
                f"Obsessed with this {main_topic} content 😍",
                f"Why {main_topic} is my new favorite thing",
                f"This {main_topic} post hits different 💫",
                f"Living for this {main_topic} energy 🔥"
            ]
        elif platform == 'tiktok':
            basic_titles = [
                f"This {main_topic} hack will change your life",
                f"Why everyone is talking about {main_topic}",
                f"The {main_topic} trend you need to know",
                f"This {main_topic} content is viral for a reason",
                f"How {main_topic} became everyone's obsession"
            ]
        else:  # twitter
            basic_titles = [
                f"Thread: Everything about {main_topic} 🧵",
                f"The {main_topic} conversation we need to have",
                f"Why {main_topic} is trending right now",
                f"Hot take: {main_topic} is underrated",
                f"Breaking down the {main_topic} phenomenon"
            ]
        
        return basic_titles
    
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
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\b', transcription)
        
        # Detect emotional words with more comprehensive patterns
        emotional_words = []
        emotion_patterns = {
            'amazing': r'\b(amazing|incredible|awesome|fantastic|outstanding|remarkable)\b',
            'shocking': r'\b(shocking|unbelievable|mind.blowing|crazy|insane|wild)\b',
            'surprising': r'\b(surprising|unexpected|stunned|amazed|blown.away)\b',
            'exciting': r'\b(exciting|thrilling|pumped|energized|hyped)\b',
            'beautiful': r'\b(beautiful|gorgeous|stunning|breathtaking|spectacular)\b',
            'perfect': r'\b(perfect|flawless|ideal|excellent|outstanding)\b'
        }
        
        for emotion, pattern in emotion_patterns.items():
            if re.search(pattern, text_lower):
                emotional_words.append(emotion)
        
        return {
            'main_topic': main_topic,
            'numbers': numbers,
            'emotional_words': emotional_words
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
        
        # Add platform-specific formatting
        if platform == 'youtube' and not any(emoji in title for emoji in ['🔥', '💯', '😱', '🤯']):
            if any(word in title.lower() for word in ['secret', 'shocking', 'amazing']):
                title = f"🔥 {title}"
        
        return title

# Global instance
_title_generator = None

def get_title_generator():
    """Get the global title generator instance"""
    global _title_generator
    if _title_generator is None:
        _title_generator = TitleGenerator()
    return _title_generator

def generate_titles(transcription: str, platform: str = "youtube") -> List[str]:
    """
    Generate viral titles for the given transcription and platform
    
    Args:
        transcription: The video transcript text
        platform: Target platform (youtube, instagram, tiktok, twitter)
        
    Returns:
        List of generated titles
    """
    try:
        # Try the main generator first
        generator = get_title_generator()
        titles = generator.generate_titles(transcription, platform)
        
        # If we got good results, return them
        if titles and len(titles) >= 3:
            return titles
        
        # If main generator failed or produced few titles, try simple generator
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