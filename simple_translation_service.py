import logging
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

# Simple translation service using Google Translate as fallback
logger = logging.getLogger(__name__)

class SimpleTranslationService:
    """
    Simple translation service that uses Google Translate as fallback
    when NLLB-200 model is not available.
    """
    
    def __init__(self):
        self.translator = None
        self.model_loaded = False
        self.supported_languages = {
            # Most common languages for video content
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese (Simplified)',
            'zh-cn': 'Chinese (Simplified)',
            'zh-tw': 'Chinese (Traditional)',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'tr': 'Turkish',
            'nl': 'Dutch',
            'pl': 'Polish',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
            'he': 'Hebrew',
            'cs': 'Czech',
            'hu': 'Hungarian',
            'ro': 'Romanian',
            'uk': 'Ukrainian',
            'bg': 'Bulgarian',
            'hr': 'Croatian',
            'sk': 'Slovak',
            'sl': 'Slovenian',
            'et': 'Estonian',
            'lv': 'Latvian',
            'lt': 'Lithuanian',
            'mt': 'Maltese',
            'id': 'Indonesian',
            'ms': 'Malay',
            'tl': 'Filipino',
            'sw': 'Swahili'
        }
        
        # Initialize translator
        self._initialize_translator()
    
    def _initialize_translator(self):
        """Initialize the Google Translator"""
        try:
            logger.info("Loading Google Translator...")
            from googletrans import Translator
            
            self.translator = Translator()
            self.model_loaded = True
            
            logger.info("✅ Google Translator loaded successfully")
            
        except ImportError as e:
            logger.error(f"❌ Missing googletrans library: {e}")
            logger.info("Please install: pip install googletrans==4.0.0rc1")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"❌ Failed to load Google Translator: {e}")
            self.model_loaded = False
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Return supported language codes and their names"""
        return self.supported_languages
    
    def translate_text(self, text: str, source_lang: str = 'en', target_lang: str = 'es') -> Optional[str]:
        """
        Translate a single text string from source language to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'en')
            target_lang: Target language code (e.g., 'es')
            
        Returns:
            Translated text or None if translation fails
        """
        if not self.model_loaded:
            logger.error("Translation service not loaded")
            return None
            
        if not text or not text.strip():
            return text
            
        # Skip translation if source and target are the same
        if source_lang == target_lang:
            return text
            
        try:
            # Perform translation
            result = self.translator.translate(text, src=source_lang, dest=target_lang)
            
            if result and result.text:
                translated_text = result.text
                logger.debug(f"Translated '{text[:50]}...' from {source_lang} to {target_lang}")
                return translated_text
            else:
                logger.error(f"Invalid translation result: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return None
    
    def translate_segments(self, segments: List[Dict], source_lang: str = 'en', target_lang: str = 'es') -> List[Dict]:
        """
        Translate transcription segments while preserving timing information
        
        Args:
            segments: List of transcription segments with text and timing
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated segments with preserved timing
        """
        if not self.model_loaded:
            logger.error("Translation service not loaded")
            return segments
            
        if source_lang == target_lang:
            return segments
            
        translated_segments = []
        
        for segment in segments:
            # Create a copy of the segment to preserve original data
            translated_segment = segment.copy()
            
            # Translate the main text
            original_text = segment.get('text', '')
            if original_text:
                translated_text = self.translate_text(original_text, source_lang, target_lang)
                if translated_text:
                    translated_segment['text'] = translated_text
                    # Keep track of original text for reference
                    translated_segment['original_text'] = original_text
            
            # For word-level translations, we'll use the segment translation
            # and redistribute the timing (this is a simplified approach)
            if 'words' in segment and segment['words']:
                translated_words = []
                if 'text' in translated_segment:
                    translated_word_list = translated_segment['text'].split()
                    original_words = segment['words']
                    
                    # Calculate timing distribution
                    segment_start = segment.get('start', 0)
                    segment_end = segment.get('end', 0)
                    segment_duration = segment_end - segment_start
                    
                    if translated_word_list and segment_duration > 0:
                        word_duration = segment_duration / len(translated_word_list)
                        
                        for i, translated_word in enumerate(translated_word_list):
                            word_start = segment_start + (i * word_duration)
                            word_end = segment_start + ((i + 1) * word_duration)
                            
                            translated_words.append({
                                'word': translated_word,
                                'start': word_start,
                                'end': word_end,
                                'original_word': original_words[i]['word'] if i < len(original_words) else translated_word
                            })
                    else:
                        # Fallback: keep original timing structure
                        for word_info in original_words:
                            translated_words.append(word_info)
                
                translated_segment['words'] = translated_words
            
            translated_segments.append(translated_segment)
        
        return translated_segments
    
    def translate_transcription(self, transcription_data: Dict, source_lang: str = 'en', target_lang: str = 'es') -> Dict:
        """
        Translate an entire transcription object while preserving structure
        
        Args:
            transcription_data: Transcription data with segments and metadata
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated transcription data
        """
        if not self.model_loaded:
            logger.error("Translation service not loaded")
            return transcription_data
            
        if source_lang == target_lang:
            return transcription_data
            
        # Create a copy to avoid modifying original data
        translated_data = transcription_data.copy()
        
        # Translate segments
        if 'segments' in transcription_data:
            translated_segments = self.translate_segments(
                transcription_data['segments'], 
                source_lang, 
                target_lang
            )
            translated_data['segments'] = translated_segments
        
        # Translate main text if available
        if 'text' in transcription_data:
            translated_text = self.translate_text(
                transcription_data['text'], 
                source_lang, 
                target_lang
            )
            if translated_text:
                translated_data['text'] = translated_text
                translated_data['original_text'] = transcription_data['text']
        
        # Add translation metadata
        translated_data['translation_info'] = {
            'source_language': source_lang,
            'target_language': target_lang,
            'translated_at': datetime.now().isoformat(),
            'translation_model': 'googletrans'
        }
        
        return translated_data
    
    def batch_translate(self, texts: List[str], source_lang: str = 'en', target_lang: str = 'es') -> List[Optional[str]]:
        """
        Translate multiple texts in batch for better efficiency
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts (same order as input)
        """
        if not self.model_loaded:
            logger.error("Translation service not loaded")
            return texts
            
        if source_lang == target_lang:
            return texts
            
        translated_texts = []
        
        for text in texts:
            translated_text = self.translate_text(text, source_lang, target_lang)
            translated_texts.append(translated_text if translated_text else text)
        
        return translated_texts

# Global instance
simple_translation_service = SimpleTranslationService()