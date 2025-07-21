import logging
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

# Translation service using Facebook's NLLB-200 model
logger = logging.getLogger(__name__)

class TranslationService:
    """
    Translation service using Facebook's NLLB-200 model for global audience reach.
    Supports translation of video captions and transcriptions to multiple languages.
    """
    
    def __init__(self):
        self.pipe = None
        self.model_loaded = False
        self.supported_languages = {
            # Most common languages for video content
            'en': 'eng_Latn',  # English
            'es': 'spa_Latn',  # Spanish
            'fr': 'fra_Latn',  # French
            'de': 'deu_Latn',  # German
            'it': 'ita_Latn',  # Italian
            'pt': 'por_Latn',  # Portuguese
            'ru': 'rus_Cyrl',  # Russian
            'ja': 'jpn_Jpan',  # Japanese
            'ko': 'kor_Hang',  # Korean
            'zh': 'zho_Hans',  # Chinese (Simplified)
            'zh-tw': 'zho_Hant',  # Chinese (Traditional)
            'ar': 'arb_Arab',  # Arabic
            'hi': 'hin_Deva',  # Hindi
            'th': 'tha_Thai',  # Thai
            'vi': 'vie_Latn',  # Vietnamese
            'tr': 'tur_Latn',  # Turkish
            'nl': 'nld_Latn',  # Dutch
            'pl': 'pol_Latn',  # Polish
            'sv': 'swe_Latn',  # Swedish
            'da': 'dan_Latn',  # Danish
            'no': 'nor_Latn',  # Norwegian
            'fi': 'fin_Latn',  # Finnish
            'he': 'heb_Hebr',  # Hebrew
            'cs': 'ces_Latn',  # Czech
            'hu': 'hun_Latn',  # Hungarian
            'ro': 'ron_Latn',  # Romanian
            'uk': 'ukr_Cyrl',  # Ukrainian
            'bg': 'bul_Cyrl',  # Bulgarian
            'hr': 'hrv_Latn',  # Croatian
            'sk': 'slk_Latn',  # Slovak
            'sl': 'slv_Latn',  # Slovenian
            'et': 'est_Latn',  # Estonian
            'lv': 'lav_Latn',  # Latvian
            'lt': 'lit_Latn',  # Lithuanian
            'mt': 'mlt_Latn',  # Maltese
            'id': 'ind_Latn',  # Indonesian
            'ms': 'zsm_Latn',  # Malay
            'tl': 'tgl_Latn',  # Filipino
            'sw': 'swh_Latn',  # Swahili
            'am': 'amh_Ethi',  # Amharic
            'bn': 'ben_Beng',  # Bengali
            'gu': 'guj_Gujr',  # Gujarati
            'kn': 'kan_Knda',  # Kannada
            'ml': 'mal_Mlym',  # Malayalam
            'mr': 'mar_Deva',  # Marathi
            'ne': 'nep_Deva',  # Nepali
            'or': 'ory_Orya',  # Odia
            'pa': 'pan_Guru',  # Punjabi
            'si': 'sin_Sinh',  # Sinhala
            'ta': 'tam_Taml',  # Tamil
            'te': 'tel_Telu',  # Telugu
            'ur': 'urd_Arab',  # Urdu
        }
        
        # Initialize model lazily
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the NLLB-200 translation model"""
        try:
            logger.info("Loading NLLB-200 translation model...")
            
            # Try to fix protobuf compatibility issues
            import os
            os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
            
            # Import with error handling
            from transformers import pipeline
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Use the pipeline with CPU to avoid accelerate dependency issues
            self.pipe = pipeline(
                "translation", 
                model="facebook/nllb-200-distilled-600M",
                device=-1  # Force CPU
            )
            self.model_loaded = True
            
            logger.info("✅ NLLB-200 translation model loaded successfully")
            
        except ImportError as e:
            logger.error(f"❌ Missing required libraries for translation: {e}")
            logger.info("Please install required packages: pip install transformers torch")
            self.model_loaded = False
        except Exception as e:
            logger.error(f"❌ Failed to load NLLB-200 translation model: {e}")
            logger.info("Attempting to use CPU-only fallback...")
            
            # Try CPU fallback with basic configuration
            try:
                from transformers import pipeline
                import warnings
                warnings.filterwarnings("ignore")
                
                self.pipe = pipeline(
                    "translation", 
                    model="facebook/nllb-200-distilled-600M",
                    device=-1  # Force CPU
                )
                self.model_loaded = True
                logger.info("✅ NLLB-200 translation model loaded successfully on CPU")
                
            except Exception as cpu_error:
                logger.error(f"❌ CPU fallback also failed: {cpu_error}")
                logger.info("Will use Google Translate fallback instead")
                self.model_loaded = False
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Return supported language codes and their names"""
        language_names = {
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
            'sw': 'Swahili',
            'am': 'Amharic',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'mr': 'Marathi',
            'ne': 'Nepali',
            'or': 'Odia',
            'pa': 'Punjabi',
            'si': 'Sinhala',
            'ta': 'Tamil',
            'te': 'Telugu',
            'ur': 'Urdu'
        }
        
        return {code: language_names.get(code, code) for code in self.supported_languages.keys()}
    
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
            logger.error("Translation model not loaded")
            return None
            
        if not text or not text.strip():
            return text
            
        # Validate language codes
        if source_lang not in self.supported_languages:
            logger.error(f"Unsupported source language: {source_lang}")
            return None
            
        if target_lang not in self.supported_languages:
            logger.error(f"Unsupported target language: {target_lang}")
            return None
            
        # Skip translation if source and target are the same
        if source_lang == target_lang:
            return text
            
        try:
            # Get the proper language codes for NLLB
            src_code = self.supported_languages[source_lang]
            tgt_code = self.supported_languages[target_lang]
            
            # Perform translation
            result = self.pipe(text, src_lang=src_code, tgt_lang=tgt_code)
            
            if result and isinstance(result, list) and len(result) > 0:
                translated_text = result[0].get('translation_text', '')
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
            logger.error("Translation model not loaded")
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
            
            # Handle word-level translations if available
            if 'words' in segment and segment['words']:
                translated_words = []
                for word_info in segment['words']:
                    word_text = word_info.get('word', '')
                    if word_text:
                        # For word-level translation, we need to be careful about context
                        # For now, we'll translate individual words, but this could be improved
                        # by translating phrases or using the segment translation to map words
                        translated_word_text = self.translate_text(word_text, source_lang, target_lang)
                        if translated_word_text:
                            translated_word = word_info.copy()
                            translated_word['word'] = translated_word_text
                            translated_word['original_word'] = word_text
                            translated_words.append(translated_word)
                        else:
                            translated_words.append(word_info)
                    else:
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
            logger.error("Translation model not loaded")
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
            'translation_model': 'facebook/nllb-200-distilled-600M'
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
            logger.error("Translation model not loaded")
            return texts
            
        if source_lang == target_lang:
            return texts
            
        translated_texts = []
        
        for text in texts:
            translated_text = self.translate_text(text, source_lang, target_lang)
            translated_texts.append(translated_text if translated_text else text)
        
        return translated_texts

# Global instance with fallback
try:
    translation_service = TranslationService()
    if not translation_service.model_loaded:
        logger.info("NLLB-200 model not available, using Google Translate fallback")
        from simple_translation_service import simple_translation_service
        translation_service = simple_translation_service
except Exception as e:
    logger.error(f"Failed to initialize primary translation service: {e}")
    logger.info("Using Google Translate fallback")
    from simple_translation_service import simple_translation_service
    translation_service = simple_translation_service