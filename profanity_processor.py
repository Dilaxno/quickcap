"""
Profanity processor for video captions
"""

import logging
from profanity_list import contains_profanity, censor_profanity, PROFANITY_WORDS

logger = logging.getLogger(__name__)

class ProfanityProcessor:
    def __init__(self, enable_filter=False, custom_words=None, filter_mode='both'):
        """
        Initialize profanity processor
        
        Args:
            enable_filter: Whether profanity filtering is enabled
            custom_words: List of custom profanity words to add
            filter_mode: Filtering mode - 'text', 'audio', or 'both'
        """
        self.enable_filter = enable_filter
        self.filter_mode = filter_mode  # 'text', 'audio', or 'both'
        self.profanity_words = set(PROFANITY_WORDS)
        
        # Add custom words if provided
        if custom_words:
            for word in custom_words:
                if word and word.strip():
                    self.profanity_words.add(word.strip().lower())
            logger.info(f"Added {len(custom_words)} custom profanity words")
        
        logger.info(f"Profanity processor initialized with mode: {filter_mode}")
    
    def contains_profanity_custom(self, word):
        """
        Check if a word contains profanity using both default and custom words
        
        Args:
            word: The word to check
            
        Returns:
            bool: True if profanity is found, False otherwise
        """
        if not self.enable_filter or not word:
            return False
        
        word_lower = word.lower().strip()
        
        # Direct match in our custom set
        if word_lower in self.profanity_words:
            return True
        
        # Use the original function for more complex matching
        return contains_profanity(word)
    
    def process_words_for_profanity(self, words):
        """
        Process a list of word dictionaries and identify profanity
        
        Args:
            words: List of word dictionaries with 'word', 'start', 'end' keys
            
        Returns:
            tuple: (processed_words, profanity_timestamps)
                - processed_words: List of words with text censoring applied based on filter mode
                - profanity_timestamps: List of (start_time, end_time, original_word) for audio beeps
        """
        if not self.enable_filter:
            return words, []
        
        processed_words = []
        profanity_timestamps = []
        
        for word_info in words:
            word = word_info['word']
            start_time = word_info['start']
            end_time = word_info['end']
            
            if self.contains_profanity_custom(word):
                # This word contains profanity
                processed_word_info = word_info.copy()
                processed_word_info['is_profanity'] = True
                processed_word_info['original_word'] = word
                processed_word_info['filter_mode'] = self.filter_mode
                
                # Apply text censoring based on filter mode
                if self.filter_mode in ['text', 'both']:
                    # Censor the text in captions
                    censored_word = censor_profanity(word)
                    processed_word_info['word'] = censored_word
                    logger.info(f"Text censored: '{word}' -> '{censored_word}' at {start_time:.2f}s-{end_time:.2f}s")
                else:
                    # Keep original word in captions (audio-only mode)
                    processed_word_info['word'] = word
                    logger.info(f"Audio-only filtering: '{word}' will be beeped but not censored in text at {start_time:.2f}s-{end_time:.2f}s")
                
                # Add to audio beeping list based on filter mode
                if self.filter_mode in ['audio', 'both']:
                    profanity_timestamps.append((start_time, end_time, word))
                    logger.info(f"Audio beep scheduled for: '{word}' at {start_time:.2f}s-{end_time:.2f}s")
                
                processed_words.append(processed_word_info)
            else:
                # Clean word, keep as is
                word_info_copy = word_info.copy()
                word_info_copy['is_profanity'] = False
                processed_words.append(word_info_copy)
        
        text_censored = len([w for w in processed_words if w.get('is_profanity') and self.filter_mode in ['text', 'both']])
        audio_beeped = len(profanity_timestamps)
        
        logger.info(f"Profanity processing complete (mode: {self.filter_mode}): {text_censored} text censored, {audio_beeped} audio beeps scheduled")
        return processed_words, profanity_timestamps
    
    def process_phrases_for_profanity(self, phrases):
        """
        Process phrases and identify profanity within them
        
        Args:
            phrases: List of phrases (each phrase is a list of word dictionaries)
            
        Returns:
            tuple: (processed_phrases, profanity_timestamps)
        """
        if not self.enable_filter:
            return phrases, []
        
        processed_phrases = []
        all_profanity_timestamps = []
        
        for phrase in phrases:
            processed_phrase, profanity_timestamps = self.process_words_for_profanity(phrase)
            processed_phrases.append(processed_phrase)
            all_profanity_timestamps.extend(profanity_timestamps)
        
        return processed_phrases, all_profanity_timestamps
    
    def get_profanity_color(self):
        """
        Get the color to use for profanity highlighting (red)
        
        Returns:
            tuple: RGBA color tuple for red
        """
        return (255, 0, 0, 255)  # Red color for profanity
    
    def should_highlight_as_profanity(self, word_info):
        """
        Check if a word should be highlighted as profanity
        
        Args:
            word_info: Word dictionary with profanity information
            
        Returns:
            bool: True if word should be highlighted as profanity
        """
        return self.enable_filter and word_info.get('is_profanity', False)
    
    def should_apply_audio_beep(self):
        """
        Check if audio beeping should be applied based on filter mode
        
        Returns:
            bool: True if audio beeping should be applied
        """
        return self.enable_filter and self.filter_mode in ['audio', 'both']
    
    def should_apply_text_censoring(self):
        """
        Check if text censoring should be applied based on filter mode
        
        Returns:
            bool: True if text censoring should be applied
        """
        return self.enable_filter and self.filter_mode in ['text', 'both']
    
    def get_filter_mode_description(self):
        """
        Get a human-readable description of the current filter mode
        
        Returns:
            str: Description of the filter mode
        """
        if not self.enable_filter:
            return "Disabled"
        
        descriptions = {
            'text': 'Text censoring only (words hidden in captions)',
            'audio': 'Audio censoring only (beep sounds, words visible)',
            'both': 'Full censoring (both text and audio)'
        }
        return descriptions.get(self.filter_mode, 'Unknown mode')