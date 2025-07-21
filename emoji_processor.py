# -*- coding: utf-8 -*-
"""
Emoji processor module for adding emojis to captions
"""
import re
import json
import os
from typing import Dict, List, Optional, Union, Any

# Load emoji keyword mapping
EMOJI_MAPPING = {
    # Emotions and reactions
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "excited": "🤩",
    "love": "❤️",
    "laugh": "😂",
    "smile": "😄",
    "crying": "😭",
    "shocked": "😲",
    "surprised": "😮",
    "confused": "😕",
    "worried": "😟",
    "scared": "😨",
    "tired": "😴",
    "sick": "🤒",
    
    # Actions
    "eat": "🍽️",
    "eating": "🍽️",
    "drink": "🥤",
    "drinking": "🥤",
    "sleep": "💤",
    "sleeping": "💤",
    "run": "🏃",
    "running": "🏃",
    "walk": "🚶",
    "walking": "🚶",
    "dance": "💃",
    "dancing": "💃",
    "sing": "🎤",
    "singing": "🎤",
    "work": "💼",
    "working": "💼",
    "study": "📚",
    "studying": "📚",
    "write": "✍️",
    "writing": "✍️",
    "read": "📖",
    "reading": "📖",
    "play": "🎮",
    "playing": "🎮",
    "swim": "🏊",
    "swimming": "🏊",
    
    # Objects
    "phone": "📱",
    "computer": "💻",
    "laptop": "💻",
    "book": "📕",
    "books": "📚",
    "car": "🚗",
    "cars": "🚗",
    "bike": "🚲",
    "bicycle": "🚲",
    "house": "🏠",
    "home": "🏠",
    "money": "💰",
    "cash": "💵",
    "camera": "📷",
    "photo": "📸",
    "picture": "🖼️",
    "music": "🎵",
    "song": "🎵",
    "food": "🍲",
    "coffee": "☕",
    "tea": "🍵",
    "water": "💧",
    "beer": "🍺",
    "wine": "🍷",
    
    # Nature
    "sun": "☀️",
    "moon": "🌙",
    "star": "⭐",
    "stars": "✨",
    "cloud": "☁️",
    "clouds": "☁️",
    "rain": "🌧️",
    "snow": "❄️",
    "tree": "🌳",
    "trees": "🌳",
    "flower": "🌸",
    "flowers": "🌸",
    "mountain": "⛰️",
    "mountains": "🏔️",
    "beach": "🏖️",
    "ocean": "🌊",
    "sea": "🌊",
    "river": "🏞️",
    
    # Time
    "time": "⏰",
    "clock": "🕒",
    "today": "📅",
    "tomorrow": "📆",
    "yesterday": "📅",
    "morning": "🌅",
    "afternoon": "🌇",
    "evening": "🌆",
    "night": "🌃",
    "weekend": "🏖️",
    
    # Places
    "school": "🏫",
    "university": "🎓",
    "office": "🏢",
    "hospital": "🏥",
    "restaurant": "🍽️",
    "cafe": "☕",
    "shop": "🛍️",
    "store": "🏪",
    "market": "🛒",
    "airport": "✈️",
    "station": "🚉",
    "hotel": "🏨",
    "gym": "💪",
    
    # Technology
    "internet": "🌐",
    "web": "🕸️",
    "email": "📧",
    "message": "💬",
    "chat": "💬",
    "video": "🎥",
    "movie": "🎬",
    "game": "🎮",
    "app": "📱",
    "software": "💾",
    "download": "⬇️",
    "upload": "⬆️",
    "search": "🔍",
    "battery": "🔋",
    "wifi": "📶",
    
    # Social media
    "like": "👍",
    "dislike": "👎",
    "share": "📤",
    "comment": "💬",
    "follow": "➡️",
    "post": "📝",
    "tweet": "🐦",
    "instagram": "📸",
    "facebook": "👤",
    "youtube": "▶️",
    "tiktok": "🎵",
    
    # Business
    "meeting": "👥",
    "presentation": "📊",
    "chart": "📈",
    "graph": "📊",
    "growth": "📈",
    "decline": "📉",
    "profit": "💰",
    "loss": "📉",
    "sale": "🏷️",
    "discount": "💸",
    "deal": "🤝",
    "contract": "📜",
    "startup": "🚀",
    "business": "💼",
    "company": "🏢",
    
    # Communication
    "talk": "🗣️",
    "talking": "🗣️",
    "speak": "🗣️",
    "speaking": "🗣️",
    "listen": "👂",
    "listening": "👂",
    "call": "📞",
    "calling": "📞",
    "hello": "👋",
    "goodbye": "👋",
    "thanks": "🙏",
    "thank you": "🙏",
    "please": "🙏",
    "sorry": "😔",
    "congratulations": "🎉",
    "congrats": "🎉",
    
    # Travel
    "travel": "✈️",
    "traveling": "✈️",
    "trip": "🧳",
    "vacation": "🏖️",
    "holiday": "🎄",
    "flight": "✈️",
    "train": "🚆",
    "bus": "🚌",
    "taxi": "🚕",
    "subway": "🚇",
    "cruise": "🚢",
    "passport": "🛂",
    "luggage": "🧳",
    "map": "🗺️",
    "compass": "🧭",
    
    # Events
    "party": "🎉",
    "birthday": "🎂",
    "wedding": "💒",
    "graduation": "🎓",
    "anniversary": "🎊",
    "celebration": "🎊",
    "festival": "🎭",
    "concert": "🎵",
    "show": "🎭",
    "exhibition": "🖼️",
    "conference": "👥",
    
    # Sports
    "sport": "🏅",
    "sports": "🏅",
    "football": "🏈",
    "soccer": "⚽",
    "basketball": "🏀",
    "baseball": "⚾",
    "tennis": "🎾",
    "golf": "⛳",
    "swimming": "🏊",
    "running": "🏃",
    "cycling": "🚴",
    "skiing": "⛷️",
    "snowboarding": "🏂",
    "surfing": "🏄",
    "yoga": "🧘",
    "gym": "🏋️",
    "fitness": "💪",
    
    # Weather
    "weather": "🌦️",
    "sunny": "☀️",
    "cloudy": "☁️",
    "rainy": "🌧️",
    "snowy": "❄️",
    "windy": "💨",
    "storm": "⛈️",
    "thunder": "⚡",
    "lightning": "⚡",
    "rainbow": "🌈",
    "hot": "🔥",
    "cold": "❄️",
    "warm": "🌡️",
    "cool": "❄️",
    "humid": "💦",
    
    # Health
    "health": "🏥",
    "healthy": "💪",
    "doctor": "👨‍⚕️",
    "medicine": "💊",
    "pill": "💊",
    "pills": "💊",
    "exercise": "🏃",
    "workout": "🏋️",
    "diet": "🥗",
    "vitamin": "💊",
    "rest": "😴",
    "fever": "🤒",
    "pain": "🤕",
    
    # Food
    "breakfast": "🍳",
    "lunch": "🍽️",
    "dinner": "🍽️",
    "meal": "🍽️",
    "snack": "🍿",
    "fruit": "🍎",
    "vegetable": "🥦",
    "meat": "🥩",
    "chicken": "🍗",
    "fish": "🐟",
    "bread": "🍞",
    "cheese": "🧀",
    "egg": "🥚",
    "pizza": "🍕",
    "burger": "🍔",
    "sandwich": "🥪",
    "salad": "🥗",
    "pasta": "🍝",
    "rice": "🍚",
    "dessert": "🍰",
    "cake": "🎂",
    "ice cream": "🍦",
    "chocolate": "🍫",
    "chocolate chip": "🍫",
    "mint chocolate": "🍫",
    "mint": "🌿",
    "chip": "🍪",
    "candy": "🍬",
    
    # Drinks
    "drink": "🥤",
    "water": "💧",
    "coffee": "☕",
    "tea": "🍵",
    "juice": "🧃",
    "soda": "🥤",
    "beer": "🍺",
    "wine": "🍷",
    "cocktail": "🍸",
    "alcohol": "🍷",
    
    # Holidays
    "christmas": "🎄",
    "new year": "🎆",
    "halloween": "🎃",
    "easter": "🐰",
    "thanksgiving": "🦃",
    "valentine": "❤️",
    
    # Directions
    "up": "⬆️",
    "down": "⬇️",
    "left": "⬅️",
    "right": "➡️",
    "north": "⬆️",
    "south": "⬇️",
    "east": "➡️",
    "west": "⬅️",
    
    # Descriptive
    "new": "✨",
    "old": "👴",
    "big": "🔝",
    "small": "🔽",
    "fast": "⚡",
    "slow": "🐢",
    "high": "⬆️",
    "low": "⬇️",
    "good": "👍",
    "bad": "👎",
    "best": "🏆",
    "worst": "👎",
    "beautiful": "✨",
    "ugly": "😖",
    "strong": "💪",
    "weak": "😓",
    "rich": "💰",
    "poor": "💸",
    "expensive": "💰",
    "cheap": "💸",
    
    # Numbers and symbols
    "number": "🔢",
    "percent": "💯",
    "dollar": "💵",
    "euro": "💶",
    "pound": "💷",
    "yen": "💴",
    "infinity": "♾️",
    "plus": "➕",
    "minus": "➖",
    "equals": "🟰",
    "question": "❓",
    "exclamation": "❗",
    
    # Content creation
    "content": "📝",
    "creator": "🎬",
    "influencer": "🌟",
    "viral": "📈",
    "trending": "📈",
    "subscribe": "🔔",
    "channel": "📺",
    "stream": "🎮",
    "streaming": "🎮",
    "podcast": "🎙️",
    "blog": "📝",
    "vlog": "📹",
    "edit": "✂️",
    "editing": "✂️",
    "caption": "💬",
    "thumbnail": "🖼️",
    "views": "👁️",
    "likes": "👍",
    "comments": "💬",
    "subscribers": "👥",
    "monetize": "💰",
    "sponsor": "🤝",
    "collaboration": "🤝",
    "collab": "🤝",
    "review": "⭐",
    "tutorial": "📚",
    "howto": "📝",
    "unboxing": "📦",
    "challenge": "🏆",
    "reaction": "😲",
    "prank": "😜",
    "gaming": "🎮",
    "livestream": "🔴",
    "shorts": "📱",
    "reels": "📱",
}

class EmojiProcessor:
    """
    Processes text to add emojis based on keyword matching
    """
    
    def __init__(self, emoji_mapping=None, max_emojis_per_segment=2, exact_word_timestamps=False):
        """
        Initialize the emoji processor
        
        Args:
            emoji_mapping: Dictionary mapping keywords to emojis
            max_emojis_per_segment: Maximum number of emojis to add per text segment
            exact_word_timestamps: Whether to add emojis with exact word timestamps
        """
        self.emoji_mapping = emoji_mapping or EMOJI_MAPPING
        self.max_emojis_per_segment = max_emojis_per_segment
        self.exact_word_timestamps = exact_word_timestamps
    
    def add_emojis_to_text(self, text: str) -> str:
        """
        Add emojis to text based on keyword matching
        
        Args:
            text: Text to process
            
        Returns:
            Text with emojis added
        """
        if not text:
            return text
        
        # Split text into sentences for better emoji distribution
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        result = []
        for sentence in sentences:
            # Split sentence into words
            words = sentence.split()
            emojis_added = []
            
            # Process each word
            processed_words = []
            for word in words:
                # Skip processing if we've already added max emojis for this sentence
                if len(emojis_added) >= self.max_emojis_per_segment:
                    processed_words.append(word)
                    continue
                
                # Clean the word for matching (remove punctuation, lowercase)
                clean_word = re.sub(r'[.,!?;:\'"\(\)]', '', word.lower())
                
                # Check if the word is in our emoji map
                if clean_word in self.emoji_mapping and clean_word not in emojis_added:
                    emojis_added.append(clean_word)
                    # Add emoji after the word
                    processed_words.append(f"{word} {self.emoji_mapping[clean_word]}")
                else:
                    processed_words.append(word)
            
            result.append(' '.join(processed_words))
        
        return ' '.join(result)
    
    def process_transcription(self, transcription: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process transcription data to add emojis to text segments
        
        Args:
            transcription: List of transcription segments
            
        Returns:
            Processed transcription with emojis added
        """
        if not transcription:
            return transcription
        
        if self.exact_word_timestamps:
            return self.process_transcription_with_word_timestamps(transcription)
        
        result = []
        for segment in transcription:
            if 'text' in segment and segment['text']:
                # Create a copy of the segment with emojis added to text
                new_segment = segment.copy()
                new_segment['text'] = self.add_emojis_to_text(segment['text'])
                result.append(new_segment)
            else:
                result.append(segment)
        
        return result
    
    def process_transcription_with_word_timestamps(self, transcription: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process transcription data to add emojis with exact word timestamps
        
        Args:
            transcription: List of transcription segments with word-level timing
            
        Returns:
            Processed transcription with emojis added at exact word timestamps
        """
        if not transcription:
            return transcription
        
        result = []
        
        for segment in transcription:
            new_segment = segment.copy()
            
            # Skip if no words with timing
            if 'words' not in segment or not segment['words']:
                result.append(new_segment)
                continue
            
            # Process words with timing
            words_with_emojis = []
            emojis_added = []
            
            # First, check for multi-word phrases
            segment_words = segment['words']
            i = 0
            while i < len(segment_words):
                # Skip if we've reached max emojis for this segment
                if len(emojis_added) >= self.max_emojis_per_segment:
                    words_with_emojis.append(segment_words[i])
                    i += 1
                    continue
                
                # Try to match multi-word phrases (up to 3 words)
                matched_multi_word = False
                for phrase_length in range(3, 0, -1):  # Try 3-word, then 2-word, then 1-word
                    if i + phrase_length <= len(segment_words):
                        # Extract words for the phrase
                        phrase_words = []
                        for j in range(phrase_length):
                            word_obj = segment_words[i + j]
                            if 'word' in word_obj:
                                clean_word = re.sub(r'[.,!?;:\'"\(\)]', '', word_obj['word'].lower())
                                phrase_words.append(clean_word)
                        
                        # Join the words to form a phrase
                        phrase = ' '.join(phrase_words)
                        
                        # Check if the phrase is in our emoji map
                        if phrase in self.emoji_mapping and phrase not in emojis_added:
                            emojis_added.append(phrase)
                            
                            # Add all words in the phrase
                            for j in range(phrase_length):
                                words_with_emojis.append(segment_words[i + j])
                            
                            # Create emoji word object after the last word of the phrase
                            last_word_obj = segment_words[i + phrase_length - 1]
                            emoji_word_obj = {
                                'word': self.emoji_mapping[phrase],
                                'start': last_word_obj['end'],
                                'end': last_word_obj['end'] + 0.5,
                                'is_emoji': True,
                                'original_phrase': phrase
                            }
                            words_with_emojis.append(emoji_word_obj)
                            
                            i += phrase_length
                            matched_multi_word = True
                            print(f"[EMOJI] Added emoji for phrase: '{phrase}' → {self.emoji_mapping[phrase]}")
                            break
                
                # If no multi-word match, process single word
                if not matched_multi_word:
                    word_obj = segment_words[i]
                    
                    # Get the word and its timing
                    if 'word' not in word_obj or 'start' not in word_obj or 'end' not in word_obj:
                        words_with_emojis.append(word_obj)
                        i += 1
                        continue
                    
                    word = word_obj['word']
                    clean_word = re.sub(r'[.,!?;:\'"\(\)]', '', word.lower())
                    
                    # Check if the word is in our emoji map
                    if clean_word in self.emoji_mapping and clean_word not in emojis_added:
                        emojis_added.append(clean_word)
                        
                        # Create a new word object with the emoji
                        emoji_word_obj = {
                            'word': self.emoji_mapping[clean_word],
                            'start': word_obj['end'],  # Place emoji right after the word
                            'end': word_obj['end'] + 0.5,  # Give it a longer duration (0.5 seconds)
                            'is_emoji': True,  # Mark as emoji for special rendering
                            'original_word': clean_word  # Store the original word that triggered this emoji
                        }
                        
                        # Add both the original word and the emoji
                        words_with_emojis.append(word_obj)
                        words_with_emojis.append(emoji_word_obj)
                        print(f"[EMOJI] Added emoji for word: '{clean_word}' → {self.emoji_mapping[clean_word]}")
                    else:
                        words_with_emojis.append(word_obj)
                    
                    i += 1
            
            # Update the segment with emoji-enhanced words
            new_segment['words'] = words_with_emojis
            
            # Also update the segment text to include emojis
            if 'text' in segment and segment['text']:
                new_segment['text'] = self.add_emojis_to_text(segment['text'])
            
            result.append(new_segment)
        
        return result