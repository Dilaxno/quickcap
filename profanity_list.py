"""
Profanity list for censoring in videos
"""

# List of profanity words to detect and censor
PROFANITY_WORDS = {
    # Common profanity
    "fuck", "shit", "ass", "damn", "bitch", "cunt", "dick", "cock", "pussy", 
    "asshole", "bastard", "motherfucker", "bullshit", "crap", "piss", "whore", 
    "slut", "tits", "twat", "wanker", "bollocks", "prick", "fag", "faggot", 
    "nigger", "nigga", "retard", "spastic", "dyke", "homo", "queer", "jerk", "sex",
    
    # Variations and misspellings
    "f*ck", "sh*t", "a$$", "b*tch", "f**k", "s**t", "a**", "b**ch", "fu*k",
    "sh!t", "a$$hole", "b!tch", "fuk", "sht", "azz", "btch", "fck", "stfu", "boobs",
    
    # Compound words
    "dumbass", "jackass", "dumbfuck", "fuckface", "motherfucking", "shithead",
    "asswipe", "dickhead", "condoms", "condom", "cocksucker", "bullcrap", "assfuck", "bitchass", "blowjob",
    
    # Phrases
    "fuck you", "fuck off", "go to hell", "son of a bitch", "piece of shit",
    "what the fuck", "holy shit", "finger me",
    
    # Mild profanity
    
    "freaking", "freakin", "frickin", "fricking", "friggin", "frigging",
    
    # Nudity related
    "nude", "naked", "topless", "bottomless", "nudes", "nudity", "nipple", "nipples",
    "breasts", "breast", "boob", "boobies", "tittie", "titties", "areola", "areolas",
    "cleavage", "bra", "bras", "lingerie", "thong", "thongs", "g-string", "panties",
    "underwear", "bikini", "bikinis", "speedo", "speedos", "buttocks", "butt", "butthole",
    "anus", "anal", "vagina", "vaginal", "vulva", "labia", "clitoris", "penis", "penile",
    "testicle", "testicles", "scrotum", "genital", "genitals", "genitalia", "groping",
    
    # Pornography related
    "porn", "porno", "pornography", "pornographic", "xxx", "adult film", "adult video",
    "smut", "erotica", "erotic", "hentai", "fetish", "kink", "kinky", "bdsm",
    "masturbate", "masturbation", "masturbating", "jerk off", "jerking off",
    "cum", "cumming", "ejaculate", "ejaculation", "orgasm", "climax",
    "dildo", "vibrator", "sex toy", "fleshlight", "butt plug", "anal beads",
    "bondage", "dominatrix", "submissive", "sadism", "masochism", "orgy",
    "gangbang", "threesome", "foursome", "bukkake", "facial", "creampie",
    "deepthroat", "deep throat", "fellatio", "cunnilingus", "rimming", "rimjob",
    "69", "sixty-nine", "doggy style", "missionary", "cowgirl", "reverse cowgirl",
    "hardcore", "softcore", "amateur", "webcam", "camgirl", "onlyfans", "stripper",
    "strip club", "lap dance", "prostitute", "hooker", "escort", "call girl",
    "pimp", "brothel", "red light", "massage parlor", "happy ending"
}

# Identify which profanity entries are phrases (contain spaces)
PROFANITY_PHRASES = {phrase for phrase in PROFANITY_WORDS if ' ' in phrase}
PROFANITY_SINGLE_WORDS = {word for word in PROFANITY_WORDS if ' ' not in word}

# Function to check if a word contains profanity and return the profane word
def contains_profanity(word):
    """
    Check if a word contains any profanity from the list
    
    Args:
        word: The word to check for profanity
        
    Returns:
        True if profanity is found, False otherwise
    """
    if not word:
        return False
        
    word_lower = word.lower().strip()
    
    # Direct match
    if word_lower in PROFANITY_WORDS:
        return True
    
    # Check for compound words containing profanity
    # We need to be more careful with substring matching to avoid false positives
    for profanity in PROFANITY_SINGLE_WORDS:  # Only check single words here
        # Skip very short profanity words (3 letters or less) for substring matching
        # to avoid false positives (e.g., "ass" in "class" or "pass")
        if len(profanity) <= 3:
            # For short words, only check for exact matches or word boundaries
            if profanity == word_lower:
                return True
            # Check if the profanity is at the beginning with a non-letter following
            elif word_lower.startswith(profanity) and (len(word_lower) == len(profanity) or not word_lower[len(profanity)].isalpha()):
                return True
            # Check if the profanity is at the end with a non-letter preceding
            elif word_lower.endswith(profanity) and (len(word_lower) == len(profanity) or not word_lower[-(len(profanity)+1)].isalpha()):
                return True
            # Check if the profanity is in the middle with non-letters on both sides
            elif f" {profanity} " in f" {word_lower} ":
                return True
        else:
            # For longer profanity words, we can be a bit more lenient
            if profanity in word_lower:
                return True
            
    return False

# Function to find the exact profane word in a phrase
def find_profane_word(phrase):
    """
    Find the specific profane word within a phrase
    
    Args:
        phrase: The phrase to check for profanity
        
    Returns:
        The profane word if found, otherwise None
    """
    if not phrase:
        return None
        
    # Split the phrase into words
    words = phrase.lower().split()
    
    # Check each word for profanity
    for word in words:
        # Clean the word of punctuation
        clean_word = word.strip('.,!?;:"\'()[]{}')
        if clean_word and clean_word in PROFANITY_SINGLE_WORDS:
            return clean_word
            
    # If no exact match, check for partial matches
    for profanity in PROFANITY_SINGLE_WORDS:
        if len(profanity) > 3 and profanity in phrase.lower():
            return profanity
            
    return None

# Function to check if a sequence of words contains a profane phrase
def contains_profane_phrase(words_list):
    """
    Check if a sequence of words contains any profane phrases from our list
    
    Args:
        words_list: List of word dictionaries with "word" keys
        
    Returns:
        Tuple of (found_phrase, start_index, end_index) if a phrase is found, otherwise (None, -1, -1)
    """
    if not words_list:
        return None, -1, -1
    
    # Extract just the words from the dictionaries
    just_words = [word_info["word"].lower().strip() for word_info in words_list]
    
    # Create a string of the words to check against phrases
    words_string = " ".join(just_words)
    
    # Check for each profane phrase
    for phrase in PROFANITY_PHRASES:
        if phrase in words_string:
            # Find where the phrase starts
            phrase_words = phrase.split()
            phrase_word_count = len(phrase_words)
            
            # Sliding window to find the exact match position
            for i in range(len(just_words) - phrase_word_count + 1):
                window = " ".join(just_words[i:i+phrase_word_count])
                if phrase in window:
                    return phrase, i, i + phrase_word_count - 1
    
    return None, -1, -1

# Function to censor a profane word
def censor_profanity(word):
    """
    Censor a profane word by keeping the first and last letters and replacing the middle with asterisks
    
    Args:
        word: The word to censor
        
    Returns:
        Censored word with first letter, asterisks, and last letter
    """
    if not word:
        return word
    
    # Handle case where the word has leading/trailing whitespace
    leading_space = ''
    trailing_space = ''
    
    # Check for leading whitespace
    if word and word[0].isspace():
        leading_space = word[0]
        word = word[1:]
    
    # Check for trailing whitespace
    if word and word[-1].isspace():
        trailing_space = word[-1]
        word = word[:-1]
    
    # If word is too short after removing spaces
    if len(word) <= 2:
        return leading_space + word + trailing_space
    
    if len(word) == 3:
        # For 3-letter words, show first letter and replace the rest
        return leading_space + word[0] + '**' + trailing_space
        
    # For longer words, show first and last letters with asterisks in between
    return leading_space + word[0] + '*' * (len(word) - 2) + word[-1] + trailing_space