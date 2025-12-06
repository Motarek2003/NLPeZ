import re
import sys
import os
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import pyarabic.araby as araby

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import constants from utils
from utils.constants import (
    # Character sets
    ARABIC_LETTERS_STR,
    EXTENDED_DIACRITICS,
    ARABIC_PUNCTUATION,
    TATWEEL,
    # Patterns
    ANNOTATION_PATTERNS_REMOVE,
    SPECIAL_SYMBOLS,
    HTML_TAG_PATTERN,
    URL_PATTERN,
    EMAIL_PATTERN,
    ENGLISH_PATTERN,
    ALL_DIGITS_PATTERN,
    WHITESPACE_PATTERN,
    # Diacritics
    ArabicDiacritics,
    UNICODE_TO_DIACRITIC,
)


# =============================================================================
# CLEANING CONFIGURATION
# =============================================================================

@dataclass
class CleaningConfig:
    """
    Configuration for Arabic text cleaning.
    """
    preserve_diacritics: bool = True
    remove_tatweel: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_numbers: bool = True
    remove_english: bool = True
    remove_punctuation: bool = False  # Keep for sentence structure
    remove_extra_whitespace: bool = True
    remove_special_symbols: bool = True
    remove_annotations: bool = True
    min_arabic_ratio: float = 0.5
    min_length: int = 1
    max_length: int = 0  # 0 = no limit


# =============================================================================
# ARABIC CLEANER CLASS
# =============================================================================

class ArabicCleaner:
    """
    General-purpose Arabic text cleaner using PyArabic.
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient reuse."""
        self.html_pattern = re.compile(HTML_TAG_PATTERN, re.UNICODE)
        self.url_pattern = re.compile(URL_PATTERN, re.IGNORECASE | re.UNICODE)
        self.email_pattern = re.compile(EMAIL_PATTERN, re.IGNORECASE)
        self.number_pattern = re.compile(ALL_DIGITS_PATTERN)
        self.english_pattern = re.compile(ENGLISH_PATTERN)
        
        punct_chars = ARABIC_PUNCTUATION + r'.,;:!?\'"()[\]{}<>«»\-_/\\|@#$%^&*+=~`'
        self.punctuation_pattern = re.compile(f'[{re.escape(punct_chars)}]+')
        
        self.annotation_patterns = [re.compile(p, re.UNICODE) for p in ANNOTATION_PATTERNS_REMOVE]
        self.control_char_pattern = re.compile(r'[\x00-\x1f\x7f-\x9f]+')
        self.zero_width_pattern = re.compile(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]+')
    
    def clean(self, text: str) -> str:
        """
        Apply all cleaning operations based on configuration.
        """
        if not text:
            return ""
        
        # 1. Structural cleaning
        if self.config.remove_html:
            text = self.html_pattern.sub(' ', text)
        if self.config.remove_urls:
            text = self.url_pattern.sub(' ', text)
        if self.config.remove_emails:
            text = self.email_pattern.sub(' ', text)
        
        # 2. Arabic specific cleaning (PyArabic)
        if self.config.remove_tatweel:
            text = araby.strip_tatweel(text)
        
        if not self.config.preserve_diacritics:
            text = araby.strip_tashkeel(text)
            
        # 3. Content filtering
        if self.config.remove_special_symbols:
            for symbol, replacement in SPECIAL_SYMBOLS.items():
                text = text.replace(symbol, replacement)
        
        if self.config.remove_annotations:
            for pattern in self.annotation_patterns:
                text = pattern.sub(' ', text)
        
        if self.config.remove_numbers:
            text = self.number_pattern.sub(' ', text)
        
        if self.config.remove_english:
            text = self.english_pattern.sub(' ', text)
        
        if self.config.remove_punctuation:
            text = self.punctuation_pattern.sub(' ', text)
            
        # 4. Cleanup
        text = self.control_char_pattern.sub(' ', text)
        text = self.zero_width_pattern.sub('', text)
        
        if self.config.remove_extra_whitespace:
            text = ' '.join(text.split())
            
        return text.strip()

    def calculate_arabic_ratio(self, text: str) -> float:
        if not text: return 0.0
        arabic_count = sum(1 for c in text if c in ARABIC_LETTERS_STR)
        total_chars = sum(1 for c in text if not c.isspace())
        return arabic_count / total_chars if total_chars > 0 else 0.0

    def is_valid_arabic_text(self, text: str) -> bool:
        if len(text) < self.config.min_length: return False
        if self.config.max_length > 0 and len(text) > self.config.max_length: return False
        if self.calculate_arabic_ratio(text) < self.config.min_arabic_ratio: return False
        return True


# =============================================================================
# DIACRITIZATION CLEANER CLASS
# =============================================================================

class DiacritizationCleaner(ArabicCleaner):
    """
    Specialized cleaner for Arabic diacritization tasks.
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        if config is None:
            config = CleaningConfig(
                preserve_diacritics=True,
                remove_tatweel=True,
                remove_html=True,
                remove_urls=True,
                remove_emails=True,
                remove_numbers=True,
                remove_english=True,
                remove_punctuation=False,
                remove_extra_whitespace=True,
            )
        super().__init__(config)
    
    def separate_diacritics(self, text: str) -> Tuple[str, List[str]]:
        """
        Separate base characters from their diacritics.
        """
        base_chars = []
        diacritics_list = []
        
        i = 0
        while i < len(text):
            char = text[i]
            
            if char in EXTENDED_DIACRITICS:
                # Skip standalone diacritics at start or double diacritics not handled below
                i += 1
                continue
                
            base_chars.append(char)
            i += 1
            
            # Collect following diacritics
            current_diacritics = ""
            while i < len(text) and text[i] in EXTENDED_DIACRITICS:
                current_diacritics += text[i]
                i += 1
            
            diacritics_list.append(current_diacritics)
            
        return ''.join(base_chars), diacritics_list
    
    def diacritic_to_label(self, diacritic: str) -> int:
        """Convert diacritic string to label ID."""
        if not diacritic:
            return ArabicDiacritics.NONE.value
        if diacritic in UNICODE_TO_DIACRITIC:
            return UNICODE_TO_DIACRITIC[diacritic].value
        return ArabicDiacritics.NONE.value
    
    def extract_labels(self, text: str) -> Tuple[str, List[int]]:
        """Extract base text and label IDs."""
        base_text, diacritics = self.separate_diacritics(text)
        labels = [self.diacritic_to_label(d) for d in diacritics]
        return base_text, labels
    
    def get_diacritization_stats(self, text: str) -> Dict:
        """Calculate diacritization statistics."""
        base_text, diacritics_list = self.separate_diacritics(text)
        total_chars = len(base_text)
        diacritized_chars = sum(1 for d in diacritics_list if d)
        
        return {
            'total_chars': total_chars,
            'diacritized_chars': diacritized_chars,
            'ratio': diacritized_chars / total_chars if total_chars > 0 else 0.0
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ARABIC DATA CLEANING TEST (PyArabic Powered)")
    print("=" * 70)
    
    sample_text = "قَوْلُهُ تَعَالَى: تُطَهِّرُهُمْ ( 7 / 401 )"
    print(f"Original: {sample_text}")

    cleaner = DiacritizationCleaner()
    cleaned = cleaner.clean(sample_text)
    print(f"Cleaned:  {cleaned}")
    
    base, labels = cleaner.extract_labels(cleaned)
    print(f"Base:     {base}")
    print(f"Labels:   {labels}")
    
    print("\nStats:", cleaner.get_diacritization_stats(cleaned))
