import sys
import os
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple
from pyarabic import araby

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.constants import (
    ALEF_VARIANTS,
    ALEF_MAKSURA,
    YEH,
    TEH_MARBUTA,
    HEH,
    TATWEEL,
    EXTENDED_DIACRITICS,
)



# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NormalizationConfig:
    """Normalization settings."""
    normalize_alef: bool = True          # أ، إ، آ → ا (usually False for diacritization)
    normalize_alef_maksura: bool = True   # ى → ي
    normalize_teh_marbuta: bool = False   # ة → ه (usually False for diacritization)
    normalize_hamza: bool = True         # ؤ، ئ → ء
    remove_tatweel: bool = True           # Remove ـ
    unicode_nfc: bool = True              # Unicode normalization


# =============================================================================
# ARABIC NORMALIZER
# =============================================================================

class ArabicNormalizer:
    """
    Arabic text normalizer using PyArabic.
    
    Example:
            normalizer = ArabicNormalizer()
             normalizer.normalize("مَـــرْحَبًا")
        'مَرْحَبًا'
    """
    
    def __init__(self, config: NormalizationConfig = None):
        self.config = config or NormalizationConfig()
    
    def normalize(self, text: str) -> str:
        """Normalize Arabic text."""
        if not text:
            return ""
        
        # Unicode NFC normalization
        if self.config.unicode_nfc:
            text = unicodedata.normalize('NFC', text)
        
        # Remove Tatweel
        if self.config.remove_tatweel:
            text = araby.strip_tatweel(text)
        
        # Normalize Alef variants (أ، إ، آ → ا)
        if self.config.normalize_alef:
           for variant, standard in ALEF_VARIANTS.items():
               text = text.replace(variant, standard)
        
        # Normalize Alef Maksura (ى → ي)
        if self.config.normalize_alef_maksura:
            text = text.replace(ALEF_MAKSURA, YEH)
        
        # Normalize Teh Marbuta (ة → ه)
        if self.config.normalize_teh_marbuta:
            text = text.replace(TEH_MARBUTA, HEH)
        
        # Normalize Hamza (ؤ، ئ → ء)
        if self.config.normalize_hamza:
            text = text.replace('ؤ', 'ء').replace('ئ', 'ء')
        
        return text
    
    def normalize_batch(self, texts: List[str]) -> List[str]:
        """Normalize multiple texts."""
        return [self.normalize(t) for t in texts]
    
    def strip_diacritics(self, text: str) -> str:
        """Remove all diacritics from text."""
        return araby.strip_tashkeel(text)


# =============================================================================
# DIACRITIZATION NORMALIZER (Conservative settings)
# =============================================================================

class DiacritizationNormalizer(ArabicNormalizer):
    """
    Normalizer with conservative settings for diacritization.
    
    Preserves:
    - Alef variants (أ، إ، آ) - important for correct diacritization
    - Teh Marbuta (ة) - different from Heh
    - Hamza forms (ؤ، ئ)
    
    Normalizes:
    - Alef Maksura (ى → ي) - same letter
    - Tatweel removal (ـ) - no meaning
    """
    
    def __init__(self):
        # For diacritization we MUST preserve Alef variants (أ، إ، آ)
        # but normalize Alef Maksura (ى) to Yeh (ي).
        config = NormalizationConfig(
            normalize_alef=False,            # keep أ/إ/آ
            normalize_alef_maksura=True,     # ى -> ي
            normalize_teh_marbuta=False,     # keep ة
            normalize_hamza=False,           # keep ؤ/ئ
            remove_tatweel=True,
        )
        super().__init__(config)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def normalize_for_diacritization(text: str) -> str:
    """Quick normalization for diacritization tasks."""
    return DiacritizationNormalizer().normalize(text)


def strip_diacritics(text: str) -> str:
    """Remove all Arabic diacritics."""
    return araby.strip_tashkeel(text)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Arabic Normalization Test")
    print("=" * 50)
    
    normalizer = DiacritizationNormalizer()
    
    tests = [
        ("مَـــرْحَبًا", "Tatweel removal"),
        ("مُوسَى", "Alef Maksura → Yeh"),
        ("أَحْمَدُ", "Alef with Hamza"),
        ("الْمَدْرَسَةُ", "Teh Marbuta"),
        ("سَلَامٌ عَلَيْكُمْ", "Diacritized text"),
        ('قِرْآنٌ كَرِيمٌ', "Diacritized text with Alef"),
        ("مُؤْمِنٌ", "Hamza on Waw and Yeh"),
        ("ئ", "Hamza on Yeh (keep)"),
    ]
    
    for text, desc in tests:
        result = normalizer.normalize(text)
        print(f"{desc}:")
        print(f"  Input:  {text}")
        print(f"  Output: {result}")
        print(f" Tashkeel Stripped: {strip_diacritics(result)}")
