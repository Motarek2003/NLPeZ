from enum import Enum

class ArabicDiactrics(Enum):
    """
    All possible diacritic labels for classification.
    Each Arabic letter can have one of these diacritics.
    """
    NONE = 0           # No diacritic
    FATHA = 1          # َ (a sound)
    FATHATAN = 2       # ً (an sound)
    DAMMA = 3          # ُ (u sound)
    DAMMATAN = 4       # ٌ (un sound)
    KASRA = 5          # ِ (i sound)
    KASRATAN = 6       # ٍ (in sound)
    SUKUN = 7          # ْ (no vowel)
    SHADDA = 8         # ّ (double consonant)
    SHADDA_FATHA = 9   # ّ + َ
    SHADDA_FATHATAN = 10
    SHADDA_DAMMA = 11
    SHADDA_DAMMATAN = 12
    SHADDA_KASRA = 13
    SHADDA_KASRATAN = 14


NUM_DIACRITICS = len(list(ArabicDiactrics))  # 15 classes


# =============================================================================
# MODEL TYPE ENUM
# =============================================================================

class ModelType(Enum):
    """Supported model architectures."""
    ARABERT = "arabert"
    CAMELBERT = "camelbert"
    BILSTM = "bilstm"
