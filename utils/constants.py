from enum import Enum


# =============================================================================
# DIACRITICS ENUM
# =============================================================================

class ArabicDiacritics(Enum):
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


NUM_DIACRITICS = len(list(ArabicDiacritics))  # 15 classes


# =============================================================================
# ARABIC CHARACTER SETS
# =============================================================================

# Arabic Letters (base characters)
ARABIC_LETTERS = [
    'ء',  # Hamza
    'آ',  # Alef with Madda
    'أ',  # Alef with Hamza above
    'ؤ',  # Waw with Hamza
    'إ',  # Alef with Hamza below
    'ئ',  # Yeh with Hamza
    'ا',  # Alef
    'ب',  # Beh
    'ة',  # Teh Marbuta
    'ت',  # Teh
    'ث',  # Theh
    'ج',  # Jeem
    'ح',  # Hah
    'خ',  # Khah
    'د',  # Dal
    'ذ',  # Thal
    'ر',  # Reh
    'ز',  # Zain
    'س',  # Seen
    'ش',  # Sheen
    'ص',  # Sad
    'ض',  # Dad
    'ط',  # Tah
    'ظ',  # Zah
    'ع',  # Ain
    'غ',  # Ghain
    'ف',  # Feh
    'ق',  # Qaf
    'ك',  # Kaf
    'ل',  # Lam
    'م',  # Meem
    'ن',  # Noon
    'ه',  # Heh
    'و',  # Waw
    'ى',  # Alef Maksura
    'ي',  # Yeh
]

# Arabic letters as string (for regex and membership checks)
ARABIC_LETTERS_STR = ''.join(ARABIC_LETTERS)


# =============================================================================
# DIACRITICS MAPPING
# =============================================================================

# Single diacritics (Unicode characters)
FATHA = '\u064E'      # َ
DAMMA = '\u064F'      # ُ
KASRA = '\u0650'      # ِ
FATHATAN = '\u064B'   # ً
DAMMATAN = '\u064C'   # ٌ
KASRATAN = '\u064D'   # ٍ
SUKUN = '\u0652'      # ْ
SHADDA = '\u0651'     # ّ

# Extended diacritics
MADDAH = '\u0653'           # ٓ
HAMZA_ABOVE = '\u0654'      # ٔ
HAMZA_BELOW = '\u0655'      # ٕ
SUBSCRIPT_ALEF = '\u0656'   # ٖ
SUPERSCRIPT_ALEF = '\u0670' # ٰ

# All core diacritics as string
CORE_DIACRITICS = FATHA + DAMMA + KASRA + FATHATAN + DAMMATAN + KASRATAN + SUKUN + SHADDA

# Extended diacritics string
EXTENDED_DIACRITICS = CORE_DIACRITICS + MADDAH + HAMZA_ABOVE + HAMZA_BELOW + SUBSCRIPT_ALEF + SUPERSCRIPT_ALEF

# Enum to Unicode mapping
DIACRITIC_TO_UNICODE = {
    ArabicDiacritics.NONE: '',
    ArabicDiacritics.FATHA: FATHA,
    ArabicDiacritics.FATHATAN: FATHATAN,
    ArabicDiacritics.DAMMA: DAMMA,
    ArabicDiacritics.DAMMATAN: DAMMATAN,
    ArabicDiacritics.KASRA: KASRA,
    ArabicDiacritics.KASRATAN: KASRATAN,
    ArabicDiacritics.SUKUN: SUKUN,
    ArabicDiacritics.SHADDA: SHADDA,
    ArabicDiacritics.SHADDA_FATHA: SHADDA + FATHA,
    ArabicDiacritics.SHADDA_FATHATAN: SHADDA + FATHATAN,
    ArabicDiacritics.SHADDA_DAMMA: SHADDA + DAMMA,
    ArabicDiacritics.SHADDA_DAMMATAN: SHADDA + DAMMATAN,
    ArabicDiacritics.SHADDA_KASRA: SHADDA + KASRA,
    ArabicDiacritics.SHADDA_KASRATAN: SHADDA + KASRATAN,
}

# Unicode to Enum mapping (reverse lookup)
UNICODE_TO_DIACRITIC = {v: k for k, v in DIACRITIC_TO_UNICODE.items() if v}


# =============================================================================
# NORMALIZATION CHARACTERS
# =============================================================================

# Alef variants (all normalize to bare Alef 'ا')
ALEF_VARIANTS = {
    'أ': 'ا',  # Alef with Hamza above
    'إ': 'ا',  # Alef with Hamza below
    'آ': 'ا',  # Alef with Madda
    'ٱ': 'ا',  # Alef Wasla
}

# Alef Maksura to Yeh (they are variants of the same letter)
ALEF_MAKSURA = 'ى'
YEH = 'ي'

# Teh Marbuta and Heh (usually NOT normalized for diacritization)
TEH_MARBUTA = 'ة'
HEH = 'ه'

# Tatweel (Kashida) - elongation character (always remove)
TATWEEL = '\u0640'  # ـ

# Arabic-Indic digits
ARABIC_INDIC_DIGITS = '٠١٢٣٤٥٦٧٨٩'
WESTERN_DIGITS = '0123456789'

# Arabic punctuation
ARABIC_PUNCTUATION = '،؛؟.«»ـ!‘’“”…()[]{}'


# =============================================================================
# NOISE PATTERNS FOR CLEANING
# =============================================================================

# Patterns that should ALWAYS be removed (non-linguistic content)
ANNOTATION_PATTERNS_REMOVE = [
    r'\(\s*\d+\s*\)',              # (123) - page numbers
    r'\(\s*\d+\s*/\s*\d+\s*\)',    # (1/234) - volume/page
    r'\[\s*\d+\s*\]',              # [123] - footnote numbers
    r'\(\s*ش\s*\)',                # (ش) - editorial mark
    r'\(\s*م\s*\d*\s*\)',          # (م) or (م1) - editorial mark
    r'\d+\s*[-–—]\s*',             # 123 - numbered references
]

# Special symbols to remove or replace
SPECIAL_SYMBOLS = {
    'ﷺ': '',           # PBUH symbol - remove (or replace with phrase)
    'ﷻ': '',           # Jalla Jalaluhu - remove
    '﷽': '',           # Bismillah - remove (or keep as phrase)
}

# Patterns that are OPTIONAL to remove (valid Arabic but may be repetitive)
# Use these only if your training data has too many of these phrases
ANNOTATION_PATTERNS_OPTIONAL = [
    r'\(\s*قَوْلُهُ\s*:',           # (قوله: - annotation start
    r'\(\s*قوله\s*:',              # (قوله: - without diacritics
    r'\[\s*قوله\s*:',              # [قوله: - annotation start
]

# Religious phrases - KEEP THESE (they are valid diacritized Arabic)
# Only listed here for documentation purposes
RELIGIOUS_PHRASES_KEEP = [
    'صَلَّى اللهُ عَلَيْهِ وَسَلَّمَ',    # PBUH (diacritized)
    'صلى الله عليه وسلم',              # PBUH (undiacritized)
    'رَضِيَ اللهُ عَنْهُ',              # May Allah be pleased with him
    'رضي الله عنه',
    'رَحِمَهُ اللهُ',                   # May Allah have mercy on him
    'رحمه الله',
    'عَزَّ وَجَلَّ',                    # Mighty and Majestic
    'سُبْحَانَهُ وَتَعَالَى',            # Glorified and Exalted
]


# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Compiled patterns for efficiency (use re.compile() in actual code)
ARABIC_LETTER_PATTERN = r'[ء-ي]'
ARABIC_LETTER_EXTENDED_PATTERN = rf'[{ARABIC_LETTERS_STR}]'
DIACRITICS_PATTERN = rf'[{CORE_DIACRITICS}]+'
EXTENDED_DIACRITICS_PATTERN = rf'[{EXTENDED_DIACRITICS}]+'
ARABIC_INDIC_DIGIT_PATTERN = rf'[{ARABIC_INDIC_DIGITS}]'
WESTERN_DIGIT_PATTERN = r'[0-9]'
ALL_DIGITS_PATTERN = rf'[0-9{ARABIC_INDIC_DIGITS}]+'
TATWEEL_PATTERN = rf'{TATWEEL}+'
WHITESPACE_PATTERN = r' +'


# Web patterns
HTML_TAG_PATTERN = r'<[^>]+>'
URL_PATTERN = r'(?:https?://|www\.|ftp://)[^\s<>"{}|\\^`\[\]]+'
EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# English/Latin pattern
ENGLISH_PATTERN = r'[a-zA-Z]+'

# Punctuation patterns
ARABIC_PUNCTUATION_PATTERN = rf'[{ARABIC_PUNCTUATION}]'
WESTERN_PUNCTUATION_PATTERN = r'[.,;:!?\'"()\[\]{}<>«»\-_/\\|@#$%^&*+=~`]'
ALL_PUNCTUATION_PATTERN = rf'[{ARABIC_PUNCTUATION}.,;:!?\'"()\[\]{{}}<>«»\-_/\\|@#$%^&*+=~`]'


# =============================================================================
# SPECIAL TOKENS (for tokenization)
# =============================================================================

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
"""
MASK_TOKEN = '<MASK>'
SEP_TOKEN = '<SEP>'
CLS_TOKEN = '<CLS>'
SPACE_TOKEN = '<SPACE>'
"""
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
# Token IDs
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


# =============================================================================
# MODEL DEFAULTS
# =============================================================================

DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_EMBEDDING_DIM = 256
DEFAULT_HIDDEN_DIM = 512
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.1
DEFAULT_LEARNING_RATE = 1e-4

