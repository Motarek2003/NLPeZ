import re
import sys
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import pyarabic.araby as araby
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
import sentencepiece as spm
from preprocessing.data_cleaning import ArabicCleaner

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import constants
from utils.constants import (
    ARABIC_PUNCTUATION,
    WHITESPACE_PATTERN,
    ARABIC_LETTERS_STR,
    EXTENDED_DIACRITICS,
)


# =============================================================================
# SENTENCE TOKENIZATION
# =============================================================================

class SentenceTokenizer:
    """
    Tokenize Arabic text into sentences.
    
    Uses Arabic punctuation from constants.py for sentence boundaries.
    """
    
    def __init__(self):
        # Sentence terminators: use all Arabic punctuation from constants
        sentence_ends = ARABIC_PUNCTUATION
        self.pattern = re.compile(
            f'([{re.escape(sentence_ends)}.,;:!?()\\[\\]{{}}\'\"«»]+)',
            re.UNICODE
        )
    
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input Arabic text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Split on sentence terminators but keep them
        parts = self.pattern.split(text)
        
        # Recombine sentences with their terminators
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i].strip()
            if i + 1 < len(parts):
                sentence += parts[i + 1]
            if sentence:
                sentences.append(sentence.strip())
        
        # Handle last part if no terminator
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())
        
        return sentences


# =============================================================================
# WORD TOKENIZATION
# =============================================================================

class WordTokenizer:
    """
    Tokenize Arabic text into words.
    
    Splits on whitespace and punctuation using constants.py definitions.
    """
    
    def __init__(self, keep_punctuation: bool = True):
        """
        Args:
            keep_punctuation: Whether to keep punctuation as separate tokens
        """
        self.keep_punctuation = keep_punctuation
        # Build pattern to split on punctuation and whitespace
        punct_pattern = f'[{re.escape(ARABIC_PUNCTUATION)}.,;:!?()\\[\\]{{}}\'\"«»]'
        self.split_pattern = re.compile(f'({punct_pattern}+|\\s+)', re.UNICODE)
        self.whitespace_pattern = re.compile(WHITESPACE_PATTERN, re.UNICODE)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into words.
        
        Args:
            text: Input Arabic text
            
        Returns:
            List of words
        """
        if not text:
            return []
        
        # Split on punctuation and whitespace
        tokens = self.split_pattern.split(text)
        
        # Filter and clean
        words = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            
            # Skip pure whitespace
            if self.whitespace_pattern.match(token):
                continue
            
            # Keep or skip punctuation
            if self.keep_punctuation:
                words.append(token)
            else:
                # Skip if token is only punctuation
                if not all(c in ARABIC_PUNCTUATION + '.,;:!?()[]{}\'\"«»' for c in token):
                    words.append(token)
        
        return words


# =============================================================================
# CHARACTER TOKENIZATION
# =============================================================================

class CharacterTokenizer:
    """
    Tokenize Arabic text into characters.
    
    Uses EXTENDED_DIACRITICS from constants.py to handle diacritics properly.
    """
    
    def __init__(self, keep_diacritics_separate: bool = False):
        """
        Args:
            keep_diacritics_separate: If True, diacritics are separate tokens
        """
        self.keep_diacritics_separate = keep_diacritics_separate
    
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into characters.
        
        Args:
            text: Input Arabic text
            
        Returns:
            List of characters
        """
        if not text:
            return []
        
        if self.keep_diacritics_separate:
            # Each character is a separate token
            return list(text)
        else:
            # Group base character with following diacritics using constants
            chars = []
            i = 0
            while i < len(text):
                char = text[i]
                i += 1
                
                # Collect following diacritics from EXTENDED_DIACRITICS
                while i < len(text) and text[i] in EXTENDED_DIACRITICS:
                    char += text[i]
                    i += 1
                
                chars.append(char)
            
            return chars


# =============================================================================
# BPE TOKENIZATION
# =============================================================================

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer using HuggingFace tokenizers.
    """
    
    def __init__(self, vocab_size: int = 10000):
        """
        Args:
            vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        self.tokenizer = None
    
    def train(self, texts: List[str], save_path: Optional[str] = None):
        """
        Train BPE tokenizer on corpus.
        
        Args:
            texts: Training corpus
            save_path: Optional path to save trained tokenizer
        """
        # Initialize BPE model
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        
        # Configure normalizer - REMOVE StripAccents to keep Arabic diacritics
        self.tokenizer.normalizer = normalizers.NFD()
        
        # Pre-tokenizer splits on whitespace
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # Train with special tokens
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
            show_progress=True,
            min_frequency=2  # Only include tokens that appear at least twice
        )
        
        # Filter empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            raise ValueError("No valid training texts provided")
        
        self.tokenizer.train_from_iterator(valid_texts, trainer=trainer)
        
        if save_path:
            self.tokenizer.save(save_path)
            print(f"✓ BPE tokenizer saved to {save_path}")
    
    def load(self, path: str):
        """Load trained tokenizer from file."""
        self.tokenizer = Tokenizer.from_file(path)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using BPE.
        
        Args:
            text: Input text (single string, not list)
            
        Returns:
            List of subword tokens
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded. Call train() or load() first.")
        
        if not text or not text.strip():
            return []
        
        encoding = self.tokenizer.encode(text)
        return encoding.tokens


# =============================================================================
# WORDPIECE TOKENIZATION
# =============================================================================

class WordPieceTokenizer:
    """
    WordPiece tokenizer (BERT-style) using HuggingFace tokenizers.
    """
    
    def __init__(self, vocab_size: int = 10000):
        """
        Args:
            vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        self.tokenizer = None
    
    def train(self, texts: List[str], save_path: Optional[str] = None):
        """
        Train WordPiece tokenizer on corpus.
        
        Args:
            texts: Training corpus
            save_path: Optional path to save trained tokenizer
        """
        # Initialize WordPiece model
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        
        # Configure normalizer 
        self.tokenizer.normalizer = normalizers.NFD()
        
        # Pre-tokenizer
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # Train with special tokens
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
            show_progress=True,
            min_frequency=2
        )
        
        # Filter empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            raise ValueError("No valid training texts provided")
        
        self.tokenizer.train_from_iterator(valid_texts, trainer=trainer)
        
        if save_path:
            self.tokenizer.save(save_path)
            print(f"✓ WordPiece tokenizer saved to {save_path}")
    
    def load(self, path: str):
        """Load trained tokenizer from file."""
        self.tokenizer = Tokenizer.from_file(path)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using WordPiece.
        
        Args:
            text: Input text
            
        Returns:
            List of subword tokens
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded. Call train() or load() first.")
        
        if not text or not text.strip():
            return []
        
        encoding = self.tokenizer.encode(text)
        return encoding.tokens


# =============================================================================
# SENTENCEPIECE TOKENIZATION
# =============================================================================

class SentencePieceTokenizer:
    """
    SentencePiece tokenizer for unsupervised subword tokenization.
    """
    
    def __init__(self, vocab_size: int = 10000, model_type: str = "unigram"):
        """
        Args:
            vocab_size: Target vocabulary size
            model_type: "unigram", "bpe", "char", or "word"
        """
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.sp = None
    
    def train(
        self, 
        input_files: List[str], 
        model_prefix: str,
        character_coverage: float = 0.9995
    ):
        """
        Train SentencePiece model.
        
        Args:
            input_files: List of training file paths
            model_prefix: Output model name prefix
            character_coverage: Character coverage for vocab
        """
        # Join multiple files
        input_str = ','.join(input_files)
        
        # Train
        spm.SentencePieceTrainer.train(
            input=input_str,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            character_coverage=character_coverage,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )
        
        # Load trained model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{model_prefix}.model")
    
    def load(self, model_path: str):
        """Load trained SentencePiece model."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using SentencePiece.
        
        Args:
            text: Input text
            
        Returns:
            List of subword tokens
        """
        if self.sp is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")
        
        return self.sp.encode_as_pieces(text)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to IDs."""
        if self.sp is None:
            raise RuntimeError("Model not trained or loaded.")
        return self.sp.encode_as_ids(text)
    
    def decode(self, ids: List[int]) -> str:
        """Decode IDs back to text."""
        if self.sp is None:
            raise RuntimeError("Model not trained or loaded.")
        return self.sp.decode_ids(ids)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ARABIC TOKENIZATION TEST")
    print("=" * 70)
    
    # Read from file train.txt
    print("\n[1/6] Reading train.txt...")
    with open('train.txt', 'r', encoding='utf-8') as file:
        real_text = file.read()
    
    print(f"✓ Read {len(real_text)} characters")
    
    # Clean the text
    print("\n[2/6] Cleaning text...")
    from preprocessing.data_cleaning import DiacritizationCleaner
    cleaner = DiacritizationCleaner()
    real_text = cleaner.clean(real_text)
    print(f"✓ Cleaned text: {len(real_text)} characters")
    
    # normalize the text
    print("\n[3/6] Normalizing text...")
    from preprocessing.data_normalization import DiacritizationNormalizer
    normalizer = DiacritizationNormalizer()
    real_text = normalizer.normalize(real_text)
    
    # Sentence Tokenization
    print("\n[4/6] Tokenizing into sentences...")
    sentence_tokenizer = SentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(real_text)
    print(f"✓ Found {len(sentences)} sentences")
    print(f"  Sample: {sentences[0][:80]}..." if sentences else "  No sentences found")
    
    # word Tokenization
    word_tokenizer = WordTokenizer(keep_punctuation=False)
    words = []
    for sentence in sentences:
        words.extend(word_tokenizer.tokenize(sentence))
    print(f"✓ Found {len(words)} words")
    print(f"  Sample: {words[:10]}...")

    # Character Tokenization
    char_tokenizer = CharacterTokenizer(keep_diacritics_separate=True)
    characters = []
    for word in words:
        characters.extend(char_tokenizer.tokenize(word))
    print(f"✓ Found {len(characters)} characters")
    print(f"  Sample: {characters[:20]}...")

    # BPE Tokenization
    print("\n[5/6] Training BPE tokenizer...")
    bpe_tokenizer = BPETokenizer(vocab_size=8000)  # Increased vocab size
    bpe_tokenizer.train(sentences, save_path="arabic_bpe.json")
    
    print("\n[6/6] Testing BPE tokenization...")
    print("-" * 70)
    for i, sentence in enumerate(sentences[:5], 1):
        tokens = bpe_tokenizer.tokenize(sentence)
        print(f"\n[{i}] Sentence: {sentence[:60]}...")
        print(f"    Tokens ({len(tokens)}): {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
    
    # Test with sample text
    print("\n" + "=" * 70)
    print("SAMPLE TEXT TEST")
    print("=" * 70)
    
    sample_text = """
    قَالَ رَسُولُ اللهِ صلى الله عليه وسلم: إِنَّمَا الأَعْمَالُ بِالنِّيَّاتِ.
    وَإِنَّمَا لِكُلِّ امْرِئٍ مَا نَوَى. فَمَنْ كَانَتْ هِجْرَتُهُ إِلَى اللهِ
    وَرَسُولِهِ فَهِجْرَتُهُ إِلَى اللهِ وَرَسُولِهِ. وَمَنْ كَانَتْ هِجْرَتُهُ
    لِدُنْيَا يُصِيبُهَا أَوِ امْرَأَةٍ يَنْكِحُهَا فَهِجْرَتُهُ إِلَى مَا هَاجَرَ إِلَيْهِ
    (رواه البخاري ومسلم).

    وَهُوَ ظَاهِرُ مَا جَزَمَ بِهِ فِي الْمُنَوِّرِ ، وَمُنْتَخَبِ الْأَدَمِيِّ .( 16 / 36 )
وَلَوْ حُمِلَ مِنْ مَجْلِس الْخِيَارِ ، وَلَمْ يُمْنَعْ مِنْ الْكَلَامِ .
[ مُحَارَبَةُ قُصَيّ لِخُزَاعَةَ وَبَنِيّ بَكْرٍ وَتَحْكِيمُ يَعْمُرَ بْنِ عَوْفٍ ]
كَبَيَّاعِ الْخُبْزِ وَشَبَهِهِ يُدْفَعُ إلَيْهِ الْخَاتَمُ وَنَحْوُهُ وَيَدَّعِي الرَّهْنِيَّةَ فَالْقَوْلُ قَوْلُهُ وَلَا يُقْبَلُ قَوْلُ صَاحِبِهِ أَنَّهُ وَدِيعَةٌ .
( وَمِنْهَا الْعَدَالَةُ حَالَ الْأَدَاءِ وَإِنْ تَحَمَّلَ فَاسِقًا إلَّا بِفِسْقٍ ) تَعَمَّدَ ( الْكَذِبَ عَلَيْهِ عَلَيْهِ السَّلَامُ عِنْدَ أَحْمَدَ وَطَائِفَةٍ ) كَأَبِي بَكْرٍ الْحُمَيْدِيِّ شَيْخُ الْبُخَارِيِّ وَالصَّيْرَفِيُّ فَإِنَّهُ عِنْدَهُمْ يُوجِبُ مَنْعَ قَبُولِ رِوَايَتِهِ أَبَدًا وَكَأَنَّهُ لَمَّا صَحَّ عَنْهُ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ أَنَّهُ قَالَ { إنَّ كَذِبًا عَلَيَّ لَيْسَ كَكَذِبٍ عَلَى أَحَدٍ مَنْ كَذَبَ عَلَيَّ مُتَعَمِّدًا فَلْيَتَبَوَّأْ مَقْعَدَهُ مِنْ النَّارِ } وَهُوَ ثَابِتٌ بِالتَّوَاتُرِ كَمَا ذَكَرَهُ ابْنُ الصَّلَاحِ وَلِمَا فِيهِ مِنْ عِظَمِ الْمَفْسَدَةِ لِأَنَّهُ يَصِيرُ شَرْعًا مُسْتَمِرًّا إلَى يَوْمِ الْقِيَامَةِ حَتَّى ذَهَبَ أَبُو مُحَمَّدٍ الْجُوَيْنِيُّ وَالِدُ إمَامِ الْحَرَمَيْنِ إلَى أَنَّهُ يَكْفُرُ وَيُرَاقُ دَمُهُ لَكِنْ ضَعَّفَهُ وَلَدُهُ وَعَدَّهُ مِنْ هَفَوَاتِهِ وَقَالَ الذَّهَبِيُّ ذَهَبَ طَائِفَةٌ مِنْ الْعُلَمَاءِ إلَى أَنَّ الْكَذِبَ عَلَى رَسُولِ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ كُفْرٌ يَنْقُلُ عَنْ الْمِلَّةِ ثُمَّ قَالَ وَلَا رَيْبَ أَنَّ تَعَمُّدَ الْكَذِبِ عَلَى اللَّهِ وَرَسُولِهِ فِي تَحْلِيلِ حَرَامٍ أَوْ تَحْرِيمِ حَلَالٍ كُفْرٌ مَحْضٌ ( وَالْوَجْهُ الْجَوَازُ ) لِرِوَايَتِهِ وَشَهَادَتِهِ ( بَعْدَ ثُبُوتِ الْعَدَالَةِ ) لِأَنَّهُ كَمَا قَالَ النَّوَوِيُّ الْمُخْتَارُ الْقَطْعُ بِصِحَّةِ تَوْبَتِهِ مِنْ ذَلِكَ وَقَبُولِ رِوَايَتِهِ بَعْدَ صِحَّةِ التَّوْبَةِ بِشُرُوطِهَا وَقَدْ أَجْمَعُوا عَلَى قَبُولِ رِوَايَةِ مَنْ كَانَ كَافِرًا ثُمَّ أَسْلَمَ وَعَلَى قَبُولِ شَهَادَتِهِ وَلَا فَرْقَ بَيْنَ الرِّوَايَةِ وَالشَّهَادَةِ ( وَهِيَ ) أَيْ الْعَدَالَةُ ( مَلَكَةٌ ) أَيْ هَيْئَةٌ رَاسِخَةٌ فِي النَّفْسِ ( تَحْمِلُ عَلَى مُلَازَمَةِ التَّقْوَى ) أَيْ اجْتِنَابِ الْكَبَائِرِ لِأَنَّ الصَّغَائِرَ مُكَفَّرَةٌ بِاجْتِنَابِهَا لِقَوْلِهِ تَعَالَى { إنْ تَجْتَنِبُوا كَبَائِرَ مَا تُنْهَوْنَ عَنْهُ نُكَفِّرْ عَنْكُمْ سَيِّئَاتِكُمْ } ( وَالْمُرُوءَةِ ) بِالْهَمْزِ وَيَجُوزُ تَرْكُهُ مَعَ تَشْدِيدِ الْوَاوِ وَهِيَ صِيَانَةُ النَّفْسِ عَنْ الْأَدْنَاسِ وَمَا يَشِينُهَا عِنْدَ النَّاسِ وَقِيلَ( 4 / 107 )
( قَوْلُهُ : مِنْ بَلْدَةٍ ) الْبَلْدَةُ بُيُوتٌ كَثِيرَةٌ مُجْتَمِعَةٌ بِحَيْثُ تُسَمَّى بَلْدَةً وَاحِدَةً ، وَالْقَرْيَةُ بُيُوتٌ قَلِيلَةٌ كَذَلِكَ .

    """
    
    sample_text = cleaner.clean(sample_text)
    sample_text = normalizer.normalize(sample_text)
    tokens = bpe_tokenizer.tokenize(sample_text)
    print(f"\nSample: {sample_text}")
    print(f"Tokens: {tokens}")
    
    print("\n" + "=" * 70)
    print("✓ Tokenization test completed successfully!")
    print("=" * 70)

