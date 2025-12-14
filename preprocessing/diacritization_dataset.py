from preprocessing.data_processor import ArabicDiacritizationProcessor
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os
from tqdm import tqdm

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class DiacritizationDataset(Dataset):
    """
    Dataset with cached FastText features for fast training.
    """

    def __init__(
        self,
        file_path: str,
        processor: "ArabicDiacritizationProcessor",
        max_seq_length: int = 256,
        fasttext_aligner=None,   # 🔴 NEW
    ):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")

        self.file_path = file_path
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.fasttext_aligner = fasttext_aligner

        self.raw_data = self._load_data()

        # Build vocab only from training data
        self.char_to_id, self.id_to_char = self._build_char_vocab(self.raw_data)

        self.label_to_id = self.processor.label_to_id
        self.id_to_label = self.processor.id_to_label

        # Initialize POS Tagger
        self.pos_pipeline = None
        self._init_pos_tagger()

        # Cache POS tags
        self.cached_pos_tags = self._cache_pos_tags()

        # Build POS Vocab
        self.pos_to_id, self.id_to_pos = self._build_pos_vocab(self.cached_pos_tags)

        # 🔴 CACHE FASTTEXT VECTORS ONCE
        self.cached_fasttext = []
        if self.fasttext_aligner is not None:
            print("Caching FastText features...")
            for sentence in self.raw_data:
                words = self.processor.tokenize_to_words(sentence)
                vecs = self.fasttext_aligner.align_sentence(
                    words,
                    max_len=self.max_seq_length
                )
                self.cached_fasttext.append(vecs)
            print("FastText caching complete.")

    def _load_data(self) -> List[str]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            return [
                self.processor.clean_text(line)
                for line in f
                if line.strip()
            ]

    def _build_char_vocab(
        self, raw_data: List[str]
    ) -> Tuple[Dict[str, int], Dict[int, str]]:

        all_chars = set()
        for sentence in raw_data:
            undiacritized = self.processor.strip_diacritics(sentence)
            for char in undiacritized:
                all_chars.add(char)

        char_to_id = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        current_id = len(SPECIAL_TOKENS)

        for char in sorted(all_chars):
            if char not in char_to_id:
                char_to_id[char] = current_id
                current_id += 1

        id_to_char = {i: c for c, i in char_to_id.items()}
        print(f"Character Vocabulary Size: {len(char_to_id)}")
        return char_to_id, id_to_char

    def _init_pos_tagger(self):
        try:
            from transformers import pipeline
            print("Initializing POS tagger (CAMeLBERT)...")
            self.pos_pipeline = pipeline(
                "token-classification",
                model="CAMeL-Lab/bert-base-arabic-camelbert-msa-pos",
                aggregation_strategy="simple", # Groups subwords
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"Warning: Could not initialize POS tagger: {e}. Using dummy tags.")
            self.pos_pipeline = None

    def _cache_pos_tags(self) -> List[List[str]]:
        cached_tags = []
        print("Caching POS tags...")
        
        if self.pos_pipeline:
            # Process in batches for speed
            batch_size = 32
            for i in tqdm(range(0, len(self.raw_data), batch_size), desc="POS Tagging"):
                batch_sentences = self.raw_data[i : i + batch_size]
                # Strip diacritics for BERT
                batch_clean = [self.processor.strip_diacritics(s) for s in batch_sentences]
                
                try:
                    results = self.pos_pipeline(batch_clean)
                    
                    for j, res in enumerate(results):
                        # res is a list of entities: [{'entity_group': 'NOUN', 'word': '...', ...}]
                        # We need to map these to our words.
                        # Since we used aggregation_strategy="simple", 'word' should be the full word (mostly).
                        
                        # Simple alignment: Just take the tags in order.
                        # If count mismatches, pad or truncate.
                        
                        original_words = self.processor.tokenize_to_words(batch_sentences[j])
                        tags = [entity['entity_group'] for entity in res]
                        
                        # Force alignment
                        if len(tags) < len(original_words):
                            tags.extend(["NOUN"] * (len(original_words) - len(tags)))
                        elif len(tags) > len(original_words):
                            tags = tags[:len(original_words)]
                            
                        cached_tags.append(tags)
                        
                except Exception as e:
                    # Fallback for batch failure
                    for sent in batch_sentences:
                        words = self.processor.tokenize_to_words(sent)
                        cached_tags.append(["NOUN"] * len(words))
        else:
            # Dummy fallback
            for sentence in self.raw_data:
                words = self.processor.tokenize_to_words(sentence)
                cached_tags.append(["NOUN"] * len(words))
                
        return cached_tags

    def _build_pos_vocab(self, all_tags: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
        unique_tags = set()
        for tags in all_tags:
            unique_tags.update(tags)
            
        # Ensure special tokens
        pos_to_id = {"<PAD>": 0, "OTHER": 1}
        current_id = 2
        
        for tag in sorted(unique_tags):
            if tag not in pos_to_id:
                pos_to_id[tag] = current_id
                current_id += 1
                
        id_to_pos = {v: k for k, v in pos_to_id.items()}
        print(f"POS Vocabulary Size: {len(pos_to_id)}")
        return pos_to_id, id_to_pos

    def _get_pos_tags(self, idx: int) -> List[str]:
        return self.cached_pos_tags[idx]

    def _align_pos_tags(self, char_seq: List[str], words: List[str], pos_tags: List[str]) -> List[str]:
        aligned_pos = []
        word_idx = 0
        char_in_word_idx = 0
        
        for char in char_seq:
            if char.strip() == "": # Space or invisible
                 aligned_pos.append("OTHER")
                 continue
            
            if word_idx < len(words):
                # Safety check for index
                tag = pos_tags[word_idx] if word_idx < len(pos_tags) else "OTHER"
                aligned_pos.append(tag)
                
                char_in_word_idx += 1
                
                current_word = words[word_idx]
                # Simple length check - this assumes perfect tokenization match which is rare
                # But for character alignment, we just need to know when to switch word.
                # Since we stripped diacritics for both, lengths should match roughly.
                
                # Better logic:
                # If we consumed all chars of current word, move to next.
                if char_in_word_idx >= len(current_word):
                    word_idx += 1
                    char_in_word_idx = 0
            else:
                aligned_pos.append("OTHER")
        return aligned_pos


    def _vectorize_and_pad(
        self,
        char_sequence: List[str],
        diacritic_sequence: List[str],
        pos_sequence: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        char_ids = [
            self.char_to_id.get(c, self.char_to_id[UNK_TOKEN])
            for c in char_sequence
        ]

        label_ids = [
            self.label_to_id.get(
                d, self.label_to_id[self.processor.NO_TASHKEEL]
            )
            for d in diacritic_sequence
        ]

        pos_ids = [
            self.pos_to_id.get(p, self.pos_to_id["OTHER"])
            for p in pos_sequence
        ]

        char_ids = char_ids[: self.max_seq_length]
        label_ids = label_ids[: self.max_seq_length]
        pos_ids = pos_ids[: self.max_seq_length]

        padding = self.max_seq_length - len(char_ids)

        char_ids.extend([self.char_to_id[PAD_TOKEN]] * padding)
        label_ids.extend(
            [self.label_to_id[self.processor.NO_TASHKEEL]] * padding
        )
        pos_ids.extend([self.pos_to_id["<PAD>"]] * padding)

        return (
            torch.tensor(char_ids, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long),
            torch.tensor(pos_ids, dtype=torch.long),
        )

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sentence = self.raw_data[idx]

        char_seq_diac, diac_seq = self.processor.split_char_diacritic(
            sentence
        )

        char_seq = [
            self.processor.strip_diacritics(c)
            for c in char_seq_diac
        ]

        # POS Tagging
        words = self.processor.tokenize_to_words(sentence)
        word_pos_tags = self._get_pos_tags(idx)
        pos_seq = self._align_pos_tags(char_seq, words, word_pos_tags)

        input_ids, labels, pos_ids = self._vectorize_and_pad(
            char_seq, diac_seq, pos_seq
        )
        actual_len = min(len(char_seq), self.max_seq_length)

        item = {
            "input_ids": input_ids,
            "labels": labels,
            "pos_ids": pos_ids,
            "lengths": torch.tensor(actual_len, dtype=torch.long),
        }

        # 🔴 RETURN CACHED FASTTEXT
        if self.fasttext_aligner is not None:
            item["fasttext"] = self.cached_fasttext[idx]

        return item
