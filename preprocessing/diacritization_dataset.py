from preprocessing.data_processor import ArabicDiacritizationProcessor
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os

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

    def _vectorize_and_pad(
        self,
        char_sequence: List[str],
        diacritic_sequence: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

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

        char_ids = char_ids[: self.max_seq_length]
        label_ids = label_ids[: self.max_seq_length]

        padding = self.max_seq_length - len(char_ids)

        char_ids.extend([self.char_to_id[PAD_TOKEN]] * padding)
        label_ids.extend(
            [self.label_to_id[self.processor.NO_TASHKEEL]] * padding
        )

        return (
            torch.tensor(char_ids, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long),
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

        input_ids, labels = self._vectorize_and_pad(
            char_seq, diac_seq
        )
        actual_len = min(len(char_seq), self.max_seq_length)

        item = {
            "input_ids": input_ids,
            "labels": labels,
            "lengths": torch.tensor(actual_len, dtype=torch.long),
        }

        # 🔴 RETURN CACHED FASTTEXT
        if self.fasttext_aligner is not None:
            item["fasttext"] = self.cached_fasttext[idx]

        return item
