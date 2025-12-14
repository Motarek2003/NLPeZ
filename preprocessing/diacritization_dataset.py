from preprocessing.data_processor import ArabicDiacritizationProcessor
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os
import re

# Import the processor (assuming it's available or defined in the same scope)
# from .arabic_processor import ArabicDiacritizationProcessor

# Define constants for special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

class DiacritizationDataset(Dataset):
    """
    A PyTorch Dataset class responsible for converting text sequences into
    padded numerical tensors for the BiLSTM-CRF model.
    """
    def __init__(self,
                 file_path: str,
                 processor: 'ArabicDiacritizationProcessor',
                 max_seq_length: int = 256):

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")

        self.file_path = file_path
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.raw_data = self._load_data()

        # Build Vocabularies only based on the training data (when used for train set)
        self.char_to_id, self.id_to_char = self._build_char_vocab(self.raw_data)

        # Diacritic vocabulary is inherited from the processor
        self.label_to_id = self.processor.label_to_id
        self.id_to_label = self.processor.id_to_label

    def _load_data(self) -> List[str]:
        """Reads the diacritized lines from the source file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # Reuses the processor's clean_text to ensure consistency
            return [self.processor.clean_text(line) for line in f if line.strip()]

    def _build_char_vocab(self, raw_data: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Builds the character vocabulary and maps characters to IDs.
        ONLY call this on the training set data.
        """
        all_chars = set()
        for sentence in raw_data:
            # We strip diacritics temporarily to get unique base characters
            undiacritized = self.processor.strip_diacritics(sentence)
            for char in undiacritized:
                # Filter out spaces and punctuation if necessary, or include them
                # For simplicity, we include all characters remaining after cleaning.
                all_chars.add(char)

        # Initialize vocab with special tokens
        char_to_id = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        current_id = len(SPECIAL_TOKENS)

        # Add all unique Arabic characters
        sorted_chars = sorted(list(all_chars))
        for char in sorted_chars:
            if char not in char_to_id:
                char_to_id[char] = current_id
                current_id += 1

        id_to_char = {id: char for char, id in char_to_id.items()}
        print(f"Character Vocabulary Size: {len(char_to_id)}")
        return char_to_id, id_to_char

    def _vectorize_and_pad(self, char_sequence: List[str], diacritic_sequence: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts sequences to IDs and applies padding/truncation.
        """
        char_ids = [self.char_to_id.get(char, self.char_to_id[UNK_TOKEN]) for char in char_sequence]
        label_ids = [self.label_to_id.get(diac, self.label_to_id[self.processor.NO_TASHKEEL]) for diac in diacritic_sequence]

        # 1. Truncate sequences if they exceed MAX_SEQ_LENGTH
        char_ids = char_ids[:self.max_seq_length]
        label_ids = label_ids[:self.max_seq_length]

        # 2. Pad sequences
        padding_needed = self.max_seq_length - len(char_ids)

        char_ids.extend([self.char_to_id[PAD_TOKEN]] * padding_needed)
        label_ids.extend([self.label_to_id[self.processor.NO_TASHKEEL]] * padding_needed)
        # Note: Padding labels with NO_TASHKEEL ID (or PAD ID) is common. Using PAD_ID for labels is often better for masking in loss functions.

        return torch.tensor(char_ids, dtype=torch.long), torch.tensor(label_ids, dtype=torch.long)

    def __len__(self):
        """Returns the total number of sentences/samples in the dataset."""
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves one sample, processes it, and returns the padded tensors.
        """
        sentence = self.raw_data[idx]

        # 1. Split text into undiacritized characters and diacritic labels
        char_seq_diacritized, diac_seq = self.processor.split_char_diacritic(sentence)

        # 2. Get the purely undiacritized character sequence (input)
        # Note: split_char_diacritic returns a list of characters, which are inherently undiacritized here
        char_seq = [self.processor.strip_diacritics(c) for c in char_seq_diacritized] # Redundant step but ensures no lingering diacritics

        # 3. Vectorize and Pad
        input_tensor, target_tensor = self._vectorize_and_pad(char_seq, diac_seq)
        # NEW STEP: Get the actual word sequence (undiacritized)
        words = self.processor.tokenize_to_words(sentence) # Needs to be implemented in processor
        # 4. Prepare Word-Level Feature Placeholder (to be integrated later)
        # For now, we only return the character features and labels
        return {
            'input_ids': input_tensor,
            'labels': target_tensor,
            'lengths': torch.tensor(len(char_seq), dtype=torch.long), # Important for packing sequences
            'words': words
        }