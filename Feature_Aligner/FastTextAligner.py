import torch
import numpy as np
from typing import List, Dict
import fasttext

from preprocessing.diacritization_dataset import UNK_TOKEN, PAD_TOKEN

# Assuming necessary classes/constants are imported from your project:
# from your_processor_module import ArabicDiacritizationProcessor
# from your_fasttext_module import FastTextEmbeddings
# from your_dataset_module import UNK_TOKEN, PAD_TOKEN

# Placeholder Vocabularies (Must be provided by the DiacritizationDataset instance)
# EXAMPLE: {1: 'ا', 2: 'ب', 3: ' ', 0: '<PAD>', 1: '<UNK>', ...}
# We use this to convert IDs back to characters/words
ID_TO_CHAR: Dict[int, str] = {} # Must be set externally

class FastTextFeatureAligner:
    """
    Manages the lookup and alignment of FastText word vectors to the character-level
    input tensor for a given batch.
    """
    def __init__(self,
                 raw_fasttext_model: 'fasttext.FastText', # Accepting the raw fasttext model
                 id_to_char_vocab: Dict[int, str]):

        self.fasttext_model = raw_fasttext_model
        self.id_to_char_vocab = id_to_char_vocab

        # INFER THE DIMENSION: Get the vector for a known word (e.g., 'UNK') and check its shape.
        # This is a robust way to get the dimension from the loaded model object.
        # Use a non-existent word to safely get the UNK vector.
        try:
            ft_vec = self.fasttext_model.get_word_vector('__dummy_word__')
            self.fasttext_dim = ft_vec.shape[0]
        except Exception as e:
             raise ValueError(f"Could not infer FastText dimension from the loaded model. Error: {e}")
        # Get IDs for special tokens
        # Assuming PAD_TOKEN ID is 0 and space ID is needed for word splitting
        self.pad_id = 0
        self.space_char = ' ' # Assuming space is retained in your character vocab

        if self.space_char not in self.id_to_char_vocab.values():
             print("Warning: Space character not found in char vocabulary. Word reconstruction may be inaccurate.")

        # Initialize an UNK vector for words not found by FastText (highly rare due to n-grams)
        self.unk_word_vector = np.zeros(self.fasttext_dim, dtype=np.float32)

    def _ids_to_words(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> List[List[str]]:
        """
        Converts a batch of character ID tensors back into a list of words per sequence.
        """
        batch_words: List[List[str]] = []

        for i, length in enumerate(lengths):
            sequence_ids = input_ids[i, :length].tolist()
            sequence_chars = [self.id_to_char_vocab.get(idx, UNK_TOKEN) for idx in sequence_ids]

            # Reconstruct the sentence string and split into words
            sentence_str = "".join(sequence_chars)
            # Remove any residual <UNK> tokens and split by space
            words = [word.strip() for word in sentence_str.split(self.space_char) if word.strip()]
            batch_words.append(words)

        return batch_words

    def align_features(self, char_ids_batch: torch.Tensor, words_batch: List[List[str]]) -> torch.Tensor:
        """
        Calculates the aligned FastText vector for every character in the batch.
        The function skips space/PAD characters when aligning so each word's characters
        receive the correct word vector even if char-level sequence retains spacing.
        """
        # Work on CPU for indexing safe calls (we'll move tensor to device later in training/eval)
        cpu_char_ids = char_ids_batch.detach().cpu()
        batch_size, max_seq_len = cpu_char_ids.shape
        aligned_feature_batch = torch.zeros(
            (batch_size, max_seq_len, self.fasttext_dim), dtype=torch.float32
        )

        for batch_idx, word_list in enumerate(words_batch):
            char_pos = 0

            for word in word_list:
                if not word:
                    continue

                # 1. Lookup the Word Vector (Handles OOV gracefully)
                word_vector = self.fasttext_model.get_word_vector(word)

                # 2. Assign the word vector to the next N non-space/non-pad chars
                assigned = 0
                target_len = len(word)

                while assigned < target_len and char_pos < max_seq_len:
                    char_id = int(cpu_char_ids[batch_idx, char_pos].item())
                    char_str = self.id_to_char_vocab.get(char_id, UNK_TOKEN)

                    # Skip PAD or explicit whitespace characters
                    if char_str == PAD_TOKEN or (isinstance(char_str, str) and char_str.strip() == ""):
                        char_pos += 1
                        continue

                    aligned_feature_batch[batch_idx, char_pos, :] = torch.tensor(
                        word_vector, dtype=torch.float32
                    )
                    assigned += 1
                    char_pos += 1

            # No explicit increment for trailing spaces - leave them zero (UNK)

        return aligned_feature_batch