import torch
import numpy as np
from typing import List, Dict
# import fasttext # Removed dependency

from preprocessing.diacritization_dataset import UNK_TOKEN, PAD_TOKEN


class FastTextFeatureAligner:
    """
    Aligns word-level FastText embeddings to character-level inputs
    by reconstructing words from character IDs.
    This preserves maximum semantic information (best DER).
    """

    def __init__(
        self,
        raw_fasttext_model, # Type hint removed to support both libs
        id_to_char_vocab: Dict[int, str],
        space_char: str = " ",
    ):
        if id_to_char_vocab is None:
            raise ValueError("id_to_char_vocab must not be None")

        self.fasttext_model = raw_fasttext_model
        self.id_to_char_vocab = id_to_char_vocab
        self.space_char = space_char

        # Infer FastText dimension safely
        if hasattr(self.fasttext_model, 'wv'):
             # Gensim
             self.fasttext_dim = self.fasttext_model.wv.vector_size
        else:
             # Fallback (original fasttext or mock)
             dummy_vec = self.fasttext_model.get_word_vector("__dummy__")
             self.fasttext_dim = dummy_vec.shape[0]

        # PAD id (assumed)
        self.pad_id = next(
            (i for i, c in id_to_char_vocab.items() if c == PAD_TOKEN),
            0,
        )

        # UNK vector (FastText almost never OOV, but safe)
        self.unk_word_vector = np.zeros(self.fasttext_dim, dtype=np.float32)

        if self.space_char not in self.id_to_char_vocab.values():
            print(
                "[Warning] Space character not found in vocabulary. "
                "Word reconstruction may be imperfect."
            )

    # --------------------------------------------------
    # INTERNAL: reconstruct words from char IDs
    # --------------------------------------------------
    def _reconstruct_words(
        self, char_ids: List[int]
    ) -> List[str]:
        chars = [
            self.id_to_char_vocab.get(cid, UNK_TOKEN)
            for cid in char_ids
            if cid != self.pad_id
        ]

        sentence = "".join(chars)
        words = [
            w for w in sentence.split(self.space_char) if w.strip()
        ]
        return words

    # --------------------------------------------------
    # MAIN ALIGNMENT FUNCTION
    # --------------------------------------------------
    def align_features(
        self,
        char_ids_batch: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:

        cpu_ids = char_ids_batch.detach().cpu()
        lengths = lengths.detach().cpu()

        batch_size, max_len = cpu_ids.shape
        aligned = torch.zeros(
            (batch_size, max_len, self.fasttext_dim),
            dtype=torch.float32,
        )

        for b in range(batch_size):
            seq_len = int(lengths[b].item())
            char_ids = cpu_ids[b, :seq_len].tolist()

            # Reconstruct words
            words = self._reconstruct_words(char_ids)

            char_pos = 0
            char_ids_len = len(char_ids)

            for word in words:
                if not word:
                    continue

                # Get FastText vector
                try:
                    if hasattr(self.fasttext_model, 'wv'):
                        # Gensim
                        word_vec = self.fasttext_model.wv[word]
                    else:
                        # Fallback
                        word_vec = self.fasttext_model.get_word_vector(word)
                except Exception:
                    word_vec = self.unk_word_vector

                target_len = len(word)
                assigned = 0

                while char_pos < char_ids_len and assigned < target_len:
                    cid = char_ids[char_pos]
                    ch = self.id_to_char_vocab.get(cid, "")

                    # Skip spaces explicitly
                    if ch == self.space_char:
                        char_pos += 1
                        continue

                    aligned[b, char_pos] = torch.from_numpy(word_vec)
                    assigned += 1
                    char_pos += 1

                # Skip any remaining spaces safely
                while char_pos < char_ids_len:
                    cid = char_ids[char_pos]
                    ch = self.id_to_char_vocab.get(cid, "")
                    if ch == self.space_char:
                        char_pos += 1
                    else:
                        break

        return aligned
