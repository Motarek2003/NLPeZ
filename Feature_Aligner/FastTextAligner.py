import torch
import numpy as np
from typing import List, Dict
import fasttext

from preprocessing.diacritization_dataset import UNK_TOKEN, PAD_TOKEN


class FastTextFeatureAligner:
    """
    Aligns word-level FastText embeddings to character-level inputs
    by reconstructing words from character IDs.
    This preserves maximum semantic information (best DER).
    """

    def __init__(
        self,
        raw_fasttext_model: "fasttext.FastText",
        id_to_char_vocab: Dict[int, str],
        space_char: str = " ",
    ):
        if id_to_char_vocab is None:
            raise ValueError("id_to_char_vocab must not be None")

        self.fasttext_model = raw_fasttext_model
        self.id_to_char_vocab = id_to_char_vocab
        self.space_char = space_char

        # Infer FastText dimension safely
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
        """
        Args:
            char_ids_batch: (B, L) LongTensor
            lengths: (B,) LongTensor

        Returns:
            aligned_features: (B, L, fasttext_dim)
        """

        cpu_ids = char_ids_batch.detach().cpu()
        lengths = lengths.detach().cpu()

        batch_size, max_len = cpu_ids.shape
        aligned = torch.zeros(
            (batch_size, max_len, self.fasttext_dim),
            dtype=torch.float32,
        )

        for b in range(batch_size):
            seq_len = lengths[b].item()
            char_ids = cpu_ids[b, :seq_len].tolist()

            # 1️⃣ Reconstruct words
            words = self._reconstruct_words(char_ids)

            char_pos = 0
            for word in words:
                if not word:
                    continue

                # 2️⃣ Get FastText word vector
                try:
                    word_vec = self.fasttext_model.get_word_vector(word)
                except Exception:
                    word_vec = self.unk_word_vector

                # 3️⃣ Broadcast to characters
                assigned = 0
                target_len = len(word)

                while char_pos < seq_len and assigned < target_len:
                    cid = char_ids[char_pos]
                    ch = self.id_to_char_vocab.get(cid, UNK_TOKEN)

                    # Skip spaces explicitly
                    if ch == self.space_char:
                        char_pos += 1
                        continue

                    aligned[b, char_pos] = torch.from_numpy(word_vec)
                    assigned += 1
                    char_pos += 1

                # Move past any spaces after the word
                while char_pos < seq_len:
                    ch = self.id_to_char_vocab.get(char_ids[char_pos], "")
                    if ch == self.space_char:
                        char_pos += 1
                    else:
                        break

        return aligned
