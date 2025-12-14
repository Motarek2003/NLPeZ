import sys
import os
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock fasttext module BEFORE importing anything that uses it
sys.modules["fasttext"] = MagicMock()

# Import the module to test
from training import BiLSTM_CRF_training

# Override constants for testing to make it fast
BiLSTM_CRF_training.NUM_EPOCHS = 1
BiLSTM_CRF_training.BATCH_SIZE = 2
BiLSTM_CRF_training.BATCH_PRINT_FREQ = 1
BiLSTM_CRF_training.device = torch.device("cpu") # Force CPU for testing

# Create dummy data
os.makedirs("tests/data", exist_ok=True)
train_file = "tests/data/train.txt"
val_file = "tests/data/val.txt"
fasttext_corpus = "tests/data/corpus.txt"

dummy_text = """
وَلَوْ جَمَعَ ثُمَّ عَلِمَ تَرْكَ رُكْنٍ مِنْ الْأُولَى بَطَلَتَا
قَالَ أَبُو زَيْدٍ أَهْلُ تِهَامَةَ يُؤَنِّثُونَ الْعَضُدَ
""".strip()

with open(train_file, "w", encoding="utf-8") as f:
    f.write(dummy_text)

with open(val_file, "w", encoding="utf-8") as f:
    f.write(dummy_text)

with open(fasttext_corpus, "w", encoding="utf-8") as f:
    f.write(dummy_text)

# Mock FastText model object
mock_ft_model = MagicMock()
# get_word_vector should return a numpy array of shape (100,)
mock_ft_model.get_word_vector.side_effect = lambda w: np.zeros(100, dtype=np.float32)
mock_ft_model.get_dimension.return_value = 100

def test_training():
    print("Starting test...")
    
    # Patch get_or_train to return our mock model
    with patch("features.FastText.FastTextEmbeddings.get_or_train", return_value=mock_ft_model):
        try:
            # Run training
            BiLSTM_CRF_training.train_diacritization_model(
                train_file=train_file,
                dev_file=val_file,
                fasttext_model_path="tests/data/fasttext.bin",
                fasttext_corpus_path=fasttext_corpus
            )
            print("Test finished successfully!")
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_training()
