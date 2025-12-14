import sys
import os
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import shutil

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
BiLSTM_CRF_training.FASTTEXT_DIM = 100 # Keep small for test
BiLSTM_CRF_training.device = torch.device("cpu") # Force CPU for testing

# Setup test data directory
TEST_DATA_DIR = os.path.abspath("tests/data")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Patch PROJECT_ROOT to point to test data so cache goes there
BiLSTM_CRF_training.PROJECT_ROOT = TEST_DATA_DIR

# Create the expected cache directory structure
os.makedirs(os.path.join(TEST_DATA_DIR, "data", "cache"), exist_ok=True)
os.makedirs(os.path.join(TEST_DATA_DIR, "training", "outputs"), exist_ok=True)

train_file = os.path.join(TEST_DATA_DIR, "train.txt")
val_file = os.path.join(TEST_DATA_DIR, "val.txt")
fasttext_corpus = os.path.join(TEST_DATA_DIR, "corpus.txt")

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
    
    # Clean up previous cache if any
    cache_dir = os.path.join(TEST_DATA_DIR, "data", "cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # Patch get_or_train to return our mock model
    with patch("features.FastText.FastTextEmbeddings.get_or_train", return_value=mock_ft_model):
        try:
            print("Running training (1st run - should create cache)...")
            # Run training
            BiLSTM_CRF_training.train_diacritization_model(
                train_file=train_file,
                dev_file=val_file,
                fasttext_model_path=os.path.join(TEST_DATA_DIR, "fasttext.bin"),
                fasttext_corpus_path=fasttext_corpus
            )
            
            # Check if cache files exist
            if os.path.exists(os.path.join(cache_dir, "train_dataset.pt")):
                print("SUCCESS: Cache file created.")
            else:
                print("FAILURE: Cache file NOT created.")
                return

            print("Running training (2nd run - should use cache)...")
            # Run training again to test cache loading
            BiLSTM_CRF_training.train_diacritization_model(
                train_file=train_file,
                dev_file=val_file,
                fasttext_model_path=os.path.join(TEST_DATA_DIR, "fasttext.bin"),
                fasttext_corpus_path=fasttext_corpus
            )
            print("Test finished successfully!")
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_training()
