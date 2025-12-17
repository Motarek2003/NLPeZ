import gensim
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
#import os
from typing import Optional
#from interfaces.feature import Feature # Assuming Feature is a base class for engineering features
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from interfaces.feature import Feature

# Set a standard dimension that is typical for a language model (e.g., 100 or 200)
EMBEDDING_DIM = 300
class FastTextEmbeddings(Feature):
    """
    A dedicated class for training and loading FastText word embeddings
    for the Arabic Diacritization task using Gensim.
    """
    def __init__(self,
                 corpus_path: str,
                 output_path: str = 'data/embeddings/fasttext_word_vectors.model',
                 dim: int = EMBEDDING_DIM): # Increased dimension for Arabic

        super().__init__()
        self.corpus_path = str(Path(corpus_path).resolve())
        self.output_path = str(Path(output_path).resolve())
        self.dim = dim
        self.model = None

    def get_or_train(self):
        """
        Loads a FastText model if it exists, otherwise trains and saves it.
        """
        if os.path.exists(self.output_path):
            print(f"Loading existing FastText model from {self.output_path}")
            try:
                return self.load_model()
            except Exception as e:
                # Common failure: numpy/gensim pickle incompatibility across environments
                print(f"Warning: failed to load existing FastText model: {e}")
                print("Attempting to retrain FastText model in current environment...")
                try:
                    # remove corrupt/ incompatible model file and retrain
                    os.remove(self.output_path)
                except Exception:
                    pass
                return self.train()
        else:
            print(f"No FastText model found. Training new model...")
            return self.train()


    def train(self) -> FastText:
        """
        Trains the unsupervised FastText skipgram model on the Arabic corpus.
            """
            # ✅ ensure output directory exists (CRITICAL)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        if os.path.exists(self.output_path):
            print(f"FastText model already exists at '{self.output_path}'. Skipping training.")
            return self.load_model()

        print(f"Starting FastText training on {self.corpus_path}...")

        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found at: {self.corpus_path}")

        # Gensim FastText training
        self.model = FastText(
            vector_size=self.dim,
            window=5,
            min_count=5,
            sentences=LineSentence(self.corpus_path),
            epochs=10,
            sg=1, # skipgram
            min_n=3,
            max_n=6
        )

        self.model.save(self.output_path)
        print(f"FastText model trained and saved at {self.output_path}")
        return self.model

    def load_model(self):
        """Loads a pre-trained model."""
        if os.path.exists(self.output_path):
            self.model = FastText.load(self.output_path)
            return self.model
        else:
            raise FileNotFoundError(f"Model file not found at: {self.output_path}. Please train first.")

    def get_word_vector(self, word: str):
        """Retrieves the vector for a given word using the trained model."""
        if self.model is None:
            self.load_model()
        # Gensim uses .wv for word vectors
        return self.model.wv[word]

# --- Usage Flow ---
# 1. Ensure your cleaned, undiacritized training data is saved to a file, e.g., 'arabic_train_words.txt'
#    (This file should contain one word or sentence per line).
# 2. Instantiate and train (guarded):
if __name__ == '__main__':
    ARABIC_CORPUS_PATH = 'data/cleaned_undiacritized/cleaned_train_data.txt'
    fasttext_feature = FastTextEmbeddings(corpus_path=ARABIC_CORPUS_PATH)
    trained_model = fasttext_feature.train()