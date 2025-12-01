from interfaces.feature import Feature

import fasttext
import os

def create_sample_data():
    # Sample sentences for training
    sentences = [
        "The king rules the kingdom",
        "The queen helps the king",
        "Running is good exercise",
        "The runner runs fast",
        "Walking is healthy activity",
        "The walker walks slowly",
        "Reading books is fun",
        "The reader reads daily"
    ]

    # Save to text file (one sentence per line)
    with open('training_data.txt', 'w') as f:
        for sentence in sentences:
            f.write(sentence.lower() + '\n')  # Convert to lowercase

    print("Training data created in 'training_data.txt'")

create_sample_data()

def create_classification_data():
    reviews = [
        ("This movie is amazing and fun", "positive"),
        ("Great acting and story", "positive"),
        ("Excellent film with good plot", "positive"),
        ("Wonderful cinematography", "positive"),
        ("Terrible movie very boring", "negative"),
        ("Bad acting and poor story", "negative"),
        ("Worst film ever made", "negative"),
        ("Boring and predictable plot", "negative")
    ]

    with open('movie_reviews.txt', 'w') as f:
        for text, label in reviews:
            f.write(f"__label__{label} {text.lower()}\n")

    print("Classification data created in 'movie_reviews.txt'")

create_classification_data()
class FastText(Feature):
    def __init__(self):
        super().__init__()

    def train(self):
        # Train skipgram model (predicts context from target word)
        model = fasttext.train_unsupervised(
            'training_data.txt',    # Input file
            model='skipgram',
            dim=50,                 # Embedding dimension
            epoch=10,               # Number of training iterations
            minCount=1,             # Minimum word frequency
            minn=3,                 # Minimum character n-gram length
            maxn=6                  # Maximum character n-gram length
        )

        model.save_model('word_vectors.bin')
        print("Model trained and saved as 'word_vectors.bin'")
        return model

    def get_word_embeddings(self, model):
        king_vector = model.get_word_vector('king')
        print(f"Vector for 'king': {king_vector[:5]}...")
        print(f"Vector shape: {king_vector.shape}")

        kingdom_vector = model.get_word_vector('kingdom')
        print(f"Vector for 'kingdom' (OOV): {kingdom_vector[:5]}...")

        return king_vector, kingdom_vector

    def find_similar_words(self, model, word, k=3):
        print(f"\nWords similar to '{word}':")
        try:
            neighbors = model.get_nearest_neighbors(word, k)
            for i, (similarity, similar_word) in enumerate(neighbors, 1):
                print(f"{i}. {similar_word}: {similarity:.4f}")
        except Exception as e:
            print(f"Error: {e}")
    def train_text_classifier(self):
        classifier = fasttext.train_supervised(
            'movie_reviews.txt',
            epoch=25,
            lr=0.1,
            wordNgrams=2,
            verbose=2
        )

        classifier.save_model('text_classifier.bin')
        print("Classifier trained and saved")
        return classifier

    def test_classifier(self, classifier):
        test_sentences = [
            "This is a fantastic movie",
            "Boring and terrible film",
            "Great story and acting",
            "Worst movie I have seen"
        ]

        print("\nClassification Results:")
        print("-" * 40)

        for sentence in test_sentences:
            labels, probabilities = classifier.predict(sentence, k=1)
            predicted_label = labels[0].replace('__label__', '')
            confidence = probabilities[0]
            print(f"Text: '{sentence}'")
            print(f"Prediction: {predicted_label} (confidence: {confidence:.4f})\n")


fast1 = FastText()
model = fast1.train()
king_vec, kingdom_vec = fast1.get_word_embeddings(model)
fast1.find_similar_words(model, 'king')
fast1.find_similar_words(model, 'running')
classifier = fast1.train_text_classifier()
fast1.test_classifier(classifier)

