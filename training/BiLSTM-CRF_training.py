
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# import os
import time
from tqdm import tqdm
from preprocessing.data_processor import ArabicDiacritizationProcessor
from models.BiLSTM_CRF_Parallel import Arabic_BiLSTM_CRF
from preprocessing.diacritization_dataset import DiacritizationDataset
from features.FastText import FastTextEmbeddings
from Feature_Aligner.FastTextAligner import FastTextFeatureAligner

from utils import collate_fn

# --- 1. Imports your Modules (Assume these are available) ---
# from .model import Arabic_BiLSTM_CRF
# from .data_processor import ArabicDiacritizationProcessor
# from .dataset import DiacritizationDataset, collate_fn, PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN
# from .feature_aligner import FastTextFeatureAligner
# from .fasttext_module import FastTextEmbeddings, EMBEDDING_DIM
# from .metrics import calculate_der # You will need to implement a DER metric function

# --- 2. Configuration Constants (Adjust these based on your data/compute) ---
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256  # Max length for padding in the Dataset
CHAR_EMB_DIM = 128
LSTM_HIDDEN_DIM = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 16
FASTTEXT_DIM = 100    # Ensure this matches your FastTextEmbeddings configuration
PATIENCE = 6 # Example: Stop if Dev DER doesn't improve for 5 epochs
# How often to print per-batch info. Set to 100 to print every 100 batches.
BATCH_PRINT_FREQ = 100
# --- 3. Hardware Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_diacritization_model(train_file: str, dev_file: str, fasttext_model_path: str):

    # --- A. Data and Feature Initialization ---

    # Initialize Processor (for vocab and data splitting)
    processor = ArabicDiacritizationProcessor()

    # Load FastText Model
    fasttext_feature = FastTextEmbeddings(corpus_path="data/cleaned_undiacritized/cleaned_train_data.txt", output_path=fasttext_model_path, dim=FASTTEXT_DIM)
    fasttext_model = fasttext_feature.load_model()

    # Initialize Datasets and Loaders
    # NOTE: The character vocabulary is built from the training set ONLY
    train_dataset = DiacritizationDataset(train_file, processor, MAX_SEQ_LENGTH)
    # The dev dataset uses the character vocabulary created by the train_dataset
    dev_dataset = DiacritizationDataset(dev_file, processor, MAX_SEQ_LENGTH)
    dev_dataset.char_to_id = train_dataset.char_to_id
    dev_dataset.id_to_char = train_dataset.id_to_char

    # Set up DataLoaders with the essential collate_fn
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize FastText Feature Aligner
    aligner = FastTextFeatureAligner(fasttext_model, train_dataset.id_to_char)

    # --- B. Model Initialization ---

    char_vocab_size = len(train_dataset.char_to_id)
    num_tags = len(processor.all_labels)

    model = Arabic_BiLSTM_CRF(
        char_vocab_size=char_vocab_size,
        num_tags=num_tags,
        char_embedding_dim=CHAR_EMB_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        fasttext_embedding_dim=FASTTEXT_DIM,
    ).to(device)

    # Initialize Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_dev_der = float('inf')
    patience_counter = 0 # Initialize patience counter

    # --- C. Training Loop ---
    print(f"\n--- Starting Training on {device} ---")

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # 1. Training Phase
        model.train()
        total_loss = 0

        # TQDM per-batch progress bar. Set miniters to BATCH_PRINT_FREQ to reduce updates frequency.
        miniters_value = max(1, BATCH_PRINT_FREQ)
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)", unit="batch", miniters=miniters_value, total=len(train_dataloader), dynamic_ncols=True)
        # Explicitly set the description in case tqdm doesn't refresh on new instances
        train_bar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)")

        # Precompute number of batches and keep last_loss to print final batch summary
        total_batches = len(train_dataloader)
        last_batch_loss = None

        for batch_idx, batch in enumerate(train_bar):


            # 1.1 Data Transfer
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'] # Keep on CPU for pack_padded_sequence
            words = batch['words']
            # 1.2 FastText Alignment (The only CPU-intensive step in the loop)
            # The output tensor is created on the GPU (device)
            fasttext_vectors = aligner.align_features(input_ids, words)
            fasttext_vectors = fasttext_vectors.to(device)

            # 1.3 Optimization
            optimizer.zero_grad()

            # 1.4 Forward Pass and Loss
            loss = model.neg_log_likelihood(input_ids, labels, lengths, fasttext_vectors)

            # 1.5 Backward Pass and Update
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            last_batch_loss = loss.item()

            # >>>>>> UPDATE TQDM POSTFIX <<<<<<
            # Displays the real-time loss for the current batch
            train_bar.set_postfix({'Loss': f'{last_batch_loss:.4f}'})

            # Batch-level logging: print a new line after every `BATCH_PRINT_FREQ` batches.
            if ((batch_idx + 1) % BATCH_PRINT_FREQ) == 0:
                percent = (batch_idx + 1) / total_batches * 100
                print(f"Train | Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx+1}/{total_batches} ({percent:.1f}%) | Loss: {last_batch_loss:.4f}", flush=True)

        # If we have a remainder of batches, print the last batch's loss as a summary line
        if last_batch_loss is not None and (total_batches % BATCH_PRINT_FREQ) != 0:
            print(f"Train | Epoch {epoch+1}/{NUM_EPOCHS} | Batch {total_batches}/{total_batches} (100.0%) | Loss: {last_batch_loss:.4f}", flush=True)

        avg_train_loss = total_loss / len(train_dataloader)

        # 2. Validation Phase (Model Selection)
        dev_der = evaluate_model(model, dev_dataloader, aligner, device) # Call the evaluation function (defined below)

        # 3. Epoch Summary and Checkpointing
        epoch_time = time.time() - start_time
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} Summary ---", flush=True)
        print(f"Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Dev DER: {dev_der:.4f}", flush=True)

        # Save best model based on Development Error Rate (DER)
        if dev_der < best_dev_der:
            best_dev_der = dev_der
            patience_counter = 0
            print(">>> New best model found! Saving checkpoint.", flush=True)
            torch.save({
            'model_state_dict': model.state_dict(),
            'char_to_id': train_dataset.char_to_id,
            'id_to_label': processor.id_to_label,
            'fasttext_dim': FASTTEXT_DIM,
            'char_emb_dim': CHAR_EMB_DIM,
            'lstm_hidden_dim': LSTM_HIDDEN_DIM,
            'fasttext_vectors': "data/embeddings/fasttext_word_vectors.bin"
        }, 'best_diacritization_model.pth')
        else:
            # No improvement: Increment counter
            patience_counter += 1
            print(f">>> Validation DER did not improve. Patience counter: {patience_counter}/{PATIENCE}", flush=True)

            if patience_counter >= PATIENCE:
                print(f"!!! Early stopping triggered after {PATIENCE} epochs without improvement.", flush=True)
                break # Exit the epoch loop

        # Optional: scheduler.step(dev_der)


def evaluate_model(model: nn.Module, dataloader: DataLoader, aligner: FastTextFeatureAligner, device: torch.device) -> float:
    """
    Evaluates the model on the given dataset and calculates the Diacritic Error Rate (DER).
    """
    model.eval()
    all_true_labels = []
    all_predictions = []

    print("Starting evaluation on dev set...", flush=True)
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths']

            # FastText Alignment
            fasttext_vectors = aligner.align_features(input_ids, batch['words'])
            fasttext_vectors = fasttext_vectors.to(device)

            # Forward Pass (Viterbi Decode)
            predictions = model.forward(input_ids, lengths, fasttext_vectors)

            # Unpack predictions and labels based on sequence length
            for i, length in enumerate(lengths.tolist()):
                # Predictions is List[List[int]], Labels is (B, L) tensor
                all_predictions.extend(predictions[i])
                all_true_labels.extend(labels[i, :length].tolist())

    # Calculate DER (You must implement this function based on the definition: D_w / (D_w + D_c))
    # dev_der = calculate_der(all_true_labels, all_predictions)

    # Placeholder for actual DER calculation
    dev_der = sum([1 for p, t in zip(all_predictions, all_true_labels) if p != t]) / len(all_true_labels)

    print(f"Evaluation complete. Dev DER: {dev_der:.4f}", flush=True)
    return dev_der

# --- Execution Example (Requires all modules to be fully implemented) ---
if __name__ == '__main__':
    # NOTE: These paths must be correctly set up
    TRAIN_FILE = 'data/cleaned_diacritized/cleaned_train_data.txt'
    DEV_FILE = 'data/cleaned_diacritized/cleaned_val_data.txt'
    FASTTEXT_MODEL_PATH = 'data/embeddings/fasttext_word_vectors.bin'

    # 1. Ensure FastText model is trained first!
    # fasttext_trainer = FastTextEmbeddings(corpus_path='data/fasttext_corpus.txt', model_output_path=FASTTEXT_MODEL_PATH)
    # fasttext_trainer.train()

    train_diacritization_model(TRAIN_FILE, DEV_FILE, FASTTEXT_MODEL_PATH)