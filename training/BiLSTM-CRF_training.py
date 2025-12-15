import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocessing.data_processor import ArabicDiacritizationProcessor
from preprocessing.diacritization_dataset import DiacritizationDataset
from models.BiLSTM_CRF_Parallel import Arabic_BiLSTM_CRF
from features.FastText import FastTextEmbeddings
from Feature_Aligner.FastTextAligner import FastTextFeatureAligner
from utils import collate_fn

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256
CHAR_EMB_DIM = 128
LSTM_HIDDEN_DIM = 256
FASTTEXT_DIM = 100
LEARNING_RATE = 1e-4
NUM_EPOCHS = 32
PATIENCE = 6
BATCH_PRINT_FREQ = 100

FASTTEXT_MODEL_PATH = "/kaggle/working/embeddings/fasttext_word_vectors.bin"
BEST_CKPT_PATH = "/kaggle/working/best_diacritization_model.pth"
EPOCH_CKPT_DIR = "/kaggle/working/checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# TRAINING FUNCTION
# ------------------------------------------------------------------
def train_diacritization_model(train_file, dev_file):

    # --------------------------------------------------------------
    # Prepare output directories (ONCE)
    # --------------------------------------------------------------
    os.makedirs(os.path.dirname(FASTTEXT_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(BEST_CKPT_PATH), exist_ok=True)
    os.makedirs(EPOCH_CKPT_DIR, exist_ok=True)

    processor = ArabicDiacritizationProcessor()

    # --------------------------------------------------------------
    # Load FastText (NEVER retrains if exists)
    # --------------------------------------------------------------
    fasttext_feature = FastTextEmbeddings(
        corpus_path="data/undiacritized/traincu_data.txt",
        output_path=FASTTEXT_MODEL_PATH,
        dim=FASTTEXT_DIM
    )
    fasttext_model = fasttext_feature.get_or_train()

    # --------------------------------------------------------------
    # Build TRAIN dataset first (for vocab)
    # --------------------------------------------------------------
    train_dataset = DiacritizationDataset(
        train_file,
        processor,
        MAX_SEQ_LENGTH
    )

    # --------------------------------------------------------------
    # Create FastText aligner (needs vocab)
    # --------------------------------------------------------------
    aligner = FastTextFeatureAligner(
        fasttext_model,
        train_dataset.id_to_char
    )

    # --------------------------------------------------------------
    # Build DEV dataset (share vocab)
    # --------------------------------------------------------------
    dev_dataset = DiacritizationDataset(
        dev_file,
        processor,
        MAX_SEQ_LENGTH
    )
    dev_dataset.char_to_id = train_dataset.char_to_id
    dev_dataset.id_to_char = train_dataset.id_to_char

    # --------------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------
    model = Arabic_BiLSTM_CRF(
        char_vocab_size=len(train_dataset.char_to_id),
        num_tags=len(processor.all_labels),
        char_embedding_dim=CHAR_EMB_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        fasttext_embedding_dim=FASTTEXT_DIM
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = amp.GradScaler(enabled=device.type == "cuda")

    best_dev_der = float("inf")
    patience_counter = 0

    print(f"\n--- Training started on {device} ---")

    # ==============================================================
    # TRAINING LOOP
    # ==============================================================
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)",
            unit="batch",
            dynamic_ncols=True
        )

        for batch_idx, batch in enumerate(train_bar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            lengths = batch["lengths"]

            # FastText alignment (accuracy-optimal)
            fasttext_vectors = aligner.align_features(
                input_ids,
                lengths
            ).to(device)

            optimizer.zero_grad()

            with amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                loss = model.neg_log_likelihood(
                    input_ids,
                    labels,
                    lengths,
                    fasttext_vectors
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

            if (batch_idx + 1) % BATCH_PRINT_FREQ == 0:
                print(
                    f"Train | Epoch {epoch+1} | "
                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}",
                    flush=True
                )

        avg_train_loss = total_loss / len(train_loader)

        # ----------------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------------
        dev_der = evaluate_model(model, dev_loader, aligner)

        epoch_time = time.time() - start_time
        print(
            f"\nEpoch {epoch+1} | "
            f"Time {epoch_time:.1f}s | "
            f"Train Loss {avg_train_loss:.4f} | "
            f"Dev DER {dev_der:.4f}",
            flush=True
        )

        # ----------------------------------------------------------
        # CHECKPOINT DICT (used for both saves)
        # ----------------------------------------------------------
        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "char_to_id": train_dataset.char_to_id,
            "id_to_label": processor.id_to_label,
            "fasttext_model_path": FASTTEXT_MODEL_PATH,
            "char_emb_dim": CHAR_EMB_DIM,
            "lstm_hidden_dim": LSTM_HIDDEN_DIM,
            "fasttext_dim": FASTTEXT_DIM,
            "epoch": epoch + 1,
            "dev_der": dev_der,
        }

        # ----------------------------------------------------------
        # SAVE EPOCH CHECKPOINT (always)
        # ----------------------------------------------------------
        epoch_ckpt_path = f"{EPOCH_CKPT_DIR}/epoch_{epoch+1}.pth"
        torch.save(checkpoint_dict, epoch_ckpt_path)

        # ----------------------------------------------------------
        # SAVE BEST CHECKPOINT (fail-hard)
        # ----------------------------------------------------------
        if dev_der < best_dev_der:
            best_dev_der = dev_der
            patience_counter = 0

            print(">>> New best model — saving checkpoint", flush=True)
            torch.save(checkpoint_dict, BEST_CKPT_PATH)
            assert os.path.exists(BEST_CKPT_PATH), "BEST checkpoint save FAILED"

        else:
            patience_counter += 1
            print(f"No improvement | Patience {patience_counter}/{PATIENCE}", flush=True)
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.", flush=True)
                break


# ------------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------------
def evaluate_model(model, dataloader, aligner):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            lengths = batch["lengths"]

            fasttext_vectors = aligner.align_features(
                input_ids,
                lengths
            ).to(device)

            predictions = model.forward(
                input_ids,
                lengths,
                fasttext_vectors
            )

            for i, length in enumerate(lengths.tolist()):
                all_preds.extend(predictions[i])
                all_labels.extend(labels[i, :length].tolist())

    return sum(p != t for p, t in zip(all_preds, all_labels)) / len(all_labels)


# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------
if __name__ == "__main__":

    TRAIN_FILE = "data/cleaned/trainc_data.txt"
    DEV_FILE = "data/cleaned/valc_data.txt"

    train_diacritization_model(TRAIN_FILE, DEV_FILE)
