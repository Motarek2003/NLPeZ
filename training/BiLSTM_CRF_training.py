import sys
import os
import time
import logging
import random
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
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
# CONFIGURATION (GPU-ENHANCED for Maximum Quality)
# ------------------------------------------------------------------
BATCH_SIZE = 32  # Smaller batch = more gradient updates = better generalization
MAX_SEQ_LENGTH = 400  # Longer context for better understanding
CHAR_EMB_DIM = 256  # Rich character representations
LSTM_HIDDEN_DIM = 512  # Large hidden state for complex patterns
FASTTEXT_DIM = 300  # Standard high-quality dimension
POS_EMB_DIM = 128  # Rich POS embeddings for grammatical context
LEARNING_RATE = 1e-3  # Lower LR for stable, quality training
NUM_EPOCHS = 10  # More epochs to fully converge
PATIENCE = 10  # Allow more exploration before early stopping
BATCH_PRINT_FREQ = 50  # More frequent logging
NUM_LAYERS = 3  # Deeper model for complex patterns
DROPOUT = 0.4  # Moderate regularization
TARGET_ACCURACY = 0.998  # Target accuracy (1 - DER)
GRADIENT_CLIP_VAL = 1.0  # Prevent exploding gradients
WEIGHT_DECAY = 1e-4  # Stronger regularization for AdamW
SEED = 42  # For reproducibility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

# ------------------------------------------------------------------
# TRAINING FUNCTION
# ------------------------------------------------------------------
def train_diacritization_model(train_file, dev_file, fasttext_model_path, fasttext_corpus_path="data/undiacritized/traincu_data.txt"):

    set_seed(SEED)
    
    output_dir = os.path.join(PROJECT_ROOT, "training", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger(os.path.join(output_dir, "training.log"))
    logger.info(f"Training started on {device}")

    processor = ArabicDiacritizationProcessor()

    # -----------------------------
    # Build / Load Datasets
    # -----------------------------
    cache_dir = os.path.join(PROJECT_ROOT, "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    train_cache_path = os.path.join(cache_dir, "train_dataset.pt")
    dev_cache_path = os.path.join(cache_dir, "dev_dataset.pt")

    # TRAIN DATASET
    if os.path.exists(train_cache_path):
        train_dataset = DiacritizationDataset.load(train_cache_path)
    else:
        # -----------------------------
        # Load / Train FastText (Only needed if building dataset)
        # -----------------------------
        fasttext_feature = FastTextEmbeddings(
            corpus_path=fasttext_corpus_path,
            output_path=fasttext_model_path,
            dim=FASTTEXT_DIM
        )
        fasttext_model = fasttext_feature.get_or_train()
        
        train_dataset = DiacritizationDataset(
            train_file,
            processor,
            MAX_SEQ_LENGTH
        )
        
        # Create aligner and cache features
        aligner = FastTextFeatureAligner(fasttext_model, train_dataset.id_to_char)
        
        print("Caching FastText features for Training set...")
        train_dataset.cached_fasttext = []
        for i in tqdm(range(len(train_dataset)), desc="FastText Align (Train)"):
            item = train_dataset[i]
            input_ids = item["input_ids"].unsqueeze(0)
            lengths = item["lengths"].unsqueeze(0)
            
            # Align features using char_ids
            vecs = aligner.align_features(input_ids, lengths)
            train_dataset.cached_fasttext.append(vecs.squeeze(0))
            
        train_dataset.save(train_cache_path)

    # DEV DATASET
    if os.path.exists(dev_cache_path):
        dev_dataset = DiacritizationDataset.load(dev_cache_path)
    else:
        # Ensure we have the aligner if we didn't build train_dataset just now
        if 'aligner' not in locals():
             fasttext_feature = FastTextEmbeddings(
                corpus_path=fasttext_corpus_path,
                output_path=fasttext_model_path,
                dim=FASTTEXT_DIM
            )
             fasttext_model = fasttext_feature.get_or_train()
             aligner = FastTextFeatureAligner(fasttext_model, train_dataset.id_to_char)

        dev_dataset = DiacritizationDataset(
            dev_file,
            processor,
            MAX_SEQ_LENGTH
        )
        # Share vocab
        dev_dataset.char_to_id = train_dataset.char_to_id
        dev_dataset.id_to_char = train_dataset.id_to_char
        dev_dataset.pos_to_id = train_dataset.pos_to_id
        dev_dataset.id_to_pos = train_dataset.id_to_pos
        
        print("Caching FastText features for Dev set...")
        dev_dataset.cached_fasttext = []
        for i in tqdm(range(len(dev_dataset)), desc="FastText Align (Dev)"):
            item = dev_dataset[i]
            input_ids = item["input_ids"].unsqueeze(0)
            lengths = item["lengths"].unsqueeze(0)
            
            vecs = aligner.align_features(input_ids, lengths)
            dev_dataset.cached_fasttext.append(vecs.squeeze(0))
            
        dev_dataset.save(dev_cache_path)

    # -----------------------------
    # DataLoaders
    # -----------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,  # OPTIMIZED: Added parallel data loading
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,  # OPTIMIZED: Added parallel data loading
        pin_memory=True
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = Arabic_BiLSTM_CRF(
        char_vocab_size=len(train_dataset.char_to_id),
        num_tags=len(processor.all_labels),
        pos_vocab_size=len(train_dataset.pos_to_id),
        char_embedding_dim=CHAR_EMB_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        fasttext_embedding_dim=FASTTEXT_DIM,
        pos_embedding_dim=POS_EMB_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_attention=True  # Enable self-attention for better long-range context
    ).to(device)
    
    # -----------------------------
    # Resume from checkpoint if available — prefer `inference/model.pkl`
    # -----------------------------
    inference_ckpt = os.path.join(PROJECT_ROOT, "inference", "model.pkl")
    output_ckpt = os.path.join(output_dir, "model.pkl")
    # pick inference checkpoint if present, otherwise output checkpoint
    checkpoint_path = inference_ckpt if os.path.exists(inference_ckpt) else output_ckpt
    if os.path.exists(checkpoint_path):
        try:
            import pickle as _pickle
            logger.info(f"Found checkpoint at {checkpoint_path} — loading weights (will resume training)")
            with open(checkpoint_path, 'rb') as _f:
                _ckpt = _pickle.load(_f)

            # Load model weights (handle CPU/GPU devices)
            model.load_state_dict(_ckpt.get('model_state_dict', _ckpt.get('state_dict')))

            # Check vocab compatibility and warn if mismatch
            ckpt_char_to_id = _ckpt.get('char_to_id')
            if ckpt_char_to_id and ckpt_char_to_id != train_dataset.char_to_id:
                logger.warning("Checkpoint char_to_id differs from current dataset vocabulary — this may cause wrong predictions")

            ckpt_pos_to_id = _ckpt.get('pos_to_id')
            if ckpt_pos_to_id and ckpt_pos_to_id != train_dataset.pos_to_id:
                logger.warning("Checkpoint pos_to_id differs from current dataset POS vocabulary")

            # Evaluate current checkpoint on dev to set baseline
            try:
                model.eval()
                baseline_dev_der = evaluate_model(model, dev_loader, device)
                best_dev_der = min(best_dev_der, baseline_dev_der)
                logger.info(f"Baseline Dev DER from checkpoint: {baseline_dev_der:.4f}")
            except Exception as e:
                logger.warning(f"Could not evaluate checkpoint before training: {e}")

        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Cosine annealing for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_dev_der = float("inf")
    patience_counter = 0

    logger.info(f"\n--- Training Loop Started ---")

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------
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
            pos_ids = batch["pos_ids"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)  # OPTIMIZED: Move to device
            
            # FastText is now part of the batch from dataset
            fasttext_vectors = batch["fasttext"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # OPTIMIZED: More efficient

            with amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                loss = model.neg_log_likelihood(
                    input_ids,
                    labels,
                    lengths,
                    fasttext_vectors,
                    pos_ids
                )

            scaler.scale(loss).backward()
            
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

            if (batch_idx + 1) % BATCH_PRINT_FREQ == 0:
                logger.info(
                    f"Train | Epoch {epoch+1} | "
                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        avg_train_loss = total_loss / len(train_loader)

        # -----------------------------
        # VALIDATION
        # -----------------------------
        dev_der = evaluate_model(model, dev_loader, device)
        
        # Update scheduler (Cosine Annealing doesn't need metrics)
        scheduler.step(epoch)

        epoch_time = time.time() - start_time
        current_accuracy = 1.0 - dev_der
        current_lr = optimizer.param_groups[0]['lr']
        epoch_msg = (
            f"Epoch {epoch+1} | "
            f"Time {epoch_time:.1f}s | "
            f"Train Loss {avg_train_loss:.4f} | "
            f"Dev DER {dev_der:.4f} | "
            f"Dev Accuracy {current_accuracy:.4f} | "
            f"LR {current_lr:.2e}"
        )
        print(epoch_msg)
        logger.info(epoch_msg)

        # -----------------------------
        # CHECKPOINT BEST MODEL
        # -----------------------------
        if dev_der < best_dev_der:
            best_dev_der = dev_der
            patience_counter = 0
            logger.info(">>> New best model — saving checkpoint")

            # Save as pickle file
            import pickle
            save_path = os.path.join(output_dir, "best_diacritization_model.pkl")
            
            checkpoint_data = {
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "char_to_id": train_dataset.char_to_id,
                "id_to_char": train_dataset.id_to_char,
                "id_to_label": processor.id_to_label,
                "label_to_id": processor.label_to_id,
                "pos_to_id": train_dataset.pos_to_id,
                "id_to_pos": train_dataset.id_to_pos,
                "fasttext_model_path": fasttext_model_path,
                "char_emb_dim": CHAR_EMB_DIM,
                "lstm_hidden_dim": LSTM_HIDDEN_DIM,
                "fasttext_dim": FASTTEXT_DIM,
                "pos_emb_dim": POS_EMB_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "max_seq_length": MAX_SEQ_LENGTH
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            print(f"Model saved to: {save_path}")

            # Log milestone
            if current_accuracy >= TARGET_ACCURACY:
                logger.info(f">>> Reached target accuracy of {TARGET_ACCURACY*100}%!")

        else:
            patience_counter += 1
            logger.info(f"No improvement | Patience {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                logger.info("Early stopping triggered.")
                break
    
    # FINAL SUMMARY
    print("=" * 70)
    print("TRAINING COMPLETED")
    print(f"Best Dev DER: {best_dev_der:.4f}")
    print(f"Best Dev Accuracy: {1 - best_dev_der:.4f}")
    print(f"Model saved to: {os.path.join(output_dir, 'best_diacritization_model.pkl')}")
    print("=" * 70)
    
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best Dev DER: {best_dev_der:.4f}")
    logger.info(f"Best Dev Accuracy: {1 - best_dev_der:.4f}")
    logger.info(f"Model saved to: {os.path.join(output_dir, 'best_diacritization_model.pkl')}")
    logger.info("=" * 70)


# ------------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pos_ids = batch["pos_ids"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)  # OPTIMIZED: Move to device

            fasttext_vectors = batch["fasttext"].to(device, non_blocking=True)

            predictions = model.forward(
                input_ids,
                lengths,
                fasttext_vectors,
                pos_ids
            )

            for i, length in enumerate(lengths.tolist()):
                all_preds.extend(predictions[i])
                all_labels.extend(labels[i, :length].tolist())

    # Calculate overall DER
    total_errors = sum(p != t for p, t in zip(all_preds, all_labels))
    overall_der = total_errors / len(all_labels)
    
    # Calculate DER only on diacritized characters (excluding <NT> label)
    # Assuming label 0 or the last label is <NT> - need to check
    # For now, let's also compute error on positions that HAVE diacritics in ground truth
    diacritized_errors = 0
    diacritized_count = 0
    nt_label_id = None
    
    # Find the <NT> label id (typically the last one or a specific one)
    # We'll use label statistics to identify the most common one (likely <NT>)
    from collections import Counter
    label_counts = Counter(all_labels)
    most_common_label = label_counts.most_common(1)[0][0]
    
    for p, t in zip(all_preds, all_labels):
        if t != most_common_label:  # Only count non-NT (diacritized) positions
            diacritized_count += 1
            if p != t:
                diacritized_errors += 1
    
    diac_der = diacritized_errors / max(diacritized_count, 1)
    
    print(f"  [Eval] Overall DER: {overall_der:.4f} | Diacritized-only DER: {diac_der:.4f} ({diacritized_count}/{len(all_labels)} chars have diacritics)")
    
    return overall_der


# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------
if __name__ == "__main__":

    TRAIN_FILE = "data/cleaned/trainc_data.txt"
    DEV_FILE = "data/cleaned/valc_data.txt"
    FASTTEXT_MODEL_PATH = "data/embeddings/fasttext_word_vectors.model"

    train_diacritization_model(
        TRAIN_FILE,
        DEV_FILE,
        FASTTEXT_MODEL_PATH
    )
