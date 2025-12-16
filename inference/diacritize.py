"""
Arabic Diacritization Inference Script
Usage:
    python inference/diacritize.py --model path/to/model.pkl --input input.txt --output output.txt
"""

import sys
import os
import argparse
import pickle
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.BiLSTM_CRF_Parallel import Arabic_BiLSTM_CRF
from preprocessing.data_processor import ArabicDiacritizationProcessor
from features.FastText import FastTextEmbeddings
from Feature_Aligner.FastTextAligner import FastTextFeatureAligner

# Try to import CAMeLBERT for POS tagging
try:
    from camel_tools.tagger.default import DefaultTagger
    POS_TAGGER = DefaultTagger.pretrained()
    HAS_POS = True
except:
    HAS_POS = False
    print("Warning: CAMeLBERT POS tagger not available. Using dummy POS tags.")


def load_model(pickle_path: str, device: torch.device):
    """Load model from pickle file."""
    print(f"Loading model from: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Extract config
    char_to_id = checkpoint["char_to_id"]
    id_to_char = checkpoint["id_to_char"]
    id_to_label = checkpoint["id_to_label"]
    label_to_id = checkpoint["label_to_id"]
    pos_to_id = checkpoint["pos_to_id"]
    id_to_pos = checkpoint["id_to_pos"]
    fasttext_model_path = checkpoint["fasttext_model_path"]
    
    char_emb_dim = checkpoint.get("char_emb_dim", 256)
    lstm_hidden_dim = checkpoint.get("lstm_hidden_dim", 512)
    fasttext_dim = checkpoint.get("fasttext_dim", 300)
    pos_emb_dim = checkpoint.get("pos_emb_dim", 128)
    num_layers = checkpoint.get("num_layers", 3)
    dropout = checkpoint.get("dropout", 0.4)
    max_seq_length = checkpoint.get("max_seq_length", 400)
    
    # Create model
    model = Arabic_BiLSTM_CRF(
        char_vocab_size=len(char_to_id),
        num_tags=len(label_to_id),
        pos_vocab_size=len(pos_to_id),
        char_embedding_dim=char_emb_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        fasttext_embedding_dim=fasttext_dim,
        pos_embedding_dim=pos_emb_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_attention=True
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  - Vocab size: {len(char_to_id)}")
    print(f"  - Num labels: {len(label_to_id)}")
    print(f"  - Max seq length: {max_seq_length}")
    
    return model, {
        "char_to_id": char_to_id,
        "id_to_char": id_to_char,
        "id_to_label": id_to_label,
        "label_to_id": label_to_id,
        "pos_to_id": pos_to_id,
        "id_to_pos": id_to_pos,
        "fasttext_model_path": fasttext_model_path,
        "max_seq_length": max_seq_length
    }


def get_pos_tags(text: str, pos_to_id: dict) -> list:
    """Get POS tags for text."""
    processor = ArabicDiacritizationProcessor()
    undiacritized = processor.strip_diacritics(text)
    
    if HAS_POS:
        words = undiacritized.split()
        if words:
            pos_tags = POS_TAGGER.tag(words)
        else:
            pos_tags = []
        
        # Map POS tags to character level
        char_pos = []
        word_idx = 0
        char_count = 0
        
        for char in undiacritized:
            if char == ' ':
                char_pos.append(pos_to_id.get("<PAD>", 0))
            else:
                if word_idx < len(pos_tags):
                    pos = pos_tags[word_idx]
                    char_pos.append(pos_to_id.get(pos, pos_to_id.get("<UNK>", 0)))
                else:
                    char_pos.append(pos_to_id.get("<UNK>", 0))
                
                char_count += 1
                # Move to next word when we've processed all chars of current word
                if word_idx < len(words) and char_count >= len(words[word_idx]):
                    word_idx += 1
                    char_count = 0
        
        return char_pos
    else:
        # Return dummy POS tags
        return [pos_to_id.get("<UNK>", 0)] * len(undiacritized)


def diacritize_chunk(text_chunk: str, model, vocab, fasttext_aligner, device, processor) -> str:
    """Diacritize a single chunk of text (up to max_seq_length)."""
    char_to_id = vocab["char_to_id"]
    id_to_label = vocab["id_to_label"]
    pos_to_id = vocab["pos_to_id"]
    max_seq_length = vocab["max_seq_length"]
    
    if not text_chunk:
        return ""
    
    # Convert characters to IDs
    char_ids = [char_to_id.get(c, char_to_id.get("<UNK>", 0)) for c in text_chunk]
    length = len(char_ids)
    
    # Pad to max_seq_length
    padded_ids = char_ids + [0] * (max_seq_length - len(char_ids))
    
    # Get POS tags
    pos_ids = get_pos_tags(text_chunk, pos_to_id)
    pos_ids = pos_ids[:max_seq_length]
    pos_ids = pos_ids + [0] * (max_seq_length - len(pos_ids))
    
    # Convert to tensors - lengths stays on CPU for pack_padded_sequence
    input_ids = torch.tensor([padded_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([length], dtype=torch.long)  # CPU for packing
    lengths_device = torch.tensor([length], dtype=torch.long, device=device)  # GPU for masking
    pos_ids_tensor = torch.tensor([pos_ids], dtype=torch.long, device=device)
    
    # Get FastText embeddings
    fasttext_vectors = fasttext_aligner.align_features(input_ids.cpu(), lengths)
    fasttext_vectors = fasttext_vectors.to(device)
    
    # Run inference - pass GPU lengths for masking
    with torch.no_grad():
        predictions = model.forward(input_ids, lengths_device, fasttext_vectors, pos_ids_tensor)
    
    # Reconstruct diacritized text
    pred_labels = predictions[0][:length]
    
    diacritized = ""
    for i, char in enumerate(text_chunk):
        diacritized += char
        if i < len(pred_labels):
            label = id_to_label.get(pred_labels[i], "<NT>")
            if label != "<NT>":
                diacritized += label
    
    return diacritized


def diacritize_text(text: str, model, vocab, fasttext_aligner, device, processor) -> str:
    """Diacritize a single line of text, handling long lines by chunking. Never truncates."""
    max_seq_length = vocab["max_seq_length"]
    
    # Strip existing diacritics but preserve ALL characters including spaces
    undiacritized = processor.strip_diacritics(text.rstrip('\n\r'))
    
    if not undiacritized:
        return ""
    
    # Process in chunks if too long - NEVER truncate
    if len(undiacritized) <= max_seq_length:
        return diacritize_chunk(undiacritized, model, vocab, fasttext_aligner, device, processor)
    
    # Split into fixed-size chunks, preserving every character
    chunk_size = max_seq_length
    diacritized_result = ""
    
    for start in range(0, len(undiacritized), chunk_size):
        chunk = undiacritized[start:start + chunk_size]
        diacritized_result += diacritize_chunk(chunk, model, vocab, fasttext_aligner, device, processor)
    
    return diacritized_result


def main():
    parser = argparse.ArgumentParser(description="Arabic Diacritization Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to pickle model file")
    parser.add_argument("--input", type=str, required=True, help="Path to input text file")
    parser.add_argument("--output", type=str, required=True, help="Path to output text file")
    parser.add_argument("--fasttext", type=str, default=None, help="Path to FastText model (optional, uses path from checkpoint)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, vocab = load_model(args.model, device)
    
    # Load FastText
    fasttext_path = args.fasttext or vocab["fasttext_model_path"]
    print(f"Loading FastText from: {fasttext_path}")
    
    fasttext_embeddings = FastTextEmbeddings(
        corpus_path="",  # Not needed for loading
        output_path=fasttext_path,
        dim=300
    )
    fasttext_model = fasttext_embeddings.get_or_train()
    fasttext_aligner = FastTextFeatureAligner(fasttext_model, vocab["id_to_char"])
    
    # Initialize processor
    processor = ArabicDiacritizationProcessor()
    
    # Read input file
    print(f"Reading input from: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Processing {len(lines)} lines...")
    
    # Diacritize each line
    diacritized_lines = []
    for line in tqdm(lines, desc="Diacritizing"):
        if line.strip():
            diacritized = diacritize_text(line, model, vocab, fasttext_aligner, device, processor)
            diacritized_lines.append(diacritized)
        else:
            diacritized_lines.append("")
    
    # Write output
    print(f"Writing output to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for line in diacritized_lines:
            f.write(line + '\n')
    
    print("Done!")
    print(f"Diacritized {len(diacritized_lines)} lines")


if __name__ == "__main__":
    main()
