from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List

import os
import tempfile
import torch
from huggingface_hub import hf_hub_download

from preprocessing.data_processor import ArabicDiacritizationProcessor
from preprocessing.diacritization_dataset import DiacritizationDataset, PAD_TOKEN, UNK_TOKEN
from features.FastText import FastTextEmbeddings
from Feature_Aligner.FastTextAligner import FastTextFeatureAligner
from models.BiLSTM_CRF_Parallel import Arabic_BiLSTM_CRF

app = FastAPI(title="Arabic Diacritization API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Configuration: file paths / HF
# ---------------------------
# Local fallback paths (if you prefer to place files locally instead of HF)
FASTTEXT_VECTORS_PATH = os.getenv("FASTTEXT_LOCAL_PATH", "data/embeddings/fasttext_word_vectors.bin")
PYTORCH_MODEL_PATH = os.getenv("PYTORCH_LOCAL_PATH", "models/fasttext_bilstm_crf.pth")

# Hugging Face repo info (optional). If provided, the server will try to download
# artifacts from HF Hub at startup. Example repo id used when uploading:
#   QuantumHayder/arabic-diacritizer
HF_REPO_ID = os.getenv("HF_REPO_ID", "QuantumHayder/arabic-diacritizer")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Cache directory for downloaded artifacts
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", os.path.join(tempfile.gettempdir(), "arabic_diacritizer"))
os.makedirs(HF_CACHE_DIR, exist_ok=True)


class TextIn(BaseModel):
    text: str

def load_resources(
    fasttext_path: str = FASTTEXT_VECTORS_PATH,
    model_path: str = PYTORCH_MODEL_PATH,
):
    processor = ArabicDiacritizationProcessor()

    # ---------------------------
    # Download from HF only if not present locally
    # ---------------------------
    HF_FASTTEXT_PATH_IN_REPO = "embeddings/fasttext_word_vectors.bin"
    HF_MODEL_PATH_IN_REPO = "models/best_model.pth"

    fasttext_local_path = fasttext_path
    pytorch_local_path = model_path

    if not os.path.exists(fasttext_local_path):
        fasttext_local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FASTTEXT_PATH_IN_REPO,
            token=HF_TOKEN,
            cache_dir=HF_CACHE_DIR,
        )
        print(f"Downloaded FastText to {fasttext_local_path}")
    else:
        print(f"Using local FastText: {fasttext_local_path}")

    if not os.path.exists(pytorch_local_path):
        pytorch_local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_PATH_IN_REPO,
            token=HF_TOKEN,
            cache_dir=HF_CACHE_DIR,
        )
        print(f"Downloaded model to {pytorch_local_path}")
    else:
        print(f"Using local model: {pytorch_local_path}")

    # ---------------------------
    # Load checkpoint
    # ---------------------------
    checkpoint = torch.load(pytorch_local_path, map_location="cpu")

    char_to_id = checkpoint["char_to_id"]
    id_to_label = checkpoint["id_to_label"]

    processor.id_to_label = id_to_label
    processor.label_to_id = {v: k for k, v in id_to_label.items()}

    # ---------------------------
    # Load FastText
    # ---------------------------
    fasttext = FastTextEmbeddings(
        corpus_path="__unused__",  # not needed at inference
        output_path=fasttext_local_path,
    )
    fasttext_model = fasttext.load_model()

    # ---------------------------
    # Build FastText aligner
    # ---------------------------
    id_to_char = {i: c for c, i in char_to_id.items()}
    fasttext_aligner = FastTextFeatureAligner(
        raw_fasttext_model=fasttext_model,
        id_to_char_vocab=id_to_char,
    )

    # ---------------------------
    # Rebuild model with saved dimensions
    # ---------------------------
    model = Arabic_BiLSTM_CRF(
        char_vocab_size=len(char_to_id),
        num_tags=len(processor.label_to_id),
        char_embedding_dim=checkpoint["char_emb_dim"],
        lstm_hidden_dim=checkpoint["lstm_hidden_dim"],
        fasttext_embedding_dim=checkpoint["fasttext_dim"],
        dropout=0.3,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return {
        "processor": processor,
        "char_to_id": char_to_id,
        "fasttext_aligner": fasttext_aligner,
        "model": model,
    }

# Load on import so the API is ready (will raise clear errors if files missing)
resources = None
try:
    resources = load_resources()
except Exception as e:
    # Keep startup lightweight: log the exception and allow the server to start so the user can see the error
    print(f"[Warning] Failed to fully initialize resources: {e}")


def _prepare_inputs(
    text: str,
    processor: ArabicDiacritizationProcessor,
    char_to_id: dict,
    max_len: int = 256,
):
    cleaned = processor.clean_text(text)
    undiacritized = processor.strip_diacritics(cleaned)

    char_seq = list(undiacritized)
    orig_len = min(len(char_seq), max_len)

    char_ids = [
        char_to_id.get(c, char_to_id[UNK_TOKEN])
        for c in char_seq[:max_len]
    ]

    padding = max_len - len(char_ids)
    if padding > 0:
        char_ids.extend([char_to_id[PAD_TOKEN]] * padding)

    input_ids = torch.tensor([char_ids], dtype=torch.long)
    lengths = torch.tensor([orig_len], dtype=torch.long)

    return input_ids, lengths, orig_len, undiacritized

def predict(text: str) -> str:
    if resources is None:
        raise RuntimeError("Models not initialized")

    processor = resources["processor"]
    char_to_id = resources["char_to_id"]
    fasttext_aligner = resources["fasttext_aligner"]
    model = resources["model"]

    input_ids, lengths, orig_len, undiacritized = _prepare_inputs(
        text, processor, char_to_id
    )

    with torch.no_grad():
        ft_vecs = fasttext_aligner.align_features(
            input_ids.cpu(), lengths.cpu()
        ).to(device)

        preds = model(
            input_ids.to(device),
            lengths.to(device),
            ft_vecs,
        )

    pred_seq = preds[0][:orig_len]

    diacritized = []
    for ch, tag_id in zip(list(undiacritized)[:orig_len], pred_seq):
        label = processor.id_to_label.get(tag_id, processor.NO_TASHKEEL)
        diacritized.append(ch + label if label != processor.NO_TASHKEEL else ch)

    return "".join(diacritized)

@app.post("/diacritize")
def diacritize(payload: TextIn):
    text = payload.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")

    try:
        result = predict(text)
        return {"diacritized_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    info = {"status": "ok", "models_loaded": resources is not None}
    if resources is not None:
        info.update({
            "hf_repo_id": HF_REPO_ID,
            "fasttext_local_path": resources.get("fasttext_local_path"),
            "pytorch_local_path": resources.get("pytorch_local_path"),
            "hf_cache_dir": HF_CACHE_DIR,
        })
    return info


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)