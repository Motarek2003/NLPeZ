from huggingface_hub import HfApi

api = HfApi()

repo_id = "QuantumHayder/arabic-diacritizer"

api.upload_file(
    path_or_fileobj="models/epoch_32.pth",
    path_in_repo="models/fasttext_bilstm_crf.pth",
    repo_id=repo_id,
)

api.upload_file(
    path_or_fileobj="data/embeddings/fasttext_word_vectors.bin",
    path_in_repo="embeddings/fasttext_word_vectors.bin",
    repo_id=repo_id,
)
