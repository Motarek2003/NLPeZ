import torch

def collate_fn(batch):
    """
    Collate function for DiacritizationDataset.
    Does NOT use words anymore.
    """

    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    pos_ids = torch.stack([item["pos_ids"] for item in batch])
    lengths = torch.tensor(
        [item["lengths"] for item in batch],
        dtype=torch.long
    )
    
    # Handle FastText if present
    fasttext = None
    if "fasttext" in batch[0]:
        fasttext = torch.stack([item["fasttext"] for item in batch])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "pos_ids": pos_ids,
        "lengths": lengths,
        "fasttext": fasttext
    }
