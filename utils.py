import torch

def collate_fn(batch):
    """
    Collate function for DiacritizationDataset.
    Does NOT use words anymore.
    """

    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    lengths = torch.tensor(
        [item["lengths"] for item in batch],
        dtype=torch.long
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "lengths": lengths
    }
