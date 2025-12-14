import torch
from typing import List, Dict

# --- Collate Function for Batching and Padding ---
def collate_fn(batch: List[Dict[str, torch.Tensor]]):
    """
    Pads/truncates the sequences in a batch to the length of the longest sequence in the batch.
    Takes the output of DiacritizationDataset: dict with keys 'input_ids', 'labels', 'lengths', 'words'.
    """
    # 1. Separate inputs using explicit keys (if dataset returns dict)
    inputs = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    # Convert lengths to native Python ints
    lengths = [int(item['lengths'].item()) if isinstance(item['lengths'], torch.Tensor) else int(item['lengths']) for item in batch]
    words = [item['words'] for item in batch]  # KEEP WORDS

    # 2. Get max length in the current batch (dynamic padding)
    max_len = max(lengths)

    # 3. Trim or keep the first `max_len` tokens from each input/label.
    # If any entry is shorter than `max_len`, we pad it using PAD_ID=0 (common for char PAD)
    padded_inputs = []
    padded_labels = []
    for inp, lab in zip(inputs, labels):
        cur_len = inp.shape[0]
        if cur_len >= max_len:
            padded_inputs.append(inp[:max_len])
            padded_labels.append(lab[:max_len])
        else:
            pad_size = max_len - cur_len
            padded_inputs.append(torch.cat([inp, torch.full((pad_size,), fill_value=0, dtype=torch.long)]))
            padded_labels.append(torch.cat([lab, torch.full((pad_size,), fill_value=0, dtype=torch.long)]))

    padded_inputs = torch.stack(padded_inputs)
    padded_labels = torch.stack(padded_labels)

    # Sort the batch by sequence length (required for pack_padded_sequence)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    sorted_lengths, sorted_indices = torch.sort(lengths_tensor, descending=True)

    return {
        'input_ids': padded_inputs[sorted_indices],
        'labels': padded_labels[sorted_indices],
        'lengths': sorted_lengths,
        'words': [words[i] for i in sorted_indices.tolist()]
    }