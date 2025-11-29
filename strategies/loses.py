"""
Concrete Loss Strategies

These classes implement the LossStrategy interface.
Only essential losses for Arabic diacritization are included.

Usage:
    strategy = CrossEntropyStrategy()
    loss = strategy.compute(outputs, targets, mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from interfaces.model import LossStrategy


# =============================================================================
# CRF LOSS STRATEGY
# =============================================================================

class CRFLossStrategy(LossStrategy):
    """
    CRF Negative Log-Likelihood Loss.
    
    Use when: Model has a CRF layer (AraBERT-CRF)
    
    Note: CRF loss is computed inside the model's forward pass.
    This strategy extracts the pre-computed loss.
    """
    
    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract CRF loss from model outputs."""
        if 'loss' not in outputs:
            raise ValueError(
                "CRFLossStrategy requires 'loss' in model outputs. "
                "Ensure your model computes CRF loss in forward pass."
            )
        return outputs['loss']
    
    def name(self) -> str:
        return "CRF_NLL"


# =============================================================================
# CROSS ENTROPY LOSS STRATEGY
# =============================================================================

class CrossEntropyStrategy(LossStrategy):
    """
    Standard Cross-Entropy Loss.
    
    Use when: Simple classification without CRF layer
    
    Features:
    - Label smoothing (optional)
    - Ignores padding tokens
    """
    
    def __init__(
        self,
        label_smoothing: float = 0.0,
        ignore_index: int = -100
    ):
        """
        Args:
            label_smoothing: Smoothing factor (0.0 to 0.2)
            ignore_index: Label value to ignore (padding)
        """
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
    
    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        if 'logits' not in outputs:
            raise ValueError("CrossEntropyStrategy requires 'logits' in outputs.")
        
        logits = outputs['logits']
        
        # Flatten for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction='mean'
        )
        
        return loss
    
    def name(self) -> str:
        return f"CrossEntropy(smoothing={self.label_smoothing})"


# =============================================================================
# FOCAL LOSS STRATEGY
# =============================================================================

class FocalLossStrategy(LossStrategy):
    """
    Focal Loss for handling class imbalance.
    
    Use when: Some diacritics are rare (imbalanced classes)
    
    How it works:
    - Down-weights easy examples
    - Focuses training on hard examples
    - gamma=0: Same as cross-entropy
    - gamma=2: Recommended default
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        ignore_index: int = -100
    ):
        """
        Args:
            gamma: Focusing parameter (0-5, default 2.0)
            ignore_index: Label value to ignore
        """
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute focal loss."""
        if 'logits' not in outputs:
            raise ValueError("FocalLossStrategy requires 'logits' in outputs.")
        
        logits = outputs['logits']
        
        # Flatten
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # Filter valid positions
        valid_mask = targets_flat != self.ignore_index
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        valid_logits = logits_flat[valid_mask]
        valid_targets = targets_flat[valid_mask]
        
        # Compute probabilities
        probs = F.softmax(valid_logits, dim=-1)
        pt = probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight and cross-entropy
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction='none')
        
        return (focal_weight * ce_loss).mean()
    
    def name(self) -> str:
        return f"FocalLoss(gamma={self.gamma})"


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_loss_strategy(name: str, **kwargs) -> LossStrategy:
    """
    Get loss strategy by name.
    
    Args:
        name: 'crf', 'crossentropy', or 'focal'
        **kwargs: Arguments for the strategy
        
    Usage:
        strategy = get_loss_strategy('crf')
        strategy = get_loss_strategy('focal', gamma=3.0)
    """
    strategies = {
        'crf': CRFLossStrategy,
        'crossentropy': CrossEntropyStrategy,
        'ce': CrossEntropyStrategy,
        'focal': FocalLossStrategy,
    }
    
    name_lower = name.lower()
    
    if name_lower not in strategies:
        available = ', '.join(['crf', 'crossentropy', 'focal'])
        raise ValueError(f"Unknown loss: '{name}'. Available: {available}")
    
    return strategies[name_lower](**kwargs)