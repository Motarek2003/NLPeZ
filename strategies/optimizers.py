import torch
import torch.nn as nn
from typing import List, Tuple
from interfaces.model import OptimizerStrategy


# =============================================================================
# ADAMW OPTIMIZER STRATEGY
# =============================================================================

class AdamWStrategy(OptimizerStrategy):
    """
    AdamW optimizer - Recommended for transformers (BERT, AraBERT).
    
    Use when: Fine-tuning pre-trained models (default choice)
    
    Features:
    - Proper weight decay handling
    - Excludes bias and LayerNorm from weight decay
    """
    
    def __init__(
        self,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8
    ):
        """
        Args:
            betas: Momentum coefficients (beta1, beta2)
            eps: Numerical stability constant
        """
        self.betas = betas
        self.eps = eps
    
    def create(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float = 0.01
    ) -> torch.optim.Optimizer:
        """Create AdamW optimizer with proper weight decay handling."""
        # Parameters that should NOT have weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        param_groups = [
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': weight_decay,
            },
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': 0.0,
            },
        ]
        
        return torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=self.betas,
            eps=self.eps
        )
    
    def name(self) -> str:
        return f"AdamW(betas={self.betas})"


# =============================================================================
# SGD OPTIMIZER STRATEGY
# =============================================================================

class SGDStrategy(OptimizerStrategy):
    """
    Stochastic Gradient Descent with momentum.
    
    Use when: Training from scratch or want simpler optimization
    """
    
    def __init__(
        self,
        momentum: float = 0.9,
        nesterov: bool = True
    ):
        """
        Args:
            momentum: Momentum factor (0.9 is standard)
            nesterov: Use Nesterov momentum (usually better)
        """
        self.momentum = momentum
        self.nesterov = nesterov
    
    def create(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float = 0.01
    ) -> torch.optim.Optimizer:
        """Create SGD optimizer with momentum."""
        return torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            momentum=self.momentum,
            weight_decay=weight_decay,
            nesterov=self.nesterov
        )
    
    def name(self) -> str:
        return f"SGD(momentum={self.momentum})"



# =============================================================================
# LAYERWISE LEARNING RATE STRATEGY
# =============================================================================

class LayerwiseLRStrategy(OptimizerStrategy):
    """
    AdamW with different learning rates per layer.
    
    Why Layerwise LR?
    - Lower layers (embeddings) learn general features → need small LR
    - Higher layers (classifier) need task-specific tuning → need larger LR
    - Prevents catastrophic forgetting of pre-trained knowledge
    
    How it works:
    - Top layer: base learning rate
    - Each lower layer: LR * decay_factor
    
    Example (decay=0.9, base_lr=1e-4):
        Layer 11: 1e-4 (top/classifier)
        Layer 10: 9e-5
        Layer 9:  8.1e-5
        ...
        Layer 0:  3.5e-5 (bottom/embeddings)
    
    Usage:
        strategy = LayerwiseLRStrategy(lr_decay=0.95)
    """
    
    def __init__(
        self,
        lr_decay: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8
    ):
        """
        Args:
            lr_decay: Decay factor per layer (0.9-0.99 typical)
                      Lower = more aggressive decay
            betas: Adam beta parameters
            eps: Numerical stability
        """
        self.lr_decay = lr_decay
        self.betas = betas
        self.eps = eps
    
    def create(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float = 0.01
    ) -> torch.optim.Optimizer:
        """Create AdamW with layer-wise learning rates."""
        param_groups = []
        
        # Check if model has BERT-style encoder
        if hasattr(model, 'bert'):
            # Get number of encoder layers
            num_layers = len(model.bert.encoder.layer)
            
            # Embeddings (lowest LR)
            embeddings_lr = learning_rate * (self.lr_decay ** num_layers)
            param_groups.append({
                'params': list(model.bert.embeddings.parameters()),
                'lr': embeddings_lr,
                'weight_decay': weight_decay,
                'name': 'embeddings'
            })
            
            # Encoder layers (progressively higher LR)
            for i, layer in enumerate(model.bert.encoder.layer):
                layer_lr = learning_rate * (self.lr_decay ** (num_layers - i - 1))
                param_groups.append({
                    'params': list(layer.parameters()),
                    'lr': layer_lr,
                    'weight_decay': weight_decay,
                    'name': f'encoder_layer_{i}'
                })
        
        # Non-BERT parameters (classifier, CRF, etc.) - use base LR
        bert_params = set()
        if hasattr(model, 'bert'):
            bert_params = set(model.bert.parameters())
        
        other_params = [
            p for p in model.parameters()
            if p.requires_grad and p not in bert_params
        ]
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': learning_rate,
                'weight_decay': weight_decay,
                'name': 'classifier'
            })
        
        return torch.optim.AdamW(
            param_groups,
            betas=self.betas,
            eps=self.eps
        )
    
    def name(self) -> str:
        return f"LayerwiseLR(decay={self.lr_decay})"


# =============================================================================
# RMSPROP OPTIMIZER STRATEGY
# =============================================================================

class RMSpropStrategy(OptimizerStrategy):
    """
    RMSprop optimizer.
    
    Why RMSprop?
    - Good for RNNs and non-stationary problems
    - Divides learning rate by running average of gradient magnitudes
    - Simpler than Adam (no momentum on gradient)
    
    Usage:
        strategy = RMSpropStrategy()
        strategy = RMSpropStrategy(alpha=0.9, centered=True)
    """
    
    def __init__(
        self,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
        centered: bool = False
    ):
        """
        Args:
            alpha: Smoothing constant (0.99 is standard)
            eps: Numerical stability
            momentum: Momentum factor
            centered: If True, compute centered RMSprop
        """
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.centered = centered
    
    def create(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float = 0.01
    ) -> torch.optim.Optimizer:
        """Create RMSprop optimizer."""
        return torch.optim.RMSprop(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            alpha=self.alpha,
            eps=self.eps,
            momentum=self.momentum,
            weight_decay=weight_decay,
            centered=self.centered
        )
    
    def name(self) -> str:
        return f"RMSprop(alpha={self.alpha})"
    


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_optimizer_strategy(name: str, **kwargs) -> OptimizerStrategy:
    """
    Get optimizer strategy by name.
    
    Args:
        name: 'adamw', 'sgd', 'rmsprop', or 'layerwise_lr'
        **kwargs: Arguments for the strategy
        
    Usage:
        strategy = get_optimizer_strategy('adamw')
        strategy = get_optimizer_strategy('sgd', momentum=0.99)
        strategy = get_optimizer_strategy('rmsprop', alpha=0.9)
        strategy = get_optimizer_strategy('layerwise_lr', lr_decay=0.95)
    """
    strategies = {
        'adamw': AdamWStrategy,
        'sgd': SGDStrategy,
        'rmsprop': RMSpropStrategy,
        'layerwise_lr': LayerwiseLRStrategy,
    }
    
    name_lower = name.lower()
    
    if name_lower not in strategies:
        raise ValueError(f"Unknown optimizer: '{name}'. Available: adamw, sgd, rmsprop, layerwise_lr")
    
    return strategies[name_lower](**kwargs)





