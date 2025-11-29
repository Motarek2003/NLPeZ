import math
import torch
from typing import Optional
from interfaces.model import SchedulerStrategy


# =============================================================================
# LINEAR WARMUP WITH DECAY STRATEGY
# =============================================================================

class LinearWarmupStrategy(SchedulerStrategy):
    """
    Linear warmup followed by linear decay.
    
    Use when: Fine-tuning AraBERT or CAMeLBERT (recommended default)
    
    How it works:
        Phase 1 (Warmup):  LR increases from 0 to base_lr
        Phase 2 (Decay):   LR decreases from base_lr to 0
  
    """
    
    def create(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create linear warmup scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            num_training_steps: Total training steps
            num_warmup_steps: Warmup steps (typically 10% of total)
            
        Returns:
            LambdaLR scheduler
        """
        def lr_lambda(current_step: int) -> float:
            # Warmup phase
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            # Decay phase
            progress = float(current_step - num_warmup_steps)
            total_decay_steps = float(max(1, num_training_steps - num_warmup_steps))
            
            return max(0.0, 1.0 - progress / total_decay_steps)
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def name(self) -> str:
        return "LinearWarmup"


# =============================================================================
# COSINE WARMUP STRATEGY
# =============================================================================

class CosineWarmupStrategy(SchedulerStrategy):
    """
    Linear warmup followed by cosine annealing decay.
    
    Use when: Want smoother decay than linear (good for longer training)
    
    How it works:
        Phase 1 (Warmup):  LR increases linearly
        Phase 2 (Cosine):  LR follows cosine curve to min_lr
    
    Good for: AraBERT, CAMeLBERT with longer training
    """
    
    def __init__(self, min_lr_ratio: float = 0.0):
        """
        Args:
            min_lr_ratio: Minimum LR as ratio of base LR (0.0 to 0.1)
                         0.0 = decay to zero
                         0.1 = decay to 10% of base LR
        """
        self.min_lr_ratio = min_lr_ratio
    
    def create(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create cosine warmup scheduler."""
        def lr_lambda(current_step: int) -> float:
            # Warmup phase
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            # Cosine decay phase
            progress = float(current_step - num_warmup_steps)
            total_decay_steps = float(max(1, num_training_steps - num_warmup_steps))
            
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress / total_decay_steps))
            
            # Scale between min_lr_ratio and 1.0
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def name(self) -> str:
        return f"CosineWarmup(min_lr={self.min_lr_ratio})"


# =============================================================================
# CONSTANT WITH WARMUP STRATEGY
# =============================================================================

class ConstantWarmupStrategy(SchedulerStrategy):
    """
    Linear warmup followed by constant learning rate.
    
    Use when: Short training or BiLSTM models
    
    How it works:
        Phase 1 (Warmup):   LR increases from 0 to base_lr
        Phase 2 (Constant): LR stays at base_lr
    
    Good for: BiLSTM, short fine-tuning
    """
    
    def create(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create constant warmup scheduler."""
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def name(self) -> str:
        return "ConstantWarmup"


# =============================================================================
# STEP DECAY STRATEGY
# =============================================================================

class StepDecayStrategy(SchedulerStrategy):
    """
    Step-wise learning rate decay.
    
    Use when: Training BiLSTM from scratch
    
    How it works:
        LR is multiplied by gamma every step_size steps
    
    Visual:
        LR │████
           │    ████
           │        ████
           │            ████
           └──────────────→ Steps
    
    Good for: BiLSTM, non-transformer models
    """
    
    def __init__(
        self,
        step_size: int = 1,
        gamma: float = 0.9
    ):
        """
        Args:
            step_size: Decay LR every N epochs (set to 1 for per-epoch decay)
            gamma: Multiplicative factor (0.9 = 10% reduction each step)
        """
        self.step_size = step_size
        self.gamma = gamma
    
    def create(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create step decay scheduler."""
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma
        )
    
    def name(self) -> str:
        return f"StepDecay(step={self.step_size}, gamma={self.gamma})"


# =============================================================================
# NO SCHEDULER STRATEGY
# =============================================================================

class NoSchedulerStrategy(SchedulerStrategy):
    """
    No learning rate scheduling (constant LR).
    
    Use when: Quick experiments or debugging
    """
    
    def create(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Return None (no scheduler)."""
        return None
    
    def name(self) -> str:
        return "None"


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_scheduler_strategy(name: str, **kwargs) -> SchedulerStrategy:
    """
    Get scheduler strategy by name.
    
    Args:
        name: 'linear', 'cosine', 'constant', 'step', or 'none'
        **kwargs: Arguments for the strategy
        
    Usage:
        strategy = get_scheduler_strategy('linear')
        strategy = get_scheduler_strategy('cosine', min_lr_ratio=0.1)
        strategy = get_scheduler_strategy('step', step_size=2, gamma=0.5)
    """
    strategies = {
        'linear': LinearWarmupStrategy,
        'cosine': CosineWarmupStrategy,
        'constant': ConstantWarmupStrategy,
        'step': StepDecayStrategy,
        'none': NoSchedulerStrategy,
    }
    
    name_lower = name.lower()
    
    if name_lower not in strategies:
        available = ', '.join(strategies.keys())
        raise ValueError(f"Unknown scheduler: '{name}'. Available: {available}")
    
    return strategies[name_lower](**kwargs)


# =============================================================================
# MODEL-SPECIFIC RECOMMENDATIONS
# =============================================================================

def get_recommended_scheduler(model_type: str) -> SchedulerStrategy:
    """
    Get recommended scheduler for model type.
    
    Args:
        model_type: 'arabert', 'camelbert', or 'bilstm'
        
    Returns:
        Recommended SchedulerStrategy
        
    Usage:
        strategy = get_recommended_scheduler('arabert')
    """
    recommendations = {
        'arabert': LinearWarmupStrategy(),
        'camelbert': LinearWarmupStrategy(),
        'bilstm': StepDecayStrategy(step_size=1, gamma=0.9),
        'lstm': StepDecayStrategy(step_size=1, gamma=0.9),
        'transformer': CosineWarmupStrategy(min_lr_ratio=0.0),
    }
    
    model_lower = model_type.lower()
    
    if model_lower not in recommendations:
        available = ', '.join(recommendations.keys())
        raise ValueError(f"Unknown model: '{model_type}'. Available: {available}")
    
    return recommendations[model_lower]