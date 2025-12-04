import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


# =============================================================================
# OPTIMIZER STRATEGY INTERFACE
# =============================================================================

class OptimizerStrategy(ABC):
    """
    Interface for optimizer creation.
    
    Any optimizer strategy must implement:
        - create(): Creates and returns an optimizer
        - name(): Returns the strategy name for logging

    """
    
    @abstractmethod
    def create(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float = 0.01
    ) -> torch.optim.Optimizer:
        """
        Create and return an optimizer for the model.
        
        Args:
            model: The neural network model
            learning_rate: Base learning rate
            weight_decay: Weight decay coefficient
            
        Returns:
            Configured optimizer instance
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of this optimizer strategy."""
        pass


# =============================================================================
# SCHEDULER STRATEGY INTERFACE
# =============================================================================

class SchedulerStrategy(ABC):
    """
    Interface for learning rate scheduler creation.
    
    Any scheduler strategy must implement:
        - create(): Creates and returns a scheduler (or None)
        - name(): Returns the strategy name for logging
    
    """
    
    @abstractmethod
    def create(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create and return a learning rate scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            num_training_steps: Total number of training steps
            num_warmup_steps: Number of warmup steps
            
        Returns:
            Configured scheduler instance or None
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of this scheduler strategy."""
        pass


# =============================================================================
# LOSS STRATEGY INTERFACE
# =============================================================================

class LossStrategy(ABC):
    """
    Interface for loss computation.
    
    Any loss strategy must implement:
        - compute(): Computes and returns the loss
        - name(): Returns the strategy name for logging
    
    
    """
    
    @abstractmethod
    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute and return the loss.
        
        Args:
            outputs: Model outputs dictionary (must contain 'logits' or 'loss')
            targets: Ground truth labels
            mask: Optional attention/padding mask
            
        Returns:
            Computed loss tensor
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of this loss strategy."""
        pass


# =============================================================================
# EVALUATION STRATEGY INTERFACE
# =============================================================================

class EvaluationStrategy(ABC):
    """
    Interface for model evaluation.
    
    Any evaluation strategy must implement:
        - evaluate(): Evaluates model and returns metrics
        - primary_metric(): Returns the main metric name
        - is_better(): Compares two metric values
    
   
    """
    
    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Evaluate the model and return metrics.
        
        Args:
            model: The model to evaluate
            dataloader: Evaluation data loader
            device: Device to run evaluation on
            
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    @abstractmethod
    def primary_metric(self) -> str:
        """
        Return the name of the primary metric for model selection.
        
        Returns:
            Metric name (e.g., 'der', 'accuracy', 'f1')
        """
        pass
    
    @abstractmethod
    def is_better(self, current: float, best: float) -> bool:
        """
        Determine if current metric value is better than best.
        
        Args:
            current: Current metric value
            best: Best metric value so far
            
        Returns:
            True if current is better than best
        """
        pass


# =============================================================================
# EARLY STOPPING STRATEGY INTERFACE
# =============================================================================

class EarlyStoppingStrategy(ABC):
    """
    Interface for early stopping logic.
    
    Any early stopping strategy must implement:
        - should_stop(): Checks if training should stop
        - update(): Updates internal state with new metric
        - reset(): Resets the strategy state
    
    Example implementations:
        - PatientEarlyStoppingStrategy
        - NoEarlyStoppingStrategy
    """
    
    @abstractmethod
    def should_stop(self) -> bool:
        """
        Determine if training should stop.
        
        Returns:
            True if training should stop
        """
        pass
    
    @abstractmethod
    def update(self, current_metric: float) -> bool:
        """
        Update internal state with new metric value.
        
        Args:
            current_metric: Current evaluation metric
            
        Returns:
            True if this is a new best metric
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the early stopping state."""
        pass
    
    @property
    @abstractmethod
    def best_metric(self) -> float:
        """Return the best metric value seen so far."""
        pass
    
    @property
    @abstractmethod
    def patience_counter(self) -> int:
        """Return current patience counter value."""
        pass


# =============================================================================
# CHECKPOINT STRATEGY INTERFACE
# =============================================================================

class CheckpointStrategy(ABC):
    """
    Interface for model checkpointing.
    
    Any checkpoint strategy must implement:
        - save(): Saves a checkpoint
        - load(): Loads a checkpoint
        - best_path(): Returns path to best checkpoint
    
    
    """
    
    @abstractmethod
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        filename: str
    ) -> str:
        """
        Save a model checkpoint.
        
        Args:
            model: The model to save
            optimizer: The optimizer state to save
            epoch: Current epoch number
            metrics: Current evaluation metrics
            filename: Checkpoint filename
            
        Returns:
            Full path to saved checkpoint
        """
        pass
    
    @abstractmethod
    def load(
        self,
        model: nn.Module,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            model: The model to load weights into
            path: Path to checkpoint file
            optimizer: Optional optimizer to load state into
            
        Returns:
            Dictionary with loaded checkpoint data
        """
        pass
    
    @abstractmethod
    def best_path(self) -> Optional[str]:
        """Return the path to the best checkpoint, if any."""
        pass


# =============================================================================
# TRAINER INTERFACE
# =============================================================================

class TrainerInterface(ABC):
    """
    Interface for model trainers.
    
    Any trainer must implement:
        - train(): Runs the training loop
        - evaluate(): Evaluates the model
        - predict(): Generates predictions
        - save_model(): Saves the model
        - load_model(): Loads the model
    
    Example implementations:
        - DiacritizationTrainer
        - ClassificationTrainer
    """
    
    @abstractmethod
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            
        Returns:
            Dictionary of training history
        """
        pass
    
    @abstractmethod
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            dataloader: Evaluation data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def predict(self, dataloader: DataLoader) -> List[Any]:
        """
        Generate predictions.
        
        Args:
            dataloader: Data loader for prediction
            
        Returns:
            List of predictions
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path: Path to load the model from
        """
        pass
