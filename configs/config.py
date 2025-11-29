"""
Configuration Classes for Arabic Diacritization

Customizable configurations for models and training.
Values are set when creating instances, with sensible defaults.

Usage:
    # Use defaults
    config = ModelConfig()
    
    # Customize values
    config = ModelConfig(model_name="aubmindlab/bert-large-arabertv2", dropout=0.2)
    
    # From dictionary
    config = ModelConfig.from_dict({'dropout': 0.3})
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from utils.constants import ArabicDiactrics, ModelType, NUM_DIACRITICS


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """
    Model architecture configuration.
    
    Attributes:
        model_name: Pre-trained model name/path
        model_type: Type of model (arabert, camelbert, bilstm)
        num_labels: Number of diacritic classes
        dropout: Dropout rate for regularization
        max_length: Maximum sequence length
        use_crf: Whether to use CRF layer
        hidden_size: Hidden size for BiLSTM (ignored for transformers)
        num_layers: Number of layers for BiLSTM (ignored for transformers)
        
        """
    
    # Pre-trained model identifier
    model_name: Optional[str] = None
    
    # Model architecture type
    model_type: Optional[ModelType] = None
    
    # Number of output classes (diacritics)
    num_labels: Optional[int] = None
    
    # Dropout probability
    dropout: Optional[float] = None
    
    # Maximum input sequence length
    max_length: Optional[int] = None
    
    # Use CRF layer for sequence labeling
    use_crf: Optional[bool] = None
    
    # BiLSTM specific: hidden layer size
    hidden_size: Optional[int] = None
    
    # BiLSTM specific: number of LSTM layers
    num_layers: Optional[int] = None
    
    # BiLSTM specific: bidirectional
    bidirectional: Optional[bool] = None
    
    # Embedding dimension for BiLSTM
    embedding_dim: Optional[int] = None
    
    # Vocabulary size for BiLSTM
    vocab_size: Optional[int] = None
    
    def __post_init__(self):
        """Apply defaults for None values."""
        defaults = self._get_defaults()
        for key, value in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, value)
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default values based on model type."""
        # Determine model type for defaults
        model_type = self.model_type or ModelType.ARABERT
        
        base_defaults = {
            'num_labels': NUM_DIACRITICS,
            'dropout': 0.1,
            'max_length': 512,
            'use_crf': True,
        }
        
        if model_type == ModelType.ARABERT:
            base_defaults.update({
                'model_name': "aubmindlab/bert-base-arabertv2",
                'model_type': ModelType.ARABERT,
            })
        elif model_type == ModelType.CAMELBERT:
            base_defaults.update({
                'model_name': "CAMeL-Lab/bert-base-arabic-camelbert-mix",
                'model_type': ModelType.CAMELBERT,
            })
        elif model_type == ModelType.BILSTM:
            base_defaults.update({
                'model_name': None,
                'model_type': ModelType.BILSTM,
                'hidden_size': 256,
                'num_layers': 2,
                'bidirectional': True,
                'embedding_dim': 300,
                'vocab_size': 50000,
                'max_length': 256,
            })
        
        return base_defaults
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        # Convert model_type string to enum if needed
        if 'model_type' in config_dict and isinstance(config_dict['model_type'], str):
            config_dict['model_type'] = ModelType(config_dict['model_type'])
        
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {
            'model_name': self.model_name,
            'model_type': self.model_type.value if self.model_type else None,
            'num_labels': self.num_labels,
            'dropout': self.dropout,
            'max_length': self.max_length,
            'use_crf': self.use_crf,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'embedding_dim': self.embedding_dim,
            'vocab_size': self.vocab_size,
        }
        return {k: v for k, v in result.items() if v is not None}
    
    def update(self, **kwargs) -> 'ModelConfig':
        """Create new config with updated values."""
        current = self.to_dict()
        current.update(kwargs)
        return ModelConfig.from_dict(current)
    
    def __repr__(self) -> str:
        fields = self.to_dict()
        items = ', '.join(f"{k}={v!r}" for k, v in fields.items())
        return f"ModelConfig({items})"


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Training hyperparameters configuration.
    
    Attributes:
        num_epochs: Number of training epochs
        learning_rate: Base learning rate
        batch_size: Training batch size
        weight_decay: Weight decay for regularization
        gradient_clip: Maximum gradient norm
        early_stopping_patience: Epochs to wait for improvement
        warmup_ratio: Fraction of steps for LR warmup
        eval_batch_size: Evaluation batch size
        save_steps: Save checkpoint every N steps (None = end of epoch)
        logging_steps: Log metrics every N steps
    
    Usage:
        # Use defaults
        config = TrainingConfig()
        
        # Custom values
        config = TrainingConfig(
            num_epochs=20,
            learning_rate=1e-5,
            batch_size=16
        )
    """
    
    # Number of training epochs
    num_epochs: Optional[int] = None
    
    # Learning rate
    learning_rate: Optional[float] = None
    
    # Batch size for training
    batch_size: Optional[int] = None
    
    # Weight decay coefficient
    weight_decay: Optional[float] = None
    
    # Gradient clipping threshold
    gradient_clip: Optional[float] = None
    
    # Early stopping patience (epochs)
    early_stopping_patience: Optional[int] = None
    
    # Warmup ratio (fraction of total steps)
    warmup_ratio: Optional[float] = None
    
    # Evaluation batch size
    eval_batch_size: Optional[int] = None
    
    # Save checkpoint every N steps
    save_steps: Optional[int] = None
    
    # Log every N steps
    logging_steps: Optional[int] = None
    
    # Random seed for reproducibility
    seed: Optional[int] = None
    
    # Mixed precision training
    fp16: Optional[bool] = None
    
    def __post_init__(self):
        """Apply defaults for None values."""
        defaults = self._get_defaults()
        for key, value in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, value)
    
    @staticmethod
    def _get_defaults() -> Dict[str, Any]:
        """Get default values."""
        return {
            'num_epochs': 10,
            'learning_rate': 2e-5,
            'batch_size': 8,
            'weight_decay': 0.01,
            'gradient_clip': 1.0,
            'early_stopping_patience': 3,
            'warmup_ratio': 0.1,
            'eval_batch_size': 16,
            'save_steps': None,
            'logging_steps': 50,
            'seed': 42,
            'fp16': False,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'gradient_clip': self.gradient_clip,
            'early_stopping_patience': self.early_stopping_patience,
            'warmup_ratio': self.warmup_ratio,
            'eval_batch_size': self.eval_batch_size,
            'save_steps': self.save_steps,
            'logging_steps': self.logging_steps,
            'seed': self.seed,
            'fp16': self.fp16,
        }
    
    def update(self, **kwargs) -> 'TrainingConfig':
        """Create new config with updated values."""
        current = self.to_dict()
        current.update(kwargs)
        return TrainingConfig.from_dict(current)
    
    def __repr__(self) -> str:
        fields = self.to_dict()
        items = ', '.join(f"{k}={v!r}" for k, v in fields.items())
        return f"TrainingConfig({items})"


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

class ConfigPresets:
    """
    Pre-defined configurations for common use cases.
    
    Usage:
        model_cfg, train_cfg = ConfigPresets.arabert_default()
        model_cfg, train_cfg = ConfigPresets.camelbert_high_accuracy()
        model_cfg, train_cfg = ConfigPresets.bilstm_default()
    """
    
    # -------------------------------------------------------------------------
    # AraBERT Presets
    # -------------------------------------------------------------------------
    
    @staticmethod
    def arabert_default() -> Tuple[ModelConfig, TrainingConfig]:
        """Default AraBERT configuration."""
        return (
            ModelConfig(model_type=ModelType.ARABERT),
            TrainingConfig()
        )
    
    @staticmethod
    def arabert_fast() -> Tuple[ModelConfig, TrainingConfig]:
        """Fast training with AraBERT (quick experiments)."""
        return (
            ModelConfig(model_type=ModelType.ARABERT, use_crf=False),
            TrainingConfig(num_epochs=3, batch_size=16, learning_rate=5e-5)
        )
    
    @staticmethod
    def arabert_high_accuracy() -> Tuple[ModelConfig, TrainingConfig]:
        """High accuracy AraBERT (slower but better)."""
        return (
            ModelConfig(model_type=ModelType.ARABERT, dropout=0.05),
            TrainingConfig(
                num_epochs=20,
                learning_rate=1e-5,
                batch_size=4,
                early_stopping_patience=5
            )
        )
    
    # -------------------------------------------------------------------------
    # CAMeLBERT Presets
    # -------------------------------------------------------------------------
    
    @staticmethod
    def camelbert_default() -> Tuple[ModelConfig, TrainingConfig]:
        """Default CAMeLBERT configuration."""
        return (
            ModelConfig(model_type=ModelType.CAMELBERT),
            TrainingConfig()
        )
    
    @staticmethod
    def camelbert_msa() -> Tuple[ModelConfig, TrainingConfig]:
        """CAMeLBERT for Modern Standard Arabic."""
        return (
            ModelConfig(
                model_name="CAMeL-Lab/bert-base-arabic-camelbert-msa",
                model_type=ModelType.CAMELBERT
            ),
            TrainingConfig(learning_rate=2e-5)
        )
    
    @staticmethod
    def camelbert_high_accuracy() -> Tuple[ModelConfig, TrainingConfig]:
        """High accuracy CAMeLBERT."""
        return (
            ModelConfig(model_type=ModelType.CAMELBERT, dropout=0.05),
            TrainingConfig(
                num_epochs=20,
                learning_rate=1e-5,
                batch_size=4,
                early_stopping_patience=5
            )
        )
    
    # -------------------------------------------------------------------------
    # BiLSTM Presets
    # -------------------------------------------------------------------------
    
    @staticmethod
    def bilstm_default() -> Tuple[ModelConfig, TrainingConfig]:
        """Default BiLSTM configuration."""
        return (
            ModelConfig(model_type=ModelType.BILSTM),
            TrainingConfig(
                learning_rate=1e-3,
                batch_size=32,
                num_epochs=30
            )
        )
    
    @staticmethod
    def bilstm_large() -> Tuple[ModelConfig, TrainingConfig]:
        """Larger BiLSTM for better accuracy."""
        return (
            ModelConfig(
                model_type=ModelType.BILSTM,
                hidden_size=512,
                num_layers=3,
                dropout=0.3
            ),
            TrainingConfig(
                learning_rate=5e-4,
                batch_size=32,
                num_epochs=50
            )
        )
    
    @staticmethod
    def bilstm_fast() -> Tuple[ModelConfig, TrainingConfig]:
        """Fast BiLSTM for quick experiments."""
        return (
            ModelConfig(
                model_type=ModelType.BILSTM,
                hidden_size=128,
                num_layers=1,
                use_crf=False
            ),
            TrainingConfig(
                learning_rate=1e-3,
                batch_size=64,
                num_epochs=10
            )
        )
    
    # -------------------------------------------------------------------------
    # Low Resource Presets
    # -------------------------------------------------------------------------
    
    @staticmethod
    def low_memory() -> Tuple[ModelConfig, TrainingConfig]:
        """Low memory configuration for limited GPU."""
        return (
            ModelConfig(
                model_type=ModelType.ARABERT,
                max_length=256
            ),
            TrainingConfig(
                batch_size=2,
                eval_batch_size=4,
                gradient_clip=0.5,
                fp16=True
            )
        )
    
    @staticmethod
    def cpu_only() -> Tuple[ModelConfig, TrainingConfig]:
        """Configuration for CPU-only training."""
        return (
            ModelConfig(
                model_type=ModelType.BILSTM,
                hidden_size=128,
                num_layers=1,
                max_length=128
            ),
            TrainingConfig(
                batch_size=16,
                num_epochs=20,
                fp16=False
            )
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_preset(name: str) -> Tuple[ModelConfig, TrainingConfig]:
    """
    Get configuration preset by name.
    
    Args:
        name: Preset name
        
    Available presets:
        - 'arabert_default', 'arabert_fast', 'arabert_high_accuracy'
        - 'camelbert_default', 'camelbert_msa', 'camelbert_high_accuracy'
        - 'bilstm_default', 'bilstm_large', 'bilstm_fast'
        - 'low_memory', 'cpu_only'
        
    Usage:
        model_cfg, train_cfg = get_preset('arabert_default')
    """
    presets = {
        'arabert_default': ConfigPresets.arabert_default,
        'arabert_fast': ConfigPresets.arabert_fast,
        'arabert_high_accuracy': ConfigPresets.arabert_high_accuracy,
        'camelbert_default': ConfigPresets.camelbert_default,
        'camelbert_msa': ConfigPresets.camelbert_msa,
        'camelbert_high_accuracy': ConfigPresets.camelbert_high_accuracy,
        'bilstm_default': ConfigPresets.bilstm_default,
        'bilstm_large': ConfigPresets.bilstm_large,
        'bilstm_fast': ConfigPresets.bilstm_fast,
        'low_memory': ConfigPresets.low_memory,
        'cpu_only': ConfigPresets.cpu_only,
    }
    
    name_lower = name.lower()
    
    if name_lower not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset: '{name}'. Available: {available}")
    
    return presets[name_lower]()


def list_presets() -> list:
    """Return list of available preset names."""
    return [
        'arabert_default', 'arabert_fast', 'arabert_high_accuracy',
        'camelbert_default', 'camelbert_msa', 'camelbert_high_accuracy',
        'bilstm_default', 'bilstm_large', 'bilstm_fast',
        'low_memory', 'cpu_only',
    ]