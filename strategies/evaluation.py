import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from interfaces.model import EvaluationStrategy
from utils.constants import ArabicDiacritics


# =============================================================================
# DIACRITIZATION EVALUATION STRATEGY
# =============================================================================

class DiacritizationEvalStrategy(EvaluationStrategy):
    """
    Comprehensive evaluation for Arabic diacritization.
    
    Metrics computed:
    - DER (Diacritic Error Rate): Main metric for diacritization
    - WER (Word Error Rate): Percentage of words with at least one error
    - Accuracy: Overall token-level accuracy
    - Per-diacritic F1 scores
    - Confusion matrix statistics
    
    Use when: Standard diacritization evaluation (recommended)
    """
    
    def __init__(
        self,
        ignore_none_class: bool = True,
        compute_per_class: bool = True
    ):
        """
        Args:
            ignore_none_class: Whether to ignore NONE class in some metrics
            compute_per_class: Whether to compute per-diacritic metrics
        """
        self.ignore_none_class = ignore_none_class
        self.compute_per_class = compute_per_class
        self.diacritic_names = [d.name for d in ArabicDiacritics]
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Evaluate model and return comprehensive metrics."""
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_words_predictions = []
        all_words_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Get model outputs
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Get predictions
                if 'predictions' in outputs:
                    # CRF model returns direct predictions
                    predictions = outputs['predictions']
                else:
                    # Non-CRF model, get from logits
                    logits = outputs['logits']
                    predictions = torch.argmax(logits, dim=-1)
                
                # Extract valid positions (not padding)
                valid_mask = attention_mask.bool()
                
                # Flatten and filter
                batch_size, seq_len = predictions.shape
                for i in range(batch_size):
                    valid_length = valid_mask[i].sum().item()
                    
                    pred_seq = predictions[i][:valid_length].cpu().numpy()
                    target_seq = labels[i][:valid_length].cpu().numpy()
                    
                    # Filter out special tokens (assuming they're marked as -100)
                    valid_indices = target_seq != -100
                    
                    if valid_indices.any():
                        pred_valid = pred_seq[valid_indices]
                        target_valid = target_seq[valid_indices]
                        
                        all_predictions.extend(pred_valid)
                        all_targets.extend(target_valid)
                        
                        # Store word-level sequences for WER calculation
                        all_words_predictions.append(pred_valid)
                        all_words_targets.append(target_valid)
        
        # Compute metrics
        metrics = self._compute_metrics(
            all_predictions,
            all_targets,
            all_words_predictions,
            all_words_targets
        )
        
        return metrics
    
    def _compute_metrics(
        self,
        predictions: List[int],
        targets: List[int],
        word_predictions: List[np.ndarray],
        word_targets: List[np.ndarray]
    ) -> Dict[str, float]:
        """Compute all evaluation metrics."""
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics = {}
        
        # 1. Diacritic Error Rate (DER) - Main metric
        der = self._compute_der(predictions, targets)
        metrics['der'] = der
        
        # 2. Word Error Rate (WER)
        wer = self._compute_wer(word_predictions, word_targets)
        metrics['wer'] = wer
        
        # 3. Overall Accuracy
        accuracy = accuracy_score(targets, predictions)
        metrics['accuracy'] = accuracy
        
        # 4. Accuracy ignoring NONE class
        if self.ignore_none_class:
            non_none_mask = targets != ArabicDiacritics.NONE.value
            if non_none_mask.any():
                acc_no_none = accuracy_score(
                    targets[non_none_mask], 
                    predictions[non_none_mask]
                )
                metrics['accuracy_no_none'] = acc_no_none
        
        # 5. Per-class metrics
        if self.compute_per_class:
            per_class_metrics = self._compute_per_class_metrics(predictions, targets)
            metrics.update(per_class_metrics)
        
        # 6. Macro averages
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average='macro', zero_division=0
        )
        metrics.update({
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1
        })
        
        # 7. Weighted averages
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        metrics.update({
            'weighted_precision': precision,
            'weighted_recall': recall,
            'weighted_f1': f1
        })
        
        return metrics
    
    def _compute_der(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute Diacritic Error Rate.
        
        DER = (Number of incorrect diacritics) / (Total diacritics)
        Lower is better.
        """
        incorrect = np.sum(predictions != targets)
        total = len(targets)
        return incorrect / total if total > 0 else 0.0
    
    def _compute_wer(
        self, 
        word_predictions: List[np.ndarray], 
        word_targets: List[np.ndarray]
    ) -> float:
        """
        Compute Word Error Rate.
        
        WER = (Words with at least one error) / (Total words)
        """
        if not word_predictions:
            return 0.0
        
        incorrect_words = 0
        total_words = len(word_predictions)
        
        for pred_word, target_word in zip(word_predictions, word_targets):
            if not np.array_equal(pred_word, target_word):
                incorrect_words += 1
        
        return incorrect_words / total_words
    
    def _compute_per_class_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute precision, recall, F1 for each diacritic class."""
        per_class_metrics = {}
        
        # Get per-class scores
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        for i, diacritic_name in enumerate(self.diacritic_names):
            if i < len(precision):
                per_class_metrics.update({
                    f'{diacritic_name.lower()}_precision': precision[i],
                    f'{diacritic_name.lower()}_recall': recall[i],
                    f'{diacritic_name.lower()}_f1': f1[i],
                    f'{diacritic_name.lower()}_support': support[i]
                })
        
        return per_class_metrics
    
    def primary_metric(self) -> str:
        """Return the primary metric name (DER for diacritization)."""
        return "der"
    
    def is_better(self, current: float, best: float) -> bool:
        """For DER, lower is better."""
        return current < best
    
    def name(self) -> str:
        return "Diacritization_Eval"


# =============================================================================
# SIMPLE ACCURACY EVALUATION STRATEGY
# =============================================================================

class SimpleAccuracyEvalStrategy(EvaluationStrategy):
    """
    Simple accuracy-based evaluation.
    
    Use when: Quick evaluation or debugging
    Only computes basic accuracy metrics.
    """
    
    def __init__(self, ignore_padding: bool = True):
        """
        Args:
            ignore_padding: Whether to ignore padding tokens (-100)
        """
        self.ignore_padding = ignore_padding
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Evaluate model and return simple accuracy."""
        model.eval()
        
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Get predictions
                if 'predictions' in outputs:
                    predictions = outputs['predictions']
                else:
                    logits = outputs['logits']
                    predictions = torch.argmax(logits, dim=-1)
                
                # Calculate accuracy
                if self.ignore_padding:
                    valid_mask = (labels != -100) & attention_mask.bool()
                else:
                    valid_mask = attention_mask.bool()
                
                if valid_mask.any():
                    correct = (predictions == labels) & valid_mask
                    total_correct += correct.sum().item()
                    total_tokens += valid_mask.sum().item()
        
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'total_tokens': total_tokens,
            'correct_tokens': total_correct
        }
    
    def primary_metric(self) -> str:
        """Return accuracy as primary metric."""
        return "accuracy"
    
    def is_better(self, current: float, best: float) -> bool:
        """For accuracy, higher is better."""
        return current > best
    
    def name(self) -> str:
        return "Simple_Accuracy"


# =============================================================================
# F1 SCORE EVALUATION STRATEGY
# =============================================================================

class F1EvalStrategy(EvaluationStrategy):
    """
    F1-score based evaluation.
    
    Use when: F1 is the primary concern (balanced precision/recall)
    """
    
    def __init__(
        self,
        average: str = 'macro',
        ignore_none_class: bool = True
    ):
        """
        Args:
            average: 'macro', 'micro', 'weighted'
            ignore_none_class: Whether to exclude NONE class from F1
        """
        self.average = average
        self.ignore_none_class = ignore_none_class
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Evaluate model and return F1-based metrics."""
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Get predictions
                if 'predictions' in outputs:
                    predictions = outputs['predictions']
                else:
                    logits = outputs['logits']
                    predictions = torch.argmax(logits, dim=-1)
                
                # Extract valid predictions
                valid_mask = (labels != -100) & attention_mask.bool()
                
                if valid_mask.any():
                    pred_valid = predictions[valid_mask].cpu().numpy()
                    target_valid = labels[valid_mask].cpu().numpy()
                    
                    all_predictions.extend(pred_valid)
                    all_targets.extend(target_valid)
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Compute F1 scores
        metrics = {}
        
        # Filter out NONE class if requested
        if self.ignore_none_class:
            non_none_mask = targets != ArabicDiacritics.NONE.value
            if non_none_mask.any():
                predictions_filtered = predictions[non_none_mask]
                targets_filtered = targets[non_none_mask]
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    targets_filtered, predictions_filtered, 
                    average=self.average, zero_division=0
                )
                
                metrics.update({
                    f'f1_{self.average}_no_none': f1,
                    f'precision_{self.average}_no_none': precision,
                    f'recall_{self.average}_no_none': recall
                })
        
        # All classes
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average=self.average, zero_division=0
        )
        
        metrics.update({
            f'f1_{self.average}': f1,
            f'precision_{self.average}': precision,
            f'recall_{self.average}': recall,
            'accuracy': accuracy_score(targets, predictions)
        })
        
        return metrics
    
    def primary_metric(self) -> str:
        """Return F1 as primary metric."""
        suffix = '_no_none' if self.ignore_none_class else ''
        return f'f1_{self.average}{suffix}'
    
    def is_better(self, current: float, best: float) -> bool:
        """For F1, higher is better."""
        return current > best
    
    def name(self) -> str:
        return f"F1_{self.average}"


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_evaluation_strategy(name: str, **kwargs) -> EvaluationStrategy:
    """
    Get evaluation strategy by name.
    
    Args:
        name: 'diacritization', 'accuracy', or 'f1'
        **kwargs: Additional arguments for the strategy
        
    Usage:
        strategy = get_evaluation_strategy('diacritization')
        strategy = get_evaluation_strategy('f1', average='weighted')
    """
    strategies = {
        'diacritization': DiacritizationEvalStrategy,
        'accuracy': SimpleAccuracyEvalStrategy,
        'f1': F1EvalStrategy,
    }
    
    name_lower = name.lower()
    
    if name_lower not in strategies:
        available = ', '.join(strategies.keys())
        raise ValueError(f"Unknown evaluation: '{name}'. Available: {available}")
    
    return strategies[name_lower](**kwargs)


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def print_evaluation_report(metrics: Dict[str, float]) -> None:
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: Dictionary of metrics from evaluation
    """
    print("\n" + "="*60)
    print("ARABIC DIACRITIZATION EVALUATION REPORT")
    print("="*60)
    
    # Main metrics
    main_metrics = ['der', 'wer', 'accuracy', 'macro_f1']
    print("\n📊 MAIN METRICS:")
    print("-" * 30)
    
    for metric in main_metrics:
        if metric in metrics:
            value = metrics[metric]
            if metric in ['der', 'wer']:
                print(f"{metric.upper():12}: {value:.4f} (lower is better)")
            else:
                print(f"{metric.upper():12}: {value:.4f}")
    
    # Per-class metrics (if available)
    per_class_f1 = {k: v for k, v in metrics.items() if k.endswith('_f1') and not k.startswith(('macro', 'weighted'))}
    
    if per_class_f1:
        print(f"\n📈 PER-DIACRITIC F1 SCORES:")
        print("-" * 30)
        
        for diacritic, f1 in sorted(per_class_f1.items()):
            diacritic_name = diacritic.replace('_f1', '').upper()
            print(f"{diacritic_name:15}: {f1:.4f}")
    
    print("\n" + "="*60)


def compare_models(
    results: Dict[str, Dict[str, float]], 
    primary_metric: str = 'der'
) -> None:
    """
    Compare multiple model results.
    
    Args:
        results: Dict of {model_name: metrics_dict}
        primary_metric: Metric to use for ranking
    """
    print(f"\n🏆 MODEL COMPARISON (by {primary_metric.upper()}):")
    print("-" * 50)
    
    # Sort by primary metric
    is_lower_better = primary_metric in ['der', 'wer', 'loss']
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get(primary_metric, float('inf')),
        reverse=not is_lower_better
    )
    
    for rank, (model_name, metrics) in enumerate(sorted_results, 1):
        metric_value = metrics.get(primary_metric, 'N/A')
        print(f"{rank}. {model_name:20} - {primary_metric}: {metric_value}")