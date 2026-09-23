"""
Evaluation suite for ScratchLM.

Provides a comprehensive suite for evaluating language models.
"""

import json
from typing import List, Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from scratchlm.model import ScratchLM
from scratchlm.tokenizer import BaseTokenizer
from scratchlm.data import TokenizedDataset
from scratchlm.utils import paths, get_logger

from .metrics import (
    PerplexityMetric,
    AccuracyMetric,
    LossMetric,
    RepetitionMetric,
    DiversityMetric,
    CoherenceMetric,
)
from .tasks import (
    NextTokenPredictionTask,
    SentenceCompletionTask,
    GrammarTask,
    VocabularyTask,
    CoherenceTask,
)


@dataclass
class EvaluationResult:
    """Results from an evaluation run."""
    metrics: Dict[str, float]
    task_results: Dict[str, Dict[str, float]]
    samples: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'metrics': self.metrics,
            'task_results': self.task_results,
            'samples': self.samples,
        }
    
    def save(self, path: Union[str, Path]):
        """Save results to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'EvaluationResult':
        """Load results from a file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            metrics=data.get('metrics', {}),
            task_results=data.get('task_results', {}),
            samples=data.get('samples'),
        )


class EvaluationSuite:
    """
    Comprehensive evaluation suite for language models.
    
    Runs multiple metrics and tasks to evaluate model performance.
    """
    
    def __init__(
        self,
        model: Optional[ScratchLM] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the evaluation suite.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer to use
            device: Device to run evaluation on
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if device is None:
            device = "cpu"
        self.device = torch.device(device)
        
        # Set up logging
        self.logger = get_logger("scratchlm.evaluation")
        
        # Initialize metrics
        self.metrics = {
            'perplexity': PerplexityMetric(),
            'loss': LossMetric(),
        }
        
        # Initialize tasks
        self.tasks = {
            'next_token': NextTokenPredictionTask(),
            'sentence_completion': SentenceCompletionTask(),
            'grammar': GrammarTask(),
            'vocabulary': VocabularyTask(),
            'coherence': CoherenceTask(),
        }
    
    def set_model(self, model: ScratchLM):
        """Set the model to evaluate."""
        self.model = model
        self.model.to(self.device)
    
    def set_tokenizer(self, tokenizer: BaseTokenizer):
        """Set the tokenizer to use."""
        self.tokenizer = tokenizer
    
    def evaluate_dataset(
        self,
        dataset: TokenizedDataset,
        split: str = "test",
        batch_size: int = 4,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            split: Name of the split
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model must be set first")
        
        self.logger.info(f"Evaluating on {split} dataset...")
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()
        
        # Evaluate
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)
        
        with torch.no_grad():
            for input_ids, labels, attention_mask in loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                
                loss = loss_fn(logits_flat, labels_flat)
                
                # Update metrics
                total_loss += loss.item() * labels.size(1)
                total_tokens += labels.size(1)
        
        # Compute final metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        
        metrics = {
            f'{split}_loss': avg_loss,
            f'{split}_perplexity': float(torch.exp(torch.tensor(avg_loss)).item()),
        }
        
        self.logger.info(f"{split} | Loss: {avg_loss:.4f} | Perplexity: {metrics[f'{split}_perplexity']:.2f}")
        
        return metrics
    
    def evaluate_tasks(
        self,
        num_samples: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model on various tasks.
        
        Args:
            num_samples: Number of samples to generate for each task
            
        Returns:
            Dictionary of task results
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set first")
        
        self.logger.info("Evaluating on tasks...")
        
        task_results = {}
        samples = []
        
        self.model.eval()
        
        for task_name, task in self.tasks.items():
            self.logger.info(f"  Running {task_name} task...")
            
            # Run task
            result = task.evaluate(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                num_samples=num_samples,
            )
            
            task_results[task_name] = result.metrics
            
            # Collect samples
            if result.samples:
                samples.extend(result.samples)
        
        return task_results, samples
    
    def run_full_evaluation(
        self,
        dataset: Optional[TokenizedDataset] = None,
        num_task_samples: int = 10,
        batch_size: int = 4,
    ) -> EvaluationResult:
        """
        Run a full evaluation.
        
        Args:
            dataset: Dataset to evaluate on (for perplexity/loss)
            num_task_samples: Number of samples for task evaluation
            batch_size: Batch size for dataset evaluation
            
        Returns:
            Complete evaluation results
        """
        metrics = {}
        task_results = {}
        samples = []
        
        # Evaluate on dataset
        if dataset is not None:
            metrics = self.evaluate_dataset(
                dataset,
                split="test",
                batch_size=batch_size,
            )
        
        # Evaluate on tasks
        task_results, samples = self.evaluate_tasks(num_task_samples)
        
        return EvaluationResult(
            metrics=metrics,
            task_results=task_results,
            samples=samples,
        )
    
    def compare_models(
        self,
        model_a: ScratchLM,
        model_b: ScratchLM,
        dataset: TokenizedDataset,
        tokenizer: BaseTokenizer,
        batch_size: int = 4,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare two models on the same dataset.
        
        Args:
            model_a: First model
            model_b: Second model
            dataset: Dataset to evaluate on
            tokenizer: Tokenizer to use
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with comparison results
        """
        # Evaluate model A
        self.set_model(model_a)
        self.set_tokenizer(tokenizer)
        results_a = self.evaluate_dataset(dataset, split="a", batch_size=batch_size)
        
        # Evaluate model B
        self.set_model(model_b)
        self.set_tokenizer(tokenizer)
        results_b = self.evaluate_dataset(dataset, split="b", batch_size=batch_size)
        
        # Compare
        comparison = {}
        for metric_name in results_a:
            comparison[metric_name] = {
                'model_a': results_a[metric_name],
                'model_b': results_b[metric_name],
                'diff': results_b[metric_name] - results_a[metric_name],
            }
        
        return comparison


if __name__ == "__main__":
    print("Testing EvaluationSuite...")
    
    # This is a minimal test - a full test would require actual data
    from scratchlm.model.config import get_config_preset
    from scratchlm.model import ScratchLM
    from scratchlm.tokenizer import BPETokenizer
    from scratchlm.data import TokenizedDataset
    
    # Create a small model
    model_config, _, _ = get_config_preset("tiny")
    model = ScratchLM(model_config)
    
    # Create a dummy tokenizer
    tokenizer = BPETokenizer(vocab_size=100)
    
    # Create evaluation suite
    suite = EvaluationSuite(
        model=model,
        tokenizer=tokenizer,
    )
    
    print("Evaluation suite created successfully!")
