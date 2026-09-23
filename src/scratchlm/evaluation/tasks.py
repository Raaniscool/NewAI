"""
Evaluation tasks for ScratchLM.

Implements specific tasks for evaluating language model capabilities.
"""

import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch

from scratchlm.model import ScratchLM
from scratchlm.tokenizer import BaseTokenizer
from .metrics import MetricResult


@dataclass
class TaskResult:
    """Result from an evaluation task."""
    metrics: Dict[str, float]
    samples: Optional[List[Dict]] = None


class BaseTask:
    """Base class for all evaluation tasks."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    def evaluate(
        self,
        model: ScratchLM,
        tokenizer: BaseTokenizer,
        device: torch.device,
        num_samples: int = 10,
    ) -> TaskResult:
        """Evaluate the task."""
        raise NotImplementedError


class NextTokenPredictionTask(BaseTask):
    """
    Next token prediction task.
    
    Evaluates the model's ability to predict the next token in a sequence.
    """
    
    def __init__(self):
        super().__init__(
            name="next_token_prediction",
            description="Predict the next token in a sequence",
        )
        
        # Simple prompts for testing
        self.prompts = [
            "The cat sat on the",
            "It was a dark and stormy",
            "Once upon a time there was a",
            "The quick brown fox jumps over the",
            "To be or not to",
            "In the beginning",
            "All's well that",
            "A stitch in time",
        ]
    
    def evaluate(
        self,
        model: ScratchLM,
        tokenizer: BaseTokenizer,
        device: torch.device,
        num_samples: int = 10,
    ) -> TaskResult:
        """
        Evaluate next token prediction.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer to use
            device: Device to run on
            num_samples: Number of samples to evaluate
            
        Returns:
            TaskResult with accuracy and samples
        """
        model.eval()
        
        correct = 0
        total = 0
        samples = []
        
        # Select prompts
        prompts = self.prompts[:num_samples]
        
        for prompt in prompts:
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            
            # Get expected next token (we'll use a simple heuristic)
            # In a real evaluation, we'd have ground truth
            # For now, we'll just check if the model produces reasonable continuations
            
            # Convert to tensor
            input_tensor = torch.tensor([input_ids], device=device)
            
            # Get next token logits
            with torch.no_grad():
                next_logits = model.get_next_token_logits(input_tensor)
            
            # Get predicted token
            predicted_token = torch.argmax(next_logits, dim=-1).item()
            predicted_text = tokenizer.decode([predicted_token])
            
            # For this simple test, we'll just count non-pad, non-unk predictions
            if predicted_token != tokenizer.pad_token_id and predicted_token != tokenizer.unk_token_id:
                correct += 1
            
            total += 1
            
            # Save sample
            samples.append({
                'prompt': prompt,
                'predicted_token': predicted_text,
                'token_id': predicted_token,
            })
        
        accuracy = correct / total if total > 0 else 0.0
        
        return TaskResult(
            metrics={'accuracy': accuracy},
            samples=samples,
        )


class SentenceCompletionTask(BaseTask):
    """
    Sentence completion task.
    
    Evaluates the model's ability to complete sentences coherently.
    """
    
    def __init__(self):
        super().__init__(
            name="sentence_completion",
            description="Complete partial sentences",
        )
        
        self.prompts = [
            "The weather today is",
            "My favorite color is",
            "When I grow up I want to be a",
            "The capital of France is",
            "A dog is a type of",
            "The sky is",
            "Water boils at",
            "The opposite of hot is",
        ]
    
    def evaluate(
        self,
        model: ScratchLM,
        tokenizer: BaseTokenizer,
        device: torch.device,
        num_samples: int = 10,
    ) -> TaskResult:
        """
        Evaluate sentence completion.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer to use
            device: Device to run on
            num_samples: Number of samples to evaluate
            
        Returns:
            TaskResult with coherence scores and samples
        """
        model.eval()
        
        samples = []
        coherence_scores = []
        
        # Select prompts
        prompts = self.prompts[:num_samples]
        
        for prompt in prompts:
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], device=device)
            
            # Generate continuation
            with torch.no_grad():
                generated = model.generate(
                    input_tensor,
                    max_length=min(50, model.config.context_length),
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode
            generated_text = tokenizer.decode(generated[0].tolist())
            
            # Simple coherence check: does it continue the sentence?
            # This is a placeholder - a real implementation would use more sophisticated metrics
            score = 1.0 if len(generated_text.split()) > len(prompt.split()) else 0.0
            coherence_scores.append(score)
            
            samples.append({
                'prompt': prompt,
                'completion': generated_text,
                'coherence_score': score,
            })
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        
        return TaskResult(
            metrics={'coherence': avg_coherence},
            samples=samples,
        )


class GrammarTask(BaseTask):
    """
    Grammar task.
    
    Evaluates the model's understanding of basic grammar.
    """
    
    def __init__(self):
        super().__init__(
            name="grammar",
            description="Test basic grammar understanding",
        )
        
        # Prompts that test grammar
        self.prompts = [
            "I am",
            "You are",
            "He is",
            "She was",
            "We have",
            "They were",
            "The dog",
            "A cat",
        ]
    
    def evaluate(
        self,
        model: ScratchLM,
        tokenizer: BaseTokenizer,
        device: torch.device,
        num_samples: int = 10,
    ) -> TaskResult:
        """
        Evaluate grammar understanding.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer to use
            device: Device to run on
            num_samples: Number of samples to evaluate
            
        Returns:
            TaskResult with grammar scores and samples
        """
        model.eval()
        
        samples = []
        grammar_scores = []
        
        # Select prompts
        prompts = self.prompts[:num_samples]
        
        for prompt in prompts:
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], device=device)
            
            # Generate
            with torch.no_grad():
                generated = model.generate(
                    input_tensor,
                    max_length=min(20, model.config.context_length),
                    temperature=0.5,
                    do_sample=False,  # Greedy for more predictable output
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode
            generated_text = tokenizer.decode(generated[0].tolist())
            
            # Simple grammar check: does it produce a grammatically reasonable continuation?
            # This is a placeholder - a real implementation would use a grammar checker
            # For now, we'll just check if it doesn't repeat the prompt
            words = generated_text.split()
            prompt_words = prompt.split()
            
            # Check if it adds new words
            if len(words) > len(prompt_words):
                score = 1.0
            else:
                score = 0.0
            
            grammar_scores.append(score)
            
            samples.append({
                'prompt': prompt,
                'completion': generated_text,
                'grammar_score': score,
            })
        
        avg_score = sum(grammar_scores) / len(grammar_scores) if grammar_scores else 0.0
        
        return TaskResult(
            metrics={'grammar_score': avg_score},
            samples=samples,
        )


class VocabularyTask(BaseTask):
    """
    Vocabulary task.
    
    Evaluates the model's vocabulary knowledge.
    """
    
    def __init__(self):
        super().__init__(
            name="vocabulary",
            description="Test vocabulary knowledge",
        )
        
        self.prompts = [
            "An apple is a type of",
            "A car has",
            "The color of the sky is",
            "A book contains",
            "A computer can",
            "Music is",
            "A tree has",
            "The sun is",
        ]
    
    def evaluate(
        self,
        model: ScratchLM,
        tokenizer: BaseTokenizer,
        device: torch.device,
        num_samples: int = 10,
    ) -> TaskResult:
        """
        Evaluate vocabulary knowledge.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer to use
            device: Device to run on
            num_samples: Number of samples to evaluate
            
        Returns:
            TaskResult with vocabulary scores and samples
        """
        model.eval()
        
        samples = []
        diversity_scores = []
        
        # Select prompts
        prompts = self.prompts[:num_samples]
        
        for prompt in prompts:
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], device=device)
            
            # Generate
            with torch.no_grad():
                generated = model.generate(
                    input_tensor,
                    max_length=min(30, model.config.context_length),
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode
            generated_text = tokenizer.decode(generated[0].tolist())
            
            # Calculate diversity score (type-token ratio)
            words = generated_text.split()
            unique_words = set(words)
            ttr = len(unique_words) / len(words) if words else 0.0
            diversity_scores.append(ttr)
            
            samples.append({
                'prompt': prompt,
                'completion': generated_text,
                'diversity_score': ttr,
            })
        
        avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0
        
        return TaskResult(
            metrics={'diversity': avg_diversity},
            samples=samples,
        )


class CoherenceTask(BaseTask):
    """
    Coherence task.
    
    Evaluates the overall coherence of generated text.
    """
    
    def __init__(self):
        super().__init__(
            name="coherence",
            description="Test overall text coherence",
        )
        
        self.prompts = [
            "Tell me about a time when you",
            "The most important thing in life is",
            "If I could travel anywhere I would go to",
            "My favorite memory is",
            "The world would be better if",
            "When I was a child",
            "The future will bring",
            "A good friend is someone who",
        ]
    
    def evaluate(
        self,
        model: ScratchLM,
        tokenizer: BaseTokenizer,
        device: torch.device,
        num_samples: int = 10,
    ) -> TaskResult:
        """
        Evaluate text coherence.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer to use
            device: Device to run on
            num_samples: Number of samples to evaluate
            
        Returns:
            TaskResult with coherence scores and samples
        """
        model.eval()
        
        samples = []
        coherence_scores = []
        
        # Select prompts
        prompts = self.prompts[:num_samples]
        
        for prompt in prompts:
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_tensor = torch.tensor([input_ids], device=device)
            
            # Generate
            with torch.no_grad():
                generated = model.generate(
                    input_tensor,
                    max_length=min(50, model.config.context_length),
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode
            generated_text = tokenizer.decode(generated[0].tolist())
            
            # Simple coherence check: count sentences and check for punctuation
            sentences = [s for s in generated_text.split('.') if s.strip()]
            sentences += [s for s in generated_text.split('!') if s.strip()]
            sentences += [s for s in generated_text.split('?') if s.strip()]
            
            # Score based on number of sentences and average length
            num_sentences = len(sentences)
            avg_length = sum(len(s.split()) for s in sentences) / num_sentences if num_sentences > 0 else 0
            
            # Simple heuristic: more sentences with reasonable length = more coherent
            score = min(1.0, num_sentences * 0.2 + avg_length * 0.05)
            coherence_scores.append(score)
            
            samples.append({
                'prompt': prompt,
                'completion': generated_text,
                'coherence_score': score,
            })
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        
        return TaskResult(
            metrics={'coherence': avg_coherence},
            samples=samples,
        )


if __name__ == "__main__":
    print("Testing evaluation tasks...")
    
    # This is a minimal test - a full test would require actual trained models
    from scratchlm.model.config import get_config_preset
    from scratchlm.model import ScratchLM
    from scratchlm.tokenizer import BPETokenizer
    
    # Create a small model
    model_config, _, _ = get_config_preset("tiny")
    model = ScratchLM(model_config)
    
    # Create a dummy tokenizer
    tokenizer = BPETokenizer(vocab_size=100)
    
    # Test tasks
    tasks = [
        NextTokenPredictionTask(),
        SentenceCompletionTask(),
        GrammarTask(),
        VocabularyTask(),
        CoherenceTask(),
    ]
    
    device = torch.device("cpu")
    
    for task in tasks:
        print(f"\n=== {task.name} ===")
        try:
            result = task.evaluate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                num_samples=2,
            )
            print(f"Metrics: {result.metrics}")
            if result.samples:
                print(f"Samples: {len(result.samples)}")
                for sample in result.samples[:1]:
                    print(f"  {sample}")
        except Exception as e:
            print(f"Error: {e}")
