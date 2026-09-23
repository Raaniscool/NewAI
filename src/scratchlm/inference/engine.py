"""
Shared Inference Engine for ScratchLM.

Provides a unified backend for both CLI (generate.py) and GUI (desktop runner):
- Checkpoint and tokenizer auto-discovery
- Robust checkpoint and configuration loading
- Token-by-token streaming text generation
- Real-time cancellation support
- Latency and token throughput statistics
"""

import time
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

from scratchlm.model import ScratchLM
from scratchlm.model.config import TransformerConfig, get_config_preset
from scratchlm.tokenizer import BaseTokenizer, load_tokenizer
from scratchlm.utils.paths import paths


@dataclass
class ModelInfo:
    """Metadata describing a loaded ScratchLM model."""
    checkpoint_path: str
    tokenizer_path: str
    preset_name: str
    version: str
    experiment_id: str
    num_params: int
    num_trainable_params: int
    vocab_size: int
    context_length: int
    d_model: int
    n_layers: int
    n_heads: int
    device: str


@dataclass
class GenerationResult:
    """Output metrics and continuation text resulting from inference."""
    prompt: str
    generated_continuation: str
    full_text: str
    tokens_generated: int
    elapsed_seconds: float
    tokens_per_second: float
    stopped_early: bool = False


class InferenceEngine:
    """
    Unified Inference Backend shared identically by CLI and GUI runners.
    """

    def __init__(self, device: str = "cpu"):
        self.device_str = device
        self.device = torch.device(device)
        self.model: Optional[ScratchLM] = None
        self.tokenizer: Optional[BaseTokenizer] = None
        self.current_model_info: Optional[ModelInfo] = None

    def find_local_checkpoints(self, search_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Search local directory tree for ScratchLM checkpoint files and metadata."""
        if search_dir is None:
            search_dir = paths.checkpoints

        checkpoints = []
        if not search_dir.exists():
            return checkpoints

        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.pt') and not file.startswith('.'):
                    pt_path = Path(root) / file
                    json_path = pt_path.with_suffix('.json')
                    
                    meta = {
                        "path": str(pt_path),
                        "filename": pt_path.name,
                        "rel_path": str(pt_path.relative_to(search_dir.parent if search_dir.parent else search_dir)),
                        "mtime": pt_path.stat().st_mtime,
                        "size_mb": round(pt_path.stat().st_size / (1024 * 1024), 2),
                    }
                    
                    if json_path.exists():
                        try:
                            import json
                            with open(json_path, 'r', encoding='utf-8') as f:
                                meta.update(json.load(f))
                        except Exception:
                            pass
                            
                    checkpoints.append(meta)

        checkpoints.sort(key=lambda x: x.get("mtime", 0), reverse=True)
        return checkpoints

    def find_local_tokenizers(self, search_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Search local directories for trained tokenizer definitions."""
        if search_dir is None:
            search_dir = paths.tokenizers

        tokenizers = []
        if not search_dir.exists():
            return tokenizers

        for root, _, files in os.walk(search_dir):
            if "config.json" in files or "vocab.json" in files:
                tok_path = Path(root)
                meta = {
                    "path": str(tok_path),
                    "name": tok_path.name,
                    "mtime": tok_path.stat().st_mtime,
                }
                tokenizers.append(meta)

        tokenizers.sort(key=lambda x: x.get("mtime", 0), reverse=True)
        return tokenizers

    def load_model_and_tokenizer(
        self,
        checkpoint_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
    ) -> ModelInfo:
        """Load ScratchLM model weights and matching subword tokenizer."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Load PyTorch state dict
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint format in {checkpoint_path}")

        # Extract config
        config = None
        if "config" in checkpoint and isinstance(checkpoint["config"], dict) and checkpoint["config"]:
            config = TransformerConfig.from_dict(checkpoint["config"])
        else:
            exp_id = checkpoint.get("experiment_id", "exp001")
            exp_file = paths.configs / "experiments" / exp_id / "model.yaml"
            if exp_file.exists():
                config = TransformerConfig.load(exp_file)
            else:
                config, _, _ = get_config_preset("tiny")

        # Instantiate model architecture
        model = ScratchLM(config)
        
        # Determine state dict key
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        self.model = model

        # Auto-resolve tokenizer if omitted
        if tokenizer_path is None:
            candidate_paths = [
                checkpoint_path.parent / "tokenizer",
                checkpoint_path.parent.parent / "tokenizer",
                paths.tokenizers / f"{config.version}_bpe_{config.vocab_size}",
            ]
            for candidate in candidate_paths:
                if candidate.exists():
                    tokenizer_path = candidate
                    break

        if tokenizer_path is None or not Path(tokenizer_path).exists():
            # Check local tokenizers
            locs = self.find_local_tokenizers()
            if locs:
                tokenizer_path = locs[0]["path"]
            else:
                raise FileNotFoundError("No matching tokenizer directory found. Please specify tokenizer path.")

        tokenizer_path = Path(tokenizer_path)
        tokenizer = load_tokenizer(tokenizer_path)
        self.tokenizer = tokenizer

        # Build ModelInfo summary
        self.current_model_info = ModelInfo(
            checkpoint_path=str(checkpoint_path.resolve()),
            tokenizer_path=str(tokenizer_path.resolve()),
            preset_name=getattr(config, "preset", "custom"),
            version=getattr(config, "version", "v0.1"),
            experiment_id=checkpoint.get("experiment_id", "exp001"),
            num_params=model.get_num_params(),
            num_trainable_params=model.get_num_trainable_params(),
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            device=str(self.device_str),
        )

        return self.current_model_info

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        greedy: bool = False,
        token_callback: Optional[Callable[[str], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> GenerationResult:
        """Autoregressively generate text with real-time token streaming and cancellation."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded prior to generation.")

        # Encode prompt
        if prompt.strip():
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        else:
            input_ids = [self.tokenizer.bos_token_id]

        input_tensor = torch.tensor([input_ids], device=self.device)
        output_ids = input_ids.copy()
        generated_token_ids = []

        start_time = time.time()
        stopped_early = False

        self.model.eval()
        with torch.no_grad():
            curr_tensor = input_tensor
            
            for _ in range(max_length):
                if cancel_callback is not None and cancel_callback():
                    stopped_early = True
                    break

                # Predict next token logits
                logits = self.model.get_next_token_logits(curr_tensor)

                if temperature > 0 and temperature != 1.0:
                    logits = logits / temperature

                if top_k is not None and top_k > 0:
                    top_logits, _ = torch.topk(logits, top_k, dim=-1)
                    mask = logits >= top_logits[:, -1:]
                    logits = logits.masked_fill(~mask, float('-inf'))

                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    mask = cum_probs >= top_p
                    cutoff = mask.int().argmax(dim=-1, keepdim=True)
                    
                    mask_filt = torch.zeros_like(logits, dtype=torch.bool)
                    for b in range(logits.size(0)):
                        mask_filt[b, sorted_indices[b, :cutoff[b] + 1]] = True
                    logits = logits.masked_fill(~mask_filt, float('-inf'))

                if greedy or temperature == 0:
                    next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)

                next_id_val = next_token_id.item()
                output_ids.append(next_id_val)
                generated_token_ids.append(next_id_val)

                # Decode piece for callback
                if token_callback is not None:
                    decoded_token = self.tokenizer.decode([next_id_val], skip_special_tokens=True)
                    token_callback(decoded_token)

                # Stop if EOS or PAD
                if next_id_val in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id):
                    break

                curr_tensor = torch.cat([curr_tensor, next_token_id], dim=1)

        elapsed = max(0.001, time.time() - start_time)
        full_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        if prompt.strip():
            # Extract generated portion
            prompt_decoded = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            if full_text.startswith(prompt_decoded):
                continuation = full_text[len(prompt_decoded):]
            else:
                continuation = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        else:
            continuation = full_text

        tok_count = len(generated_token_ids)
        tok_per_sec = tok_count / elapsed

        return GenerationResult(
            prompt=prompt,
            generated_continuation=continuation,
            full_text=full_text,
            tokens_generated=tok_count,
            elapsed_seconds=round(elapsed, 3),
            tokens_per_second=round(tok_per_sec, 2),
            stopped_early=stopped_early,
        )
