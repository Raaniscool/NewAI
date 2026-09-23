"""
Byte Pair Encoding (BPE) Tokenizer for ScratchLM.

Implements BPE tokenization from scratch, including training.
Optimized with inverted-index sequence tracking for fast sub-second BPE training.
"""

import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path

from .base import BaseTokenizer
from scratchlm.utils.paths import paths


class BPETokenizer(BaseTokenizer):
    """
    Byte Pair Encoding (BPE) Tokenizer.
    
    Implements BPE tokenization with fast inverted-index pair tracking.
    """
    
    def __init__(
        self,
        vocab_size: int = 4096,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        min_frequency: int = 2,
    ):
        super().__init__()
        
        self._target_vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        self._special_tokens = {
            pad_token: 0,
            unk_token: 1,
            bos_token: 2,
            eos_token: 3,
        }
        
        self._vocab = self._special_tokens.copy()
        self._id_to_token = {v: k for k, v in self._special_tokens.items()}
        
        self._merges: Dict[Tuple[int, int], int] = {}
        self._reverse_merges: Dict[int, Tuple[int, int]] = {}
        
        self._base_vocab: Dict[str, int] = {}
        self._next_id = len(self._special_tokens)
        self._pattern = re.compile(r"''|'|[^\w' ]|")
    
    def train(
        self,
        texts: List[str],
        num_merges: Optional[int] = None,
        show_progress: bool = True,
    ):
        """Train the BPE tokenizer on a list of texts using inverted-index pair acceleration."""
        if num_merges is None:
            num_merges = self._target_vocab_size - len(self._special_tokens)
        
        if num_merges <= 0:
            return
        
        self._build_base_vocab(texts)
        
        for i in range(num_merges):
            if len(self._vocab) >= self._target_vocab_size:
                break
            
            most_frequent_pair, pair_count = self._find_most_frequent_pair()
            
            if most_frequent_pair is None or pair_count < self.min_frequency:
                if show_progress:
                    print(f"No more frequent pairs above min_frequency={self.min_frequency}")
                break
            
            self._merge_pair(most_frequent_pair)
            
            if show_progress and (i + 1) % 1000 == 0:
                print(f"Merge {i + 1}/{num_merges} | Vocab size: {len(self._vocab)} | "
                      f"Merged pair: {most_frequent_pair} -> {self._id_to_token[self._next_id - 1]}")
        
        if show_progress:
            print(f"\nBPE Tokenizer Training complete! Final vocabulary size: {len(self._vocab)}")
    
    def _build_base_vocab(self, texts: List[str]):
        """Build the base vocabulary and set up inverted index pair tracking."""
        chars = set()
        for text in texts:
            base_tokens = self._split_into_base_tokens(text)
            chars.update(base_tokens)
        
        for char in sorted(chars):
            if char not in self._vocab:
                self._vocab[char] = self._next_id
                self._id_to_token[self._next_id] = char
                self._next_id += 1
        
        self._base_vocab = {k: v for k, v in self._vocab.items() if k not in self._special_tokens}
        
        self._sequences = []
        self._pair_counts = Counter()
        self._pair_to_seqs = defaultdict(set)
        
        for seq_idx, text in enumerate(texts):
            base_tokens = self._split_into_base_tokens(text)
            token_ids = [self._vocab.get(token, self.unk_token_id) for token in base_tokens]
            self._sequences.append(token_ids)
            
            for j in range(len(token_ids) - 1):
                pair = (token_ids[j], token_ids[j + 1])
                self._pair_counts[pair] += 1
                self._pair_to_seqs[pair].add(seq_idx)
    
    def _split_into_base_tokens(self, text: str) -> List[str]:
        """Split text into base character tokens."""
        tokens = []
        for match in self._pattern.finditer(text):
            token = match.group()
            if token:
                tokens.append(token)
        
        remaining = self._pattern.sub('', text)
        if remaining:
            tokens.extend(list(remaining))
        
        return [t for t in tokens if t.strip() or t in ' \t\n']
    
    def _find_most_frequent_pair(self) -> Tuple[Optional[Tuple[int, int]], int]:
        if not self._pair_counts:
            return None, 0
        most_common = self._pair_counts.most_common(1)
        if not most_common:
            return None, 0
        return most_common[0]
    
    def _merge_pair(self, pair: Tuple[int, int]):
        """Merge a pair of token IDs and update affected sequences via inverted index."""
        left_id, right_id = pair
        new_id = self._next_id
        self._next_id += 1
        
        left_token = self._id_to_token[left_id]
        right_token = self._id_to_token[right_id]
        new_token = left_token + right_token
        
        self._vocab[new_token] = new_id
        self._id_to_token[new_id] = new_token
        
        self._merges[pair] = new_id
        self._reverse_merges[new_id] = pair
        
        # Zero out the merged pair count
        del self._pair_counts[pair]
        
        # Retrieve only sequences containing this pair
        affected_seq_indices = list(self._pair_to_seqs.get(pair, set()))
        if pair in self._pair_to_seqs:
            del self._pair_to_seqs[pair]
            
        for seq_idx in affected_seq_indices:
            seq = self._sequences[seq_idx]
            new_seq = []
            i = 0
            n = len(seq)
            
            while i < n:
                if i + 1 < n and seq[i] == left_id and seq[i + 1] == right_id:
                    # Remove old adjacent pairs from counts & index
                    if i > 0:
                        prev_pair = (seq[i - 1], left_id)
                        self._pair_counts[prev_pair] -= 1
                        if self._pair_counts[prev_pair] <= 0:
                            del self._pair_counts[prev_pair]
                            
                    if i + 2 < n:
                        next_pair = (right_id, seq[i + 2])
                        self._pair_counts[next_pair] -= 1
                        if self._pair_counts[next_pair] <= 0:
                            del self._pair_counts[next_pair]
                            
                    new_seq.append(new_id)
                    
                    # Add new adjacent pairs to counts & index
                    if len(new_seq) > 1:
                        p_left = (new_seq[-2], new_id)
                        self._pair_counts[p_left] += 1
                        self._pair_to_seqs[p_left].add(seq_idx)
                        
                    if i + 2 < n:
                        p_right = (new_id, seq[i + 2])
                        self._pair_counts[p_right] += 1
                        self._pair_to_seqs[p_right].add(seq_idx)
                        
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
                    
            self._sequences[seq_idx] = new_seq

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text into token IDs using learned BPE merges."""
        base_tokens = self._split_into_base_tokens(text)
        token_ids = [self._vocab.get(token, self.unk_token_id) for token in base_tokens]
        token_ids = self._apply_merges(token_ids)
        
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
        return token_ids
    
    def _apply_merges(self, token_ids: List[int]) -> List[int]:
        """Fast rank-based BPE merge application."""
        if len(token_ids) < 2 or not self._merges:
            return token_ids
        
        tokens = list(token_ids)
        while len(tokens) >= 2:
            min_rank = float('inf')
            best_pair = None
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self._merges:
                    rank = self._merges[pair]
                    if rank < min_rank:
                        min_rank = rank
                        best_pair = pair
            
            if best_pair is None:
                break
                
            left_id, right_id = best_pair
            new_id = self._merges[best_pair]
            
            new_tokens = []
            i = 0
            n = len(tokens)
            while i < n:
                if i + 1 < n and tokens[i] == left_id and tokens[i + 1] == right_id:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in self._special_tokens.values():
                continue
            token = self._id_to_token.get(token_id, self.unk_token)
            tokens.append(token)
        return "".join(tokens)
    
    def tokenize(self, text: str) -> List[str]:
        token_ids = self.encode(text, add_special_tokens=False)
        return [self._id_to_token.get(id, self.unk_token) for id in token_ids]
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        token_ids = []
        for token in tokens:
            if token in self._vocab:
                token_ids.append(self._vocab[token])
            else:
                token_ids.append(self.unk_token_id)
        return token_ids
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._id_to_token.get(id, self.unk_token) for id in ids]
    
    def save(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        vocab_path = directory / "vocab.json"
        vocab_data = {
            "vocab": self._vocab,
            "id_to_token": self._id_to_token,
            "special_tokens": self._special_tokens,
        }
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2)
        
        merges_path = directory / "merges.txt"
        with open(merges_path, 'w', encoding='utf-8') as f:
            for (left, right), new_id in sorted(self._merges.items(), key=lambda x: x[1]):
                left_token = self._id_to_token[left]
                right_token = self._id_to_token[right]
                new_token = self._id_to_token[new_id]
                f.write(f"{left_token} {right_token} -> {new_token} ({left} {right} -> {new_id})\n")
        
        config_path = directory / "config.json"
        config = {
            "vocab_size": self._target_vocab_size,
            "min_frequency": self.min_frequency,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "tokenizer_type": "bpe",
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, directory: Path) -> 'BPETokenizer':
        directory = Path(directory)
        config_path = directory / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        tokenizer = cls(
            vocab_size=config.get("vocab_size", 4096),
            min_frequency=config.get("min_frequency", 2),
            pad_token=config.get("pad_token", "<pad>"),
            unk_token=config.get("unk_token", "<unk>"),
            bos_token=config.get("bos_token", "<s>"),
            eos_token=config.get("eos_token", "</s>"),
        )
        
        vocab_path = directory / "vocab.json"
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        tokenizer._vocab = vocab_data["vocab"]
        tokenizer._id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}
        tokenizer._special_tokens = vocab_data["special_tokens"]
        tokenizer._next_id = max(tokenizer._vocab.values()) + 1
        
        merges_path = directory / "merges.txt"
        if merges_path.exists():
            with open(merges_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("->")
                    if len(parts) < 2:
                        continue
                    left_right = parts[0].strip().split()
                    new_token_part = parts[1].strip().split()[0]
                    if len(left_right) == 2:
                        left_token, right_token = left_right
                        new_token = new_token_part
                        left_id = tokenizer._vocab.get(left_token)
                        right_id = tokenizer._vocab.get(right_token)
                        new_id = tokenizer._vocab.get(new_token)
                        if left_id is not None and right_id is not None and new_id is not None:
                            tokenizer._merges[(left_id, right_id)] = new_id
                            tokenizer._reverse_merges[new_id] = (left_id, right_id)
        
        return tokenizer

    def get_merge_rules(self) -> List[Tuple[str, str, str]]:
        rules = []
        for (left_id, right_id), new_id in sorted(self._merges.items(), key=lambda x: x[1]):
            left = self._id_to_token[left_id]
            right = self._id_to_token[right_id]
            new = self._id_to_token[new_id]
            rules.append((left, right, new))
        return rules


def train_bpe_tokenizer(
    texts: List[str],
    vocab_size: int = 4096,
    min_frequency: int = 2,
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
    bos_token: str = "<s>",
    eos_token: str = "</s>",
    show_progress: bool = True,
) -> BPETokenizer:
    tokenizer = BPETokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        pad_token=pad_token,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
    )
    tokenizer.train(texts, show_progress=show_progress)
    return tokenizer
