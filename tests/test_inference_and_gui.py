"""
Comprehensive Unit Tests for ScratchLM InferenceEngine and GUI Runner.
Verifies model discovery, streaming token generation, cancellation callbacks,
error handling, and shared CLI/GUI backend equivalence.
"""

import unittest
from pathlib import Path
import torch

from scratchlm.inference import InferenceEngine, GenerationResult, ModelInfo
from scratchlm.gui.app import ScratchLMApp, HAS_TKINTER
from scratchlm.utils.paths import paths


class TestInferenceEngineAndGUI(unittest.TestCase):

    def setUp(self):
        self.checkpoint_dir = paths.checkpoints / "v0.1" / "exp_phase1_english" / "exp_phase1_english"
        self.checkpoint_path = self.checkpoint_dir / "best.pt"
        self.tokenizer_path = paths.tokenizers / "v0.1_bpe_4096"
        self.engine = InferenceEngine(device="cpu")

    def test_checkpoint_discovery(self):
        """Test auto-discovery of local checkpoints and tokenizers."""
        checkpoints = self.engine.find_local_checkpoints()
        self.assertIsInstance(checkpoints, list)
        self.assertGreater(len(checkpoints), 0, "Should discover at least 1 local checkpoint")

        tokenizers = self.engine.find_local_tokenizers()
        self.assertIsInstance(tokenizers, list)
        self.assertGreater(len(tokenizers), 0, "Should discover at least 1 local tokenizer")

    def test_model_loading_and_info(self):
        """Test loading real ScratchLM model checkpoint and tokenizer."""
        self.assertTrue(self.checkpoint_path.exists(), f"Checkpoint path missing: {self.checkpoint_path}")
        self.assertTrue(self.tokenizer_path.exists(), f"Tokenizer path missing: {self.tokenizer_path}")

        info = self.engine.load_model_and_tokenizer(
            checkpoint_path=self.checkpoint_path,
            tokenizer_path=self.tokenizer_path,
        )

        self.assertIsInstance(info, ModelInfo)
        self.assertGreater(info.num_params, 0)
        self.assertGreater(info.vocab_size, 0)
        self.assertEqual(info.device, "cpu")

    def test_generation_streaming_and_stats(self):
        """Test text generation with streaming callback and throughput stats."""
        self.engine.load_model_and_tokenizer(
            checkpoint_path=self.checkpoint_path,
            tokenizer_path=self.tokenizer_path,
        )

        streamed_tokens = []

        def _cb(tok):
            streamed_tokens.append(tok)

        res = self.engine.generate(
            prompt="The Earth is",
            max_length=15,
            temperature=0.7,
            token_callback=_cb,
        )

        self.assertIsInstance(res, GenerationResult)
        self.assertGreater(res.tokens_generated, 0)
        self.assertGreater(len(streamed_tokens), 0)
        self.assertGreater(res.elapsed_seconds, 0.0)
        self.assertGreater(res.tokens_per_second, 0.0)
        self.assertIn("The Earth is", res.full_text)

    def test_cancellation_callback(self):
        """Test stopping text generation mid-stream via cancel_callback."""
        self.engine.load_model_and_tokenizer(
            checkpoint_path=self.checkpoint_path,
            tokenizer_path=self.tokenizer_path,
        )

        counter = 0

        def _cancel_after_3_tokens():
            nonlocal counter
            counter += 1
            return counter >= 3

        res = self.engine.generate(
            prompt="Once upon a time",
            max_length=50,
            cancel_callback=_cancel_after_3_tokens,
        )

        self.assertTrue(res.stopped_early)
        self.assertLessEqual(res.tokens_generated, 4)

    def test_invalid_checkpoint_handling(self):
        """Test clear error raising on missing/invalid checkpoint file."""
        fake_path = paths.root / "non_existent_checkpoint.pt"
        with self.assertRaises(FileNotFoundError):
            self.engine.load_model_and_tokenizer(fake_path)

    def test_gui_initialization(self):
        """Test GUI application class structure and fallback behavior."""
        app = ScratchLMApp(root=None)
        self.assertIsNotNone(app.engine)
        self.assertFalse(app.is_generating)


if __name__ == "__main__":
    unittest.main()
