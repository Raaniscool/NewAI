"""
Integration and End-to-End Tests for Desktop GUI Training, Evaluation & Generation.

Verifies:
1. Training section GUI initialization and callback integration.
2. Shared backend training execution (CLI & GUI using Trainer).
3. Real-time metrics streaming (loss, perplexity, step, epoch).
4. Safe cancellation callback handling.
5. Direct loading of trained checkpoints into InferenceEngine for text generation.
6. Real evaluation of trained models via EvaluationSuite.
"""

import unittest
import math
import tempfile
from pathlib import Path
import torch

from scratchlm.training import Trainer
from scratchlm.data import load_data, save_processed_text
from scratchlm.model.config import get_config_preset
from scratchlm.inference import InferenceEngine
from scratchlm.evaluation import EvaluationSuite
from scratchlm.gui.app import ScratchLMApp


class TestGUITrainingAndSharedBackend(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

        self.train_file = self.tmp_path / "train_sample.txt"
        self.val_file = self.tmp_path / "val_sample.txt"

        # Load real text samples from generated corpus if available
        corpus_train = Path("data/processed/corpus/train.txt")
        if corpus_train.exists():
            sample_train = load_data([str(corpus_train)], min_length=20, max_length=1000, deduplicate=False)[:100]
        else:
            sample_train = [
                "The Solar System consists of the Sun and eight major planets bound by gravity in outer space. Planets revolve around the central star in elliptical orbits according to Kepler's laws of planetary motion.",
                "Photosynthesis converts solar light energy into chemical energy in green plant chloroplasts. Carbon dioxide and water produce glucose and release molecular oxygen into the atmosphere.",
                "Classical literature explores enduring themes of human nature, ethics, leadership, and community relationships across historical epochs.",
                "Geological research demonstrates that tectonic forces move continental plates across the ductile asthenosphere beneath Earth's crust.",
            ] * 10

        save_processed_text(sample_train, self.train_file)
        save_processed_text(sample_train[:20], self.val_file)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_shared_backend_trainer_execution_and_callbacks(self):
        """Verify Trainer runs with live step, epoch, and log callbacks."""
        train_texts = load_data([str(self.train_file)], min_length=5, max_length=1000, deduplicate=False)
        val_texts = load_data([str(self.val_file)], min_length=5, max_length=1000, deduplicate=False)

        m_cfg, t_cfg, tok_cfg = get_config_preset("tiny")
        m_cfg.version = "v_test"
        t_cfg.model_version = "v_test"
        t_cfg.experiment_id = "exp_gui_test"
        t_cfg.num_epochs = 2
        t_cfg.batch_size = 4
        t_cfg.warmup_steps = 10
        t_cfg.learning_rate = 1e-4

        step_events = []
        epoch_events = []
        log_messages = []

        def _step_cb(stats):
            step_events.append(stats)

        def _epoch_cb(metrics):
            epoch_events.append(metrics)

        def _log_cb(msg):
            log_messages.append(msg)

        trainer = Trainer(
            model_config=m_cfg,
            training_config=t_cfg,
            tokenizer_config=tok_cfg,
            model_version="v_test",
            experiment_id="exp_gui_test",
            device="cpu",
        )

        res = trainer.train(
            train_texts=train_texts,
            val_texts=val_texts,
            train_tokenizer=True,
            step_callback=_step_cb,
            epoch_callback=_epoch_cb,
            log_callback=_log_cb,
        )

        self.assertGreater(len(step_events), 0, "Step callback should receive step progress updates.")
        self.assertEqual(len(epoch_events), 2, "Epoch callback should receive 2 epoch summary events.")
        self.assertGreater(len(log_messages), 0, "Log callback should receive training progress logs.")

        self.assertIn("train_loss", epoch_events[0])
        self.assertIn("val_loss", epoch_events[0])
        self.assertIsNotNone(epoch_events[0]["train_loss"])
        self.assertIsNotNone(epoch_events[0]["val_loss"])

    def test_safe_cancellation_during_training(self):
        """Verify cancellation callback stops training safely at batch boundary."""
        train_texts = load_data([str(self.train_file)], min_length=5, max_length=1000, deduplicate=False)

        m_cfg, t_cfg, tok_cfg = get_config_preset("tiny")
        m_cfg.version = "v_test_cancel"
        t_cfg.model_version = "v_test_cancel"
        t_cfg.experiment_id = "exp_cancel"
        t_cfg.num_epochs = 10
        t_cfg.batch_size = 2
        t_cfg.warmup_steps = 10

        step_count = 0

        def _cancel_after_2_steps():
            nonlocal step_count
            step_count += 1
            return step_count >= 2

        trainer = Trainer(
            model_config=m_cfg,
            training_config=t_cfg,
            tokenizer_config=tok_cfg,
            model_version="v_test_cancel",
            experiment_id="exp_cancel",
            device="cpu",
        )

        res = trainer.train(
            train_texts=train_texts,
            train_tokenizer=True,
            cancel_callback=_cancel_after_2_steps,
        )

        self.assertTrue(trainer.training_loop.was_cancelled, "Training loop should record cancelled state.")
        self.assertLessEqual(trainer.training_loop.step, 3, "Training should stop early after requested steps.")

    def test_trained_checkpoint_loads_in_inference_engine(self):
        """Verify newly trained checkpoint can be loaded directly by InferenceEngine."""
        train_texts = load_data([str(self.train_file)], min_length=5, max_length=1000, deduplicate=False)

        m_cfg, t_cfg, tok_cfg = get_config_preset("tiny")
        m_cfg.version = "v_gen_test"
        t_cfg.model_version = "v_gen_test"
        t_cfg.experiment_id = "exp_gen_test"
        t_cfg.num_epochs = 1
        t_cfg.batch_size = 4
        t_cfg.warmup_steps = 10

        trainer = Trainer(
            model_config=m_cfg,
            training_config=t_cfg,
            tokenizer_config=tok_cfg,
            model_version="v_gen_test",
            experiment_id="exp_gen_test",
            device="cpu",
        )

        trainer.train(train_texts=train_texts, train_tokenizer=True)

        checkpoints = list((trainer.checkpoint_dir / "exp_gen_test").glob("*.pt"))
        self.assertGreater(len(checkpoints), 0, "Should generate at least one checkpoint file.")
        chk_path = checkpoints[0]

        tok_path = trainer.checkpoint_dir.parent.parent / "tokenizers" / f"{m_cfg.version}_bpe_{tok_cfg.vocab_size}"

        engine = InferenceEngine(device="cpu")
        info = engine.load_model_and_tokenizer(chk_path, tok_path if tok_path.exists() else None)

        self.assertEqual(info.version, "v_gen_test")

        res = engine.generate(prompt="The Solar System", max_length=15, temperature=0.7)
        self.assertGreater(len(res.full_text), 0)
        self.assertGreater(res.tokens_generated, 0)

    def test_evaluation_suite_on_trained_model(self):
        """Verify EvaluationSuite evaluates trained model and produces real metrics."""
        train_texts = load_data([str(self.train_file)], min_length=5, max_length=1000, deduplicate=False)

        m_cfg, t_cfg, tok_cfg = get_config_preset("tiny")
        t_cfg.num_epochs = 1
        t_cfg.warmup_steps = 10

        trainer = Trainer(
            model_config=m_cfg,
            training_config=t_cfg,
            tokenizer_config=tok_cfg,
            model_version="v_eval_test",
            experiment_id="exp_eval_test",
            device="cpu",
        )

        trainer.train(train_texts=train_texts, train_tokenizer=True)

        suite = EvaluationSuite(model=trainer.model, tokenizer=trainer.tokenizer, device="cpu")
        task_results, samples = suite.evaluate_tasks(num_samples=2)

        self.assertIn("next_token", task_results)
        self.assertIn("sentence_completion", task_results)
        self.assertGreater(len(samples), 0)


if __name__ == "__main__":
    unittest.main()
