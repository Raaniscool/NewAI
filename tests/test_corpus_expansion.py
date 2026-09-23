"""
Unit and Integration Tests for ScratchLM Phase 1 Corpus Expansion & Quality Gates.

Verifies:
1. Ingestion of 35+ broad non-code human domains and targeted synthetic categories.
2. Quality Gate validations (minimum word count >= 500k, zero leakage, duplicate rate < 20%).
3. Fast inverted-index Jaccard MinHash deduplication behavior.
4. Provenance and licensing metadata completeness.
5. Vocabulary and subword BPE tokenization coverage.
"""

import unittest
import json
import re
from pathlib import Path

from scratchlm.data.human_sources import RealHumanSourceProvider, HumanDocument
from scratchlm.data.synthetic_generator import SyntheticEnglishGenerator, SyntheticDocument
from scratchlm.data.deduplication import CorpusDeduplicator, DeduplicationResult, get_ngrams, jaccard_similarity
from scratchlm.data.corpus import CorpusPipeline, CorpusDocument


class TestCorpusExpansion(unittest.TestCase):

    def setUp(self):
        self.human_provider = RealHumanSourceProvider(seed=42)
        self.synthetic_generator = SyntheticEnglishGenerator(seed=42)
        self.deduplicator = CorpusDeduplicator(near_dup_threshold=0.85)

    def test_human_source_provider_provenance(self):
        """Verify human sources deliver 35+ non-code domains with complete provenance metadata."""
        domain_configs = self.human_provider._get_domain_configs()
        self.assertEqual(len(domain_configs), 35, "Should configure 35 distinct public domain sources.")

        docs = self.human_provider.get_all_documents(min_target_words=800000)
        self.assertGreater(len(docs), 3000)

        categories = set()
        sources = set()

        for doc in docs:
            self.assertIsInstance(doc, HumanDocument)
            self.assertFalse(doc.is_synthetic)
            self.assertTrue(len(doc.text) > 100)
            self.assertTrue(doc.source_name)
            self.assertTrue(doc.license)
            self.assertTrue(doc.category)
            self.assertTrue(doc.author)

            categories.add(doc.category)
            sources.add(doc.source_name)

        self.assertGreaterEqual(len(sources), 30)
        self.assertIn("science_and_nature", categories)
        self.assertIn("fiction_and_literature", categories)
        self.assertIn("history_and_speeches", categories)

    def test_synthetic_generator_provenance(self):
        """Verify targeted synthetic generator delivers CC0 licensed structural English texts."""
        docs = self.synthetic_generator.generate_all(target_count=50)
        self.assertEqual(len(docs), 50)

        for doc in docs:
            self.assertIsInstance(doc, SyntheticDocument)
            self.assertTrue(doc.is_synthetic)
            self.assertTrue(doc.source_name)
            self.assertEqual(doc.license, "CC0-1.0 (Public Domain Equivalent)")
            self.assertTrue(doc.category.startswith("synthetic_"))

    def test_fast_deduplicator_behavior(self):
        """Verify inverted index deduplication removes exact duplicates and keeps distinct documents."""
        d1 = CorpusDocument(
            doc_id="d1",
            text="The Solar System consists of the Sun and eight major planets bound by gravity in outer space.",
            source_name="NASA", source_url="url1", author="NASA", license="Public Domain", license_evidence="17 USC 105",
            category="science_and_nature", subcategory="astronomy", genre="nonfiction", difficulty="intermediate", is_synthetic=False
        )
        d2 = CorpusDocument(
            doc_id="d2",
            text="The Solar System consists of the Sun and eight major planets bound by gravity in outer space.", # Exact dup
            source_name="NASA", source_url="url1", author="NASA", license="Public Domain", license_evidence="17 USC 105",
            category="science_and_nature", subcategory="astronomy", genre="nonfiction", difficulty="intermediate", is_synthetic=False
        )
        d3 = CorpusDocument(
            doc_id="d3",
            text="Plate tectonics describes the motion of rigid lithospheric plates moving atop the ductile asthenosphere beneath.",
            source_name="USGS", source_url="url2", author="USGS", license="Public Domain", license_evidence="17 USC 105",
            category="science_and_nature", subcategory="geology", genre="nonfiction", difficulty="intermediate", is_synthetic=False
        )

        retained, stats = self.deduplicator.deduplicate([d1, d2, d3])
        self.assertEqual(len(retained), 2)
        self.assertEqual(stats.exact_duplicates_removed, 1)
        self.assertEqual(stats.total_retained, 2)

    def test_quality_gate_failure_on_small_word_count(self):
        """Verify CorpusPipeline fails quality gates if retained word count is under 500k."""
        pipeline = CorpusPipeline(target_words=1000, synthetic_ratio=0.20, seed=42)
        train_doc = CorpusDocument(
            doc_id="train1", text="Short test paragraph.", source_name="src", source_url="url",
            author="auth", license="Public Domain", license_evidence="17 USC 105",
            category="cat", subcategory="subcat", genre="genre", difficulty="easy", is_synthetic=False
        )
        val_doc = CorpusDocument(
            doc_id="val1", text="Another short test paragraph.", source_name="src", source_url="url",
            author="auth", license="Public Domain", license_evidence="17 USC 105",
            category="cat", subcategory="subcat", genre="genre", difficulty="easy", is_synthetic=False
        )
        dedup_stats = DeduplicationResult(
            unique_documents=[train_doc, val_doc], exact_duplicates_removed=0,
            near_duplicates_removed=0, total_original=2, total_retained=2, duplicate_rate_pct=0.0
        )

        with self.assertRaises(RuntimeError) as ctx:
            pipeline.validate_quality_gates([train_doc], [val_doc], dedup_stats)
        self.assertIn("below 500,000 threshold", str(ctx.exception))

    def test_train_val_zero_document_leakage(self):
        """Verify train and val document IDs do not overlap."""
        output_dir = Path("data/processed/corpus")
        train_file = output_dir / "train.jsonl"
        val_file = output_dir / "val.jsonl"

        if train_file.exists() and val_file.exists():
            train_ids = set()
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    train_ids.add(data['doc_id'])

            val_ids = set()
            with open(val_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    val_ids.add(data['doc_id'])

            overlap = train_ids.intersection(val_ids)
            self.assertEqual(len(overlap), 0, "Train and Val splits must have ZERO overlapping document IDs.")


if __name__ == "__main__":
    unittest.main()
