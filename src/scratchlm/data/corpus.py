"""
Corpus Construction & Management Engine for ScratchLM Phase 1.

Implements a reproducible, end-to-end data pipeline:
1. Ingestion of 80% Real Public-Domain/Open-Licensed Human English Text
2. Ingestion of 20% Targeted Synthetic English Data
3. Cleaning, Normalization & Unicode Standardisation
4. Language Filtering (English Verification)
5. Deduplication (Exact SHA-256 + MinHash/Jaccard Near-Duplicates)
6. Document-Level Train/Validation Splitting (Zero Leakage)
7. Provenance & License Tracking
8. Automated Reporting & Analytics
"""

import json
import re
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict

from scratchlm.data.human_sources import RealHumanSourceProvider, HumanDocument
from scratchlm.data.synthetic_generator import SyntheticEnglishGenerator, SyntheticDocument
from scratchlm.data.preprocessing import clean_text, is_english_text, DEFAULT_ENGLISH_CLEANER
from scratchlm.data.deduplication import CorpusDeduplicator, DeduplicationResult
from scratchlm.data.splits import create_splits_by_document
from scratchlm.utils.paths import paths


@dataclass
class CorpusDocument:
    """Unified document representation in the ScratchLM corpus."""
    doc_id: str
    text: str
    source_name: str
    license: str
    category: str
    subcategory: str
    difficulty: str
    is_synthetic: bool
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    estimated_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.word_count:
            words = re.findall(r'\b\w+\b', self.text)
            self.word_count = len(words)
        if not self.sentence_count:
            sents = re.split(r'[.!?]+', self.text)
            self.sentence_count = len([s for s in sents if s.strip()])
        if not self.estimated_tokens:
            # Standard heuristic for English subword tokenization: ~1.3 tokens per word
            self.estimated_tokens = int(self.word_count * 1.3)


class CorpusPipeline:
    """End-to-end corpus building pipeline."""

    def __init__(
        self,
        target_total_docs: int = 200,
        synthetic_ratio: float = 0.20,
        val_ratio: float = 0.10,
        seed: int = 42,
    ):
        self.target_total_docs = target_total_docs
        self.synthetic_ratio = synthetic_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        self.deduplicator = CorpusDeduplicator(near_dup_threshold=0.85)
        self.human_provider = RealHumanSourceProvider()
        self.synthetic_generator = SyntheticEnglishGenerator(seed=seed)

    def build_corpus(self) -> Dict[str, Any]:
        """Execute full corpus pipeline and return comprehensive analytics report."""
        print("======================================================================")
        print("ScratchLM Phase 1 - Building English Foundation Corpus")
        print("======================================================================")

        # 1. Ingest Human Data (~80%)
        print("\n1. Ingesting Real Public-Domain & Open-Licensed Human English Data...")
        human_docs_raw = self.human_provider.get_all_documents()
        print(f"   Ingested {len(human_docs_raw)} human-authored documents.")

        # Calculate needed synthetic doc count
        target_human_count = len(human_docs_raw)
        target_synth_count = max(1, int(target_human_count * (self.synthetic_ratio / (1.0 - self.synthetic_ratio))))

        # 2. Ingest Synthetic Data (~20%)
        print(f"\n2. Generating Targeted Synthetic English Data ({target_synth_count} target docs)...")
        synthetic_docs_raw = self.synthetic_generator.generate_all(target_count=target_synth_count)
        print(f"   Generated {len(synthetic_docs_raw)} synthetic documents.")

        # Convert to unified CorpusDocument objects
        all_raw_docs: List[CorpusDocument] = []

        for h in human_docs_raw:
            all_raw_docs.append(CorpusDocument(
                doc_id=h.doc_id,
                text=h.text,
                source_name=h.source_name,
                license=h.license,
                category=h.category,
                subcategory=h.subcategory,
                difficulty=h.difficulty,
                is_synthetic=False,
                metadata=h.metadata
            ))

        for s in synthetic_docs_raw:
            all_raw_docs.append(CorpusDocument(
                doc_id=s.doc_id,
                text=s.text,
                source_name=s.source_name,
                license=s.license,
                category=s.category,
                subcategory=s.subcategory,
                difficulty=s.difficulty,
                is_synthetic=True,
                metadata=s.metadata
            ))

        total_ingested = len(all_raw_docs)
        print(f"\n3. Total Ingested Documents: {total_ingested}")

        # 3. Quality, Cleaning & Language Filtering
        print("\n4. Applying Quality Cleaning & Language Verification Filtering...")
        cleaned_docs: List[CorpusDocument] = []
        filtered_out = 0

        for doc in all_raw_docs:
            c_text = clean_text(doc.text, cleaner=DEFAULT_ENGLISH_CLEANER)
            if not c_text or not is_english_text(c_text, min_english_ratio=0.20):
                filtered_out += 1
                continue
            
            # Create cleaned copy
            cleaned_doc = CorpusDocument(
                doc_id=doc.doc_id,
                text=c_text,
                source_name=doc.source_name,
                license=doc.license,
                category=doc.category,
                subcategory=doc.subcategory,
                difficulty=doc.difficulty,
                is_synthetic=doc.is_synthetic,
                metadata=doc.metadata
            )
            cleaned_docs.append(cleaned_doc)

        print(f"   Passed filtering: {len(cleaned_docs)} docs ({filtered_out} discarded).")

        # 4. Deduplication
        print("\n5. Running Exact (SHA-256) & Near-Duplicate (Jaccard MinHash) Deduplication...")
        dedup_docs, dedup_stats = self.deduplicator.deduplicate(cleaned_docs, text_key="text")
        print(f"   Duplicates removed: {dedup_stats.exact_duplicates_removed} exact, {dedup_stats.near_duplicates_removed} near-dups.")
        print(f"   Retained unique docs: {len(dedup_docs)} (Duplicate rate: {dedup_stats.duplicate_rate_pct}%).")

        # 5. Train/Validation Split by Document (Zero Leakage)
        print("\n6. Splitting into Train and Validation sets (Document-level split)...")
        human_retained = [d for d in dedup_docs if not d.is_synthetic]
        synth_retained = [d for d in dedup_docs if d.is_synthetic]

        # Split human and synthetic separately to preserve 80/20 ratio in both splits
        val_human_cnt = max(1, int(len(human_retained) * self.val_ratio))
        val_synth_cnt = max(1, int(len(synth_retained) * self.val_ratio))

        val_docs = human_retained[:val_human_cnt] + synth_retained[:val_synth_cnt]
        train_docs = human_retained[val_human_cnt:] + synth_retained[val_synth_cnt:]

        # 6. Save Artifacts & Reports
        output_dir = paths.data_processed / "corpus"
        output_dir.mkdir(parents=True, exist_ok=True)

        self._export_splits(train_docs, val_docs, output_dir)

        # 7. Generate Full Analytics Report
        report = self._generate_report(train_docs, val_docs, dedup_stats)
        
        report_file = output_dir / "corpus_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        md_report_file = output_dir / "CORPUS_REPORT.md"
        self._write_markdown_report(report, md_report_file)

        print("\n======================================================================")
        print("Corpus Construction Successful!")
        print("======================================================================")
        print(f"  Processed Output Dir: {output_dir}")
        print(f"  Train Documents:      {report['train_documents']}")
        print(f"  Validation Documents: {report['val_documents']}")
        print(f"  Total Words:          {report['total_words']:,}")
        print(f"  Estimated Tokens:     {report['total_estimated_tokens']:,}")
        print(f"  Real Human Ratio:     {report['human_percentage']:.1f}%")
        print(f"  Synthetic Ratio:      {report['synthetic_percentage']:.1f}%")
        print("======================================================================\n")

        return report

    def _export_splits(
        self,
        train_docs: List[CorpusDocument],
        val_docs: List[CorpusDocument],
        output_dir: Path,
    ):
        """Export train/val splits to plain text and JSONL with metadata."""
        # Plain text files for trainer
        with open(output_dir / "train.txt", 'w', encoding='utf-8') as f:
            for d in train_docs:
                f.write(d.text + "\n\n")

        with open(output_dir / "val.txt", 'w', encoding='utf-8') as f:
            for d in val_docs:
                f.write(d.text + "\n\n")

        # JSONL with complete metadata & provenance
        with open(output_dir / "train.jsonl", 'w', encoding='utf-8') as f:
            for d in train_docs:
                f.write(json.dumps(asdict(d)) + "\n")

        with open(output_dir / "val.jsonl", 'w', encoding='utf-8') as f:
            for d in val_docs:
                f.write(json.dumps(asdict(d)) + "\n")

    def _generate_report(
        self,
        train_docs: List[CorpusDocument],
        val_docs: List[CorpusDocument],
        dedup_stats: DeduplicationResult,
    ) -> Dict[str, Any]:
        """Compile detailed statistics across categories, lengths, and provenance."""
        all_docs = train_docs + val_docs

        total_docs = len(all_docs)
        human_docs = [d for d in all_docs if not d.is_synthetic]
        synth_docs = [d for d in all_docs if d.is_synthetic]

        total_chars = sum(d.char_count for d in all_docs)
        total_words = sum(d.word_count for d in all_docs)
        total_tokens = sum(d.estimated_tokens for d in all_docs)

        category_counts: Dict[str, int] = {}
        for d in all_docs:
            category_counts[d.category] = category_counts.get(d.category, 0) + 1

        source_counts: Dict[str, int] = {}
        for d in all_docs:
            source_counts[d.source_name] = source_counts.get(d.source_name, 0) + 1

        avg_doc_len_words = total_words / total_docs if total_docs > 0 else 0
        avg_doc_len_chars = total_chars / total_docs if total_docs > 0 else 0

        # Unique vocabulary size across corpus
        all_vocab = set()
        for d in all_docs:
            words = re.findall(r'\b\w+\b', d.text.lower())
            all_vocab.update(words)

        return {
            "corpus_name": "ScratchLM Phase 1 English Foundation Corpus",
            "version": "1.0.0",
            "total_documents": total_docs,
            "train_documents": len(train_docs),
            "val_documents": len(val_docs),
            "total_characters": total_chars,
            "total_words": total_words,
            "total_estimated_tokens": total_tokens,
            "unique_vocabulary_words": len(all_vocab),
            "human_documents": len(human_docs),
            "synthetic_documents": len(synth_docs),
            "human_percentage": round((len(human_docs) / total_docs * 100.0) if total_docs else 0, 1),
            "synthetic_percentage": round((len(synth_docs) / total_docs * 100.0) if total_docs else 0, 1),
            "avg_document_words": round(avg_doc_len_words, 1),
            "avg_document_chars": round(avg_doc_len_chars, 1),
            "deduplication": {
                "total_original": dedup_stats.total_original,
                "exact_duplicates_removed": dedup_stats.exact_duplicates_removed,
                "near_duplicates_removed": dedup_stats.near_duplicates_removed,
                "retained_documents": dedup_stats.total_retained,
                "duplicate_rate_pct": dedup_stats.duplicate_rate_pct,
            },
            "category_distribution": category_counts,
            "source_distribution": source_counts,
        }

    def _write_markdown_report(self, report: Dict[str, Any], path: Path):
        """Write a formatted Markdown report summary for easy reading."""
        md = f"""# {report['corpus_name']} (v{report['version']})

## Overview & Executive Summary
- **Total Documents**: {report['total_documents']} ({report['train_documents']} Train / {report['val_documents']} Validation)
- **Total Words**: {report['total_words']:,}
- **Total Characters**: {report['total_characters']:,}
- **Estimated Subword Tokens**: {report['total_estimated_tokens']:,}
- **Unique Vocabulary (Words)**: {report['unique_vocabulary_words']:,}
- **Average Document Length**: {report['avg_document_words']} words ({report['avg_document_chars']} characters)

## Source Composition (80 / 20 Target Split)
- **Real Human Public-Domain / Open Sources**: {report['human_documents']} docs ({report['human_percentage']}%)
- **Targeted Synthetic Augmentations**: {report['synthetic_documents']} docs ({report['synthetic_percentage']}%)

## Deduplication & Quality Filtering Results
- **Original Ingested Candidates**: {report['deduplication']['total_original']}
- **Exact SHA-256 Duplicates Removed**: {report['deduplication']['exact_duplicates_removed']}
- **Near-Duplicate Passages Removed (MinHash Jaccard)**: {report['deduplication']['near_duplicates_removed']}
- **Duplicate Rate**: {report['deduplication']['duplicate_rate_pct']}%

## Category Distribution
"""
        for cat, cnt in report['category_distribution'].items():
            pct = (cnt / report['total_documents']) * 100.0
            md += f"- **{cat}**: {cnt} documents ({pct:.1f}%)\n"

        md += "\n## Source Provenance & Licensing Registry\n"
        for src, cnt in report['source_distribution'].items():
            pct = (cnt / report['total_documents']) * 100.0
            md += f"- **{src}**: {cnt} documents ({pct:.1f}%)\n"

        md += """
## How to Train ScratchLM on this Corpus
Run the following command in PowerShell / Terminal:
```powershell
python src/scripts/train.py --data data/processed/corpus/train.txt --val-data data/processed/corpus/val.txt --preset tiny --epochs 10 --experiment-id english_phase1
```
"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(md)
