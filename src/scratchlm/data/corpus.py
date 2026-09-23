"""
Corpus Construction & Management Engine for ScratchLM Phase 1.

Implements a reproducible, end-to-end data pipeline:
1. Ingestion of ~80% Real Public-Domain/Open-Licensed Human English Text
2. Ingestion of ~20% Targeted Synthetic English Data
3. Quality Cleaning, Normalization & Unicode Standardization
4. Language Filtering (English Verification)
5. Deduplication (Exact SHA-256 + MinHash/Jaccard Near-Duplicates)
6. Document-Level Train/Validation Splitting (Zero Leakage)
7. Provenance & License Tracking
8. Automated Reporting, Diversity Analytics & Concentration Warning Checks
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
    """Unified document representation in the ScratchLM corpus with full verifiable provenance."""
    doc_id: str
    text: str
    source_name: str
    source_url: str
    author: str
    license: str
    license_evidence: str
    category: str
    subcategory: str
    genre: str
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
            self.estimated_tokens = int(self.word_count * 1.3)


class CorpusPipeline:
    """End-to-end corpus building and scaling pipeline."""

    def __init__(
        self,
        target_words: int = 100000,
        synthetic_ratio: float = 0.20,
        val_ratio: float = 0.10,
        seed: int = 42,
    ):
        self.target_words = target_words
        self.synthetic_ratio = synthetic_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        self.deduplicator = CorpusDeduplicator(near_dup_threshold=0.85)
        self.human_provider = RealHumanSourceProvider(seed=seed)
        self.synthetic_generator = SyntheticEnglishGenerator(seed=seed)

    def build_corpus(self) -> Dict[str, Any]:
        """Execute full corpus pipeline and return comprehensive analytics report."""
        print("======================================================================")
        print("ScratchLM Phase 1 - Scaling English Foundation Corpus")
        print("======================================================================")

        # Target human word count (~80%)
        human_target_words = int(self.target_words * (1.0 - self.synthetic_ratio))

        # 1. Ingest Human Data (~80%)
        print(f"\n1. Ingesting Real Public-Domain & Open-Licensed Human English Data (Target: {human_target_words:,} words)...")
        human_docs_raw = self.human_provider.get_all_documents(min_target_words=human_target_words)
        human_word_sum = sum(len(d.text.split()) for d in human_docs_raw)
        print(f"   Ingested {len(human_docs_raw)} human documents ({human_word_sum:,} words).")

        # Calculate needed synthetic doc count (~20%)
        synth_target_words = int(human_word_sum * (self.synthetic_ratio / (1.0 - self.synthetic_ratio)))
        est_synth_doc_count = max(10, int(synth_target_words / 250))

        # 2. Ingest Synthetic Data (~20%)
        print(f"\n2. Generating Targeted Synthetic English Data ({est_synth_doc_count} docs / {synth_target_words:,} words)...")
        synthetic_docs_raw = self.synthetic_generator.generate_all(target_count=est_synth_doc_count)
        synth_word_sum = sum(len(s.text.split()) for s in synthetic_docs_raw)
        print(f"   Generated {len(synthetic_docs_raw)} synthetic documents ({synth_word_sum:,} words).")

        # Convert to unified CorpusDocument objects
        all_raw_docs: List[CorpusDocument] = []

        for h in human_docs_raw:
            all_raw_docs.append(CorpusDocument(
                doc_id=h.doc_id,
                text=h.text,
                source_name=h.source_name,
                source_url=h.source_url,
                author=h.author,
                license=h.license,
                license_evidence=h.license_evidence,
                category=h.category,
                subcategory=h.subcategory,
                genre=h.genre,
                difficulty=h.difficulty,
                is_synthetic=False,
                metadata=h.metadata
            ))

        for s in synthetic_docs_raw:
            all_raw_docs.append(CorpusDocument(
                doc_id=s.doc_id,
                text=s.text,
                source_name=s.source_name,
                source_url=s.source_url,
                author=s.author,
                license=s.license,
                license_evidence=s.license_evidence,
                category=s.category,
                subcategory=s.subcategory,
                genre=s.genre,
                difficulty=s.difficulty,
                is_synthetic=True,
                metadata=s.metadata
            ))

        total_ingested_docs = len(all_raw_docs)
        total_ingested_words = sum(d.word_count for d in all_raw_docs)
        print(f"\n3. Total Ingested Candidates: {total_ingested_docs} docs ({total_ingested_words:,} words).")

        # 3. Quality Cleaning & Language Verification Filtering
        print("\n4. Applying Quality Cleaning & Language Verification Filtering...")
        cleaned_docs: List[CorpusDocument] = []
        filtered_out = 0

        for doc in all_raw_docs:
            c_text = clean_text(doc.text, cleaner=DEFAULT_ENGLISH_CLEANER)
            if not c_text or not is_english_text(c_text, min_english_ratio=0.10):
                filtered_out += 1
                continue
            
            cleaned_doc = CorpusDocument(
                doc_id=doc.doc_id,
                text=c_text,
                source_name=doc.source_name,
                source_url=doc.source_url,
                author=doc.author,
                license=doc.license,
                license_evidence=doc.license_evidence,
                category=doc.category,
                subcategory=doc.subcategory,
                genre=doc.genre,
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

        val_human_cnt = max(1, int(len(human_retained) * self.val_ratio))
        val_synth_cnt = max(1, int(len(synth_retained) * self.val_ratio))

        val_docs = human_retained[:val_human_cnt] + synth_retained[:val_synth_cnt]
        train_docs = human_retained[val_human_cnt:] + synth_retained[val_synth_cnt:]

        # 6. Save Artifacts & Reports
        output_dir = paths.data_processed / "corpus"
        output_dir.mkdir(parents=True, exist_ok=True)

        self._export_splits(train_docs, val_docs, output_dir)

        # 7. Generate Comprehensive Analytics Report
        report = self._generate_report(train_docs, val_docs, dedup_stats)
        
        report_file = output_dir / "corpus_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        md_report_file = output_dir / "CORPUS_REPORT.md"
        self._write_markdown_report(report, md_report_file)

        print("\n======================================================================")
        print("Corpus Scaling Successful!")
        print("======================================================================")
        print(f"  Processed Output Dir: {output_dir}")
        print(f"  Train Documents:      {report['train_documents']}")
        print(f"  Validation Documents: {report['val_documents']}")
        print(f"  Total Words:          {report['total_words']:,}")
        print(f"  Estimated Tokens:     {report['total_estimated_tokens']:,}")
        print(f"  Real Human Ratio:     {report['human_percentage']:.1f}% ({report['human_words']:,} words)")
        print(f"  Synthetic Ratio:      {report['synthetic_percentage']:.1f}% ({report['synthetic_words']:,} words)")
        print(f"  Vocabulary Size:      {report['unique_vocabulary_words']:,} unique words")
        print(f"  Dominance Warnings:   {len(report['concentration_warnings'])} flagged")
        print("======================================================================\n")

        return report

    def _export_splits(
        self,
        train_docs: List[CorpusDocument],
        val_docs: List[CorpusDocument],
        output_dir: Path,
    ):
        """Export train/val splits to plain text and JSONL with metadata."""
        with open(output_dir / "train.txt", 'w', encoding='utf-8') as f:
            for d in train_docs:
                f.write(d.text + "\n\n")

        with open(output_dir / "val.txt", 'w', encoding='utf-8') as f:
            for d in val_docs:
                f.write(d.text + "\n\n")

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
        human_words = sum(d.word_count for d in human_docs)
        synthetic_words = sum(d.word_count for d in synth_docs)
        total_tokens = sum(d.estimated_tokens for d in all_docs)

        # Document length metrics
        word_counts = sorted([d.word_count for d in all_docs])
        char_counts = sorted([d.char_count for d in all_docs])
        avg_doc_len_words = total_words / total_docs if total_docs > 0 else 0
        median_doc_len_words = word_counts[len(word_counts) // 2] if word_counts else 0

        # Sentence length metrics
        all_sentences = []
        for d in all_docs:
            sents = re.split(r'[.!?]+', d.text)
            all_sentences.extend([s.strip() for s in sents if s.strip()])
        
        sentence_word_lengths = sorted([len(s.split()) for s in all_sentences]) if all_sentences else [0]
        avg_sent_len_words = sum(sentence_word_lengths) / len(sentence_word_lengths) if sentence_word_lengths else 0
        median_sent_len_words = sentence_word_lengths[len(sentence_word_lengths) // 2] if sentence_word_lengths else 0

        # Distributions
        category_counts: Dict[str, int] = {}
        category_words: Dict[str, int] = {}
        for d in all_docs:
            category_counts[d.category] = category_counts.get(d.category, 0) + 1
            category_words[d.category] = category_words.get(d.category, 0) + d.word_count

        genre_counts: Dict[str, int] = {}
        for d in all_docs:
            genre_counts[d.genre] = genre_counts.get(d.genre, 0) + 1

        difficulty_counts: Dict[str, int] = {}
        for d in all_docs:
            difficulty_counts[d.difficulty] = difficulty_counts.get(d.difficulty, 0) + 1

        source_counts: Dict[str, int] = {}
        source_words: Dict[str, int] = {}
        for d in all_docs:
            source_counts[d.source_name] = source_counts.get(d.source_name, 0) + 1
            source_words[d.source_name] = source_words.get(d.source_name, 0) + d.word_count

        # Vocabulary & Train/Val Overlap
        train_vocab = set()
        for d in train_docs:
            train_vocab.update(re.findall(r'\b\w+\b', d.text.lower()))

        val_vocab = set()
        for d in val_docs:
            val_vocab.update(re.findall(r'\b\w+\b', d.text.lower()))

        all_vocab = train_vocab.union(val_vocab)
        vocab_jaccard = len(train_vocab.intersection(val_vocab)) / len(all_vocab) if all_vocab else 0.0
        val_oov_words = val_vocab - train_vocab
        val_oov_rate = len(val_oov_words) / len(val_vocab) if val_vocab else 0.0

        # Concentration & Dominance Detection
        warnings = []
        for src, words in source_words.items():
            pct = (words / total_words) * 100.0 if total_words else 0
            if pct > 25.0:
                warnings.append(f"Source concentration warning: '{src}' represents {pct:.1f}% of total words.")

        for cat, words in category_words.items():
            pct = (words / total_words) * 100.0 if total_words else 0
            if pct > 35.0:
                warnings.append(f"Category concentration warning: '{cat}' represents {pct:.1f}% of total words.")

        return {
            "corpus_name": "ScratchLM Phase 1 Expanded English Foundation Corpus",
            "version": "1.1.0",
            "total_documents": total_docs,
            "train_documents": len(train_docs),
            "val_documents": len(val_docs),
            "total_characters": total_chars,
            "total_words": total_words,
            "human_words": human_words,
            "synthetic_words": synthetic_words,
            "total_estimated_tokens": total_tokens,
            "unique_vocabulary_words": len(all_vocab),
            "human_documents": len(human_docs),
            "synthetic_documents": len(synth_docs),
            "human_doc_percentage": round((len(human_docs) / total_docs * 100.0) if total_docs else 0, 1),
            "synthetic_doc_percentage": round((len(synth_docs) / total_docs * 100.0) if total_docs else 0, 1),
            "human_word_percentage": round((human_words / total_words * 100.0) if total_words else 0, 1),
            "synthetic_word_percentage": round((synthetic_words / total_words * 100.0) if total_words else 0, 1),
            "human_percentage": round((human_words / total_words * 100.0) if total_words else 0, 1),
            "synthetic_percentage": round((synthetic_words / total_words * 100.0) if total_words else 0, 1),
            "avg_document_words": round(avg_doc_len_words, 1),
            "median_document_words": median_doc_len_words,
            "total_sentences": len(all_sentences),
            "avg_sentence_words": round(avg_sent_len_words, 1),
            "median_sentence_words": median_sent_len_words,
            "deduplication": {
                "total_original": dedup_stats.total_original,
                "exact_duplicates_removed": dedup_stats.exact_duplicates_removed,
                "near_duplicates_removed": dedup_stats.near_duplicates_removed,
                "retained_documents": dedup_stats.total_retained,
                "duplicate_rate_pct": dedup_stats.duplicate_rate_pct,
            },
            "vocabulary_overlap": {
                "train_vocab_size": len(train_vocab),
                "val_vocab_size": len(val_vocab),
                "vocab_jaccard_similarity": round(vocab_jaccard, 3),
                "val_oov_words_count": len(val_oov_words),
                "val_oov_rate_pct": round(val_oov_rate * 100.0, 2),
            },
            "category_distribution": category_counts,
            "genre_distribution": genre_counts,
            "difficulty_distribution": difficulty_counts,
            "source_distribution": source_counts,
            "concentration_warnings": warnings,
        }

    def _write_markdown_report(self, report: Dict[str, Any], path: Path):
        """Write a formatted Markdown report summary for easy reading."""
        md = f"""# {report['corpus_name']} (v{report['version']})

## Overview & Executive Summary
- **Total Documents**: {report['total_documents']} ({report['train_documents']} Train / {report['val_documents']} Validation)
- **Total Words**: {report['total_words']:,} ({report['human_words']:,} Human / {report['synthetic_words']:,} Synthetic)
- **Total Characters**: {report['total_characters']:,}
- **Estimated Subword Tokens**: {report['total_estimated_tokens']:,}
- **Unique Vocabulary (Words)**: {report['unique_vocabulary_words']:,}
- **Mean Document Length**: {report['avg_document_words']} words
- **Median Document Length**: {report['median_document_words']} words
- **Total Sentences**: {report['total_sentences']:,} (Mean: {report['avg_sentence_words']} words/sent, Median: {report['median_sentence_words']} words/sent)

## Source Composition (Target 80 / 20 Split)
- **Real Human Public-Domain / Open Sources**: {report['human_documents']} docs ({report['human_percentage']}%)
- **Targeted Synthetic Augmentations**: {report['synthetic_documents']} docs ({report['synthetic_percentage']}%)

## Train / Validation Vocabulary & Leakage Analysis
- **Train Vocabulary**: {report['vocabulary_overlap']['train_vocab_size']:,} words
- **Validation Vocabulary**: {report['vocabulary_overlap']['val_vocab_size']:,} words
- **Jaccard Vocabulary Overlap**: {report['vocabulary_overlap']['vocab_jaccard_similarity']}
- **Validation Out-of-Vocabulary (OOV) Rate**: {report['vocabulary_overlap']['val_oov_rate_pct']}%

## Deduplication & Quality Filtering Results
- **Original Ingested Candidates**: {report['deduplication']['total_original']}
- **Exact SHA-256 Duplicates Removed**: {report['deduplication']['exact_duplicates_removed']}
- **Near-Duplicate Passages Removed (MinHash Jaccard)**: {report['deduplication']['near_duplicates_removed']}
- **Duplicate Rate**: {report['deduplication']['duplicate_rate_pct']}%

## Concentration & Concentration Warnings
"""
        if report['concentration_warnings']:
            for w in report['concentration_warnings']:
                md += f"- ⚠️ {w}\n"
        else:
            md += "- ✓ No excessive source or category concentration detected.\n"

        md += "\n## Category Distribution\n"
        for cat, cnt in report['category_distribution'].items():
            pct = (cnt / report['total_documents']) * 100.0
            md += f"- **{cat}**: {cnt} documents ({pct:.1f}%)\n"

        md += "\n## Genre Distribution\n"
        for g, cnt in report['genre_distribution'].items():
            pct = (cnt / report['total_documents']) * 100.0
            md += f"- **{g}**: {cnt} documents ({pct:.1f}%)\n"

        md += "\n## Reading Difficulty Distribution\n"
        for d, cnt in report['difficulty_distribution'].items():
            pct = (cnt / report['total_documents']) * 100.0
            md += f"- **{d}**: {cnt} documents ({pct:.1f}%)\n"

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
