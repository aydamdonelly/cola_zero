"""
Calibration Data Source Loaders

Provides unified interface to load calibration data from different sources:
- WikiText-2 (Wikipedia articles)
- C4 (web-crawled, deduplicated)
- MathQA (math problems with reasoning)
"""

from datasets import load_dataset
from typing import List, Set
import re
import hashlib


def compute_ngram_hash(text: str, n: int = 3) -> Set[str]:
    """Compute n-gram hashes for near-duplicate detection."""
    words = text.lower().split()
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.add(hashlib.md5(ngram.encode()).hexdigest()[:8])
    return ngrams


def deduplicate_texts(texts: List[str], min_unique_ratio: float = 0.3) -> List[str]:
    """
    Remove near-duplicates using 3-gram Jaccard similarity.

    Args:
        texts: List of documents
        min_unique_ratio: Minimum ratio of unique 3-grams to keep document

    Returns:
        Deduplicated list of documents
    """
    seen_ngrams = set()
    deduplicated = []

    for text in texts:
        doc_ngrams = compute_ngram_hash(text, n=3)

        if len(doc_ngrams) == 0:
            continue

        # Check overlap with seen ngrams
        overlap = len(doc_ngrams & seen_ngrams)
        unique_ratio = 1.0 - (overlap / len(doc_ngrams))

        if unique_ratio >= min_unique_ratio:
            deduplicated.append(text)
            seen_ngrams.update(doc_ngrams)

    return deduplicated


def load_wikitext_calib(split: str = 'train') -> List[str]:
    """
    Load WikiText-2 for calibration.

    Returns:
        List of text documents
    """
    print(f"[CALIB-SOURCE] Loading WikiText-2 ({split})...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)

    # Filter non-empty documents
    documents = [text for text in dataset['text'] if len(text.strip()) > 0]

    print(f"[CALIB-SOURCE] WikiText-2: {len(documents)} documents")
    return documents


def load_c4_calib(split: str = 'validation', lang: str = 'en', max_docs: int = 30000) -> List[str]:
    """
    Load C4 (Colossal Clean Crawled Corpus) for calibration.

    Args:
        split: 'train' or 'validation'
        lang: Language code (default: 'en')
        max_docs: Maximum documents to load (C4 is huge!)

    Returns:
        List of text documents
    """
    print(f"[CALIB-SOURCE] Loading C4 ({split}, {lang}, max={max_docs})...")

    # C4 is massive, we sample from validation split for speed
    dataset = load_dataset('allenai/c4', lang, split=split, streaming=True)

    # Boilerplate/junk patterns to filter
    junk_patterns = [
        'click here', 'subscribe', 'cookie', 'privacy policy',
        '©', 'all rights reserved', 'terms of service',
        'advertisement', '</div>', '<div', 'javascript'
    ]

    documents = []
    filtered_junk = 0
    filtered_short = 0
    filtered_low_diversity = 0

    for i, example in enumerate(dataset):
        if i >= max_docs:
            break

        text = example['text'].strip()

        # Filter 1: Very short docs
        if len(text) < 200:  # Increased from 100
            filtered_short += 1
            continue

        # Filter 2: Boilerplate/junk
        text_lower = text.lower()
        junk_count = sum(1 for pattern in junk_patterns if pattern in text_lower)
        if junk_count >= 3:  # If 3+ junk patterns found
            filtered_junk += 1
            continue

        # Filter 3: Low lexical diversity (likely repetitive/boilerplate)
        words = text.split()
        if len(words) > 20:  # Only check if enough words
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.15:  # Less than 15% unique words
                filtered_low_diversity += 1
                continue

        documents.append(text)

        if (i + 1) % 5000 == 0:
            print(f"  Loaded {i + 1} documents... (kept: {len(documents)}, filtered: junk={filtered_junk}, short={filtered_short}, low_div={filtered_low_diversity})")

    print(f"[CALIB-SOURCE] C4 filtering: kept={len(documents)}, filtered: junk={filtered_junk}, short={filtered_short}, low_diversity={filtered_low_diversity}")

    # Deduplicate
    print(f"[CALIB-SOURCE] Deduplicating C4 documents...")
    documents = deduplicate_texts(documents, min_unique_ratio=0.3)

    print(f"[CALIB-SOURCE] C4: {len(documents)} documents (after deduplication)")
    return documents


def load_mathqa_calib(split: str = 'train', max_docs: int = 10000) -> List[str]:
    """
    Load MathQA for calibration.

    MathQA contains math word problems with step-by-step reasoning.
    We format each example as: Question + Rationale + Answer

    Args:
        split: 'train' or 'validation'
        max_docs: Maximum documents to load

    Returns:
        List of formatted math problem texts
    """
    print(f"[CALIB-SOURCE] Loading MathQA ({split}, max={max_docs})...")

    dataset = load_dataset('math_qa', split=split)

    documents = []
    for i, example in enumerate(dataset):
        if i >= max_docs:
            break

        # Format: Question + Rationale + Answer
        question = example.get('Problem', '').strip()
        rationale = example.get('Rationale', '').strip()
        answer = example.get('correct', '').strip()

        if not question:
            continue

        # Build formatted text
        text_parts = [f"Question: {question}"]

        if rationale:
            text_parts.append(f"Reasoning: {rationale}")

        if answer:
            text_parts.append(f"Answer: {answer}")

        formatted_text = "\n\n".join(text_parts)
        documents.append(formatted_text)

    print(f"[CALIB-SOURCE] MathQA: {len(documents)} formatted problems")
    return documents


def get_calibration_source(source_name: str, split: str = 'train') -> List[str]:
    """
    Unified interface to load calibration data from different sources.

    Args:
        source_name: One of ['wikitext', 'c4', 'mathqa']
        split: Data split to use

    Returns:
        List of text documents
    """
    source_name = source_name.lower()

    if source_name == 'wikitext':
        return load_wikitext_calib(split=split)
    elif source_name == 'c4':
        return load_c4_calib(split='validation')  # Always use validation for C4
    elif source_name == 'mathqa':
        return load_mathqa_calib(split=split)
    else:
        raise ValueError(f"Unknown calibration source: {source_name}. "
                        f"Must be one of: ['wikitext', 'c4', 'mathqa']")
