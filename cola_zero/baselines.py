"""
Baseline calibration data samplers for comparison.

Implements:
- RandomSampler: Standard random sampling (mimics AutoGPTQ examples)
- StratifiedSampler: Naive stratification by document length
"""

import random
import torch
import numpy as np
from typing import List, Dict, Tuple


class RandomSampler:
    """
    Random calibration sample selection (baseline).

    Mimics AutoGPTQ's basic_usage_wikitext2.py approach:
    concatenates all texts and samples random contiguous chunks.
    """

    def __init__(self, tokenizer, random_state: int = 42):
        """
        Initialize random sampler.

        Args:
            tokenizer: HuggingFace tokenizer
            random_state: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.random_state = random_state
        self._rng = random.Random(self.random_state)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def select_samples(
        self,
        texts: List[str],
        n_samples: int = 128,
        seq_len: int = 2048
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, object]]:
        """
        Randomly select n_samples from texts.

        This concatenates all texts and samples random chunks,
        matching the approach in AutoGPTQ's examples.

        Args:
            texts: List of candidate documents
            n_samples: Number of samples to select
            seq_len: Sequence length per sample

        Returns:
            List of dicts with 'input_ids' and 'attention_mask'
        """
        print(f"\n{'='*60}")
        print(f"Random Baseline Sampling")
        print(f"{'='*60}")
        print(f"Selecting {n_samples} samples of length {seq_len}...")

        # Concatenate all texts (like WikiText example)
        full_text = "\n\n".join(texts)
        print(f"  Concatenated corpus length: {len(full_text)} characters")

        # Tokenize entire corpus
        full_tokens = self.tokenizer(full_text, return_tensors='pt')
        input_ids = full_tokens['input_ids']
        total_tokens = input_ids.shape[1]

        print(f"  Total tokens in corpus: {total_tokens}")

        # Random sampling
        self._rng.seed(self.random_state)
        calibration_data = []
        start_positions = []

        for i in range(n_samples):
            # Random start position
            if total_tokens <= seq_len:
                start_idx = 0
            else:
                start_idx = self._rng.randint(0, total_tokens - seq_len - 1)

            end_idx = start_idx + seq_len
            start_positions.append(int(start_idx))

            # Extract chunk
            chunk_ids = input_ids[:, start_idx:end_idx]

            # Pad if necessary (in case we're at the end)
            if chunk_ids.shape[1] < seq_len:
                padding = torch.full(
                    (1, seq_len - chunk_ids.shape[1]),
                    self.tokenizer.pad_token_id,
                    dtype=chunk_ids.dtype
                )
                chunk_ids = torch.cat([chunk_ids, padding], dim=1)

            attention_mask = (chunk_ids != self.tokenizer.pad_token_id).long()

            calibration_data.append({
                'input_ids': chunk_ids.squeeze(0),
                'attention_mask': attention_mask.squeeze(0)
            })

        print(f"  Random sampling complete: {len(calibration_data)} samples")
        print(f"{'='*60}\n")

        metadata = {
            "start_positions": start_positions,
            "seq_len": seq_len,
            "total_tokens": int(total_tokens)
        }

        return calibration_data, metadata


class StratifiedSampler:
    """
    Stratified sampling by document length (naive baseline).

    Buckets documents into length categories (short/medium/long)
    and samples proportionally from each bucket.
    """

    def __init__(self, tokenizer, random_state: int = 42):
        """
        Initialize stratified sampler.

        Args:
            tokenizer: HuggingFace tokenizer
            random_state: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.random_state = random_state

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def select_samples(
        self,
        texts: List[str],
        n_samples: int = 128,
        seq_len: int = 2048,
        min_length: int = 100
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, object]]:
        """
        Select samples stratified by document length.

        Args:
            texts: List of candidate documents
            n_samples: Number of samples to select
            seq_len: Sequence length per sample
            min_length: Minimum character length for valid documents

        Returns:
            List of dicts with 'input_ids' and 'attention_mask'
        """
        print(f"\n{'='*60}")
        print(f"Stratified (by Length) Sampling")
        print(f"{'='*60}")

        # Filter valid documents
        valid_texts = []
        valid_indices = []
        for original_idx, text in enumerate(texts):
            if len(text.strip()) >= min_length:
                valid_texts.append(text)
                valid_indices.append(original_idx)
        print(f"  Valid documents: {len(valid_texts)} / {len(texts)}")

        if len(valid_texts) < n_samples:
            raise ValueError(
                f"Not enough valid documents ({len(valid_texts)}) "
                f"to select {n_samples} samples"
            )

        # Compute document lengths (in tokens)
        print("  Computing document lengths...")
        doc_lengths = []
        for text in valid_texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            doc_lengths.append(len(tokens))

        doc_lengths = np.array(doc_lengths)

        # Define buckets based on percentiles
        p33 = np.percentile(doc_lengths, 33)
        p66 = np.percentile(doc_lengths, 66)

        print(f"  Length distribution: min={doc_lengths.min()}, "
              f"33rd%={p33:.0f}, median={np.median(doc_lengths):.0f}, "
              f"67th%={p66:.0f}, max={doc_lengths.max()}")

        # Assign to buckets
        buckets = {
            'short': [],   # < p33
            'medium': [],  # p33 <= x < p66
            'long': []     # >= p66
        }

        for idx, length in enumerate(doc_lengths):
            if length < p33:
                buckets['short'].append(idx)
            elif length < p66:
                buckets['medium'].append(idx)
            else:
                buckets['long'].append(idx)

        print(f"  Bucket sizes: short={len(buckets['short'])}, "
              f"medium={len(buckets['medium'])}, long={len(buckets['long'])}")

        # Sample proportionally from each bucket
        np.random.seed(self.random_state)
        samples_per_bucket = n_samples // 3
        remainder = n_samples % 3

        selected_filtered_indices = []
        selected_doc_indices = []

        for i, (bucket_name, bucket_indices) in enumerate(buckets.items()):
            # Add one extra to first bucket if there's a remainder
            n_to_sample = samples_per_bucket + (1 if i < remainder else 0)

            # Sample from this bucket
            if len(bucket_indices) >= n_to_sample:
                sampled = np.random.choice(bucket_indices, size=n_to_sample, replace=False)
            else:
                # If bucket too small, sample with replacement
                sampled = np.random.choice(bucket_indices, size=n_to_sample, replace=True)

            selected_filtered_indices.extend(int(idx) for idx in sampled)
            selected_doc_indices.extend(int(valid_indices[idx]) for idx in sampled)
            print(f"    Sampled {len(sampled)} from {bucket_name} bucket")

        # Get selected texts
        selected_texts = [valid_texts[idx] for idx in selected_filtered_indices]

        # Tokenize
        print(f"\n  Tokenizing {len(selected_texts)} samples...")
        calibration_data = []

        for text in selected_texts:
            tokens = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=seq_len,
                truncation=True,
                padding='max_length',
                add_special_tokens=True
            )

            calibration_data.append({
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0)
            })

        print(f"  Stratified sampling complete: {len(calibration_data)} samples")
        print(f"{'='*60}\n")

        metadata = {
            "doc_indices": selected_doc_indices,
            "filtered_doc_indices": selected_filtered_indices,
            "seq_len": seq_len,
            "min_length": min_length
        }

        return calibration_data, metadata
