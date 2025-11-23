"""
COLA-Zero Sampler: Main calibration data selection class.

Uses text-based features to approximate activation-based selection
without requiring forward passes through the target model.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple
from .features import extract_all_features


class COLAZeroSampler:
    """
    COLA-Zero: Calibration data selection without model forward passes.

    Uses text-based features (TF-IDF, perplexity, length, diversity) to
    approximate activation-based selection via k-means clustering.
    """

    def __init__(
        self,
        tokenizer,
        device: str = 'cuda',
        random_state: int = 42,
        include_perplexity: bool = True
    ):
        """
        Initialize sampler.

        Args:
            tokenizer: HuggingFace tokenizer for target model
            device: Device for perplexity computation ('cuda' or 'cpu')
            random_state: Random seed for reproducibility
            include_perplexity: Whether to include perplexity feature
                               (can be disabled for faster iteration)
        """
        self.tokenizer = tokenizer
        self.device = device
        self.random_state = random_state
        self.include_perplexity = include_perplexity
        self.rng = np.random.RandomState(self.random_state)

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def select_samples(
        self,
        texts: List[str],
        n_samples: int = 128,
        seq_len: int = 2048,
        min_length: int = 175  # Adaptive threshold, default 175 tokens
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, object]]:
        """
        Select n_samples diverse calibration samples from texts.

        Args:
            texts: Candidate documents
            n_samples: Number of samples to select (typically 128)
            seq_len: Target sequence length (typically 2048)
            min_length: Minimum text length (filter very short docs)

        Returns:
            Tuple containing:
              - List of dicts with 'input_ids' and 'attention_mask'
              - Metadata dict with selection details for reproducibility
        """
        print(f"\n{'='*60}")
        print("COLA-Zero Sample Selection")
        print(f"{'='*60}")
        print(f"Include perplexity: {self.include_perplexity}")

        target_tokens = n_samples * seq_len
        coverage_threshold = 1.10  # Require at least 110% coverage to avoid overlaps

        min_length_candidates = sorted({min_length, 200, 225})
        docs_per_cluster_options = [3, 2]

        best_selection = None
        best_coverage = -1.0

        def _run_selection(current_min_length: int, docs_per_cluster: int):
            print(f"\nStep 1: Filtering documents (min_length={current_min_length} tokens)...")
            valid_texts = []
            valid_indices = []
            for original_idx, t in enumerate(texts):
                n_tokens = len(self.tokenizer.encode(t, add_special_tokens=False))
                if n_tokens >= current_min_length:
                    valid_texts.append(t)
                    valid_indices.append(original_idx)
            print(f"  Valid documents: {len(valid_texts)} / {len(texts)}")

            if len(valid_texts) < n_samples:
                print(f"  Warning: Not enough valid documents for min_length={current_min_length}")
                return None

            print(f"\nStep 2: Extracting features from {len(valid_texts)} documents...")
            features = extract_all_features(
                texts=valid_texts,
                tokenizer=self.tokenizer,
                device=self.device,
                include_perplexity=self.include_perplexity
            )
            print(f"  Feature matrix shape: {features.shape}")

            print(f"\nStep 3: Clustering into {n_samples} groups (docs_per_cluster={docs_per_cluster})...")
            kmeans = KMeans(
                n_clusters=n_samples,
                random_state=self.random_state,
                n_init=50,  # Increased from 10 for stability (matches Balanced/Vanille)
                max_iter=300,
                verbose=0
            )
            cluster_labels = kmeans.fit_predict(features)
            centroids = kmeans.cluster_centers_
            print(f"  Clustering complete. Inertia: {kmeans.inertia_:.2f}")

            print(f"\nStep 4: Selecting representative documents from each cluster...")
            selected_texts = []
            selected_indices = []
            selected_filtered_indices = []

            for cluster_id in range(n_samples):
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]

                if len(cluster_indices) == 0:
                    print(f"  Warning: Cluster {cluster_id} is empty!")
                    continue

                cluster_features = features[cluster_indices]
                centroid = centroids[cluster_id]

                distances = np.linalg.norm(cluster_features - centroid, axis=1)
                top_k = min(docs_per_cluster, len(cluster_indices))
                closest_indices_in_cluster = np.argsort(distances)[:top_k]

                for idx_in_cluster in closest_indices_in_cluster:
                    filtered_idx = cluster_indices[idx_in_cluster]
                    original_idx = valid_indices[filtered_idx]
                    selected_filtered_indices.append(int(filtered_idx))
                    selected_indices.append(int(original_idx))
                    selected_texts.append(valid_texts[filtered_idx])

            print(f"  Selected {len(selected_texts)} representative documents "
                  f"(avg {len(selected_texts) / max(1, n_samples):.1f} per cluster)")

            total_tokens = sum(
                len(self.tokenizer.encode(text, add_special_tokens=False))
                for text in selected_texts
            )

            print(f"\n  Total tokens in selected documents: {total_tokens:,}")
            print(f"  Target tokens for {n_samples} chunks: {target_tokens:,}")
            coverage = total_tokens / max(1, target_tokens)
            print(f"  Coverage: {coverage * 100:.1f}%")

            return {
                "valid_texts_len": len(valid_texts),
                "selected_texts": selected_texts,
                "selected_indices": selected_indices,
                "selected_filtered_indices": selected_filtered_indices,
                "total_tokens": total_tokens,
                "coverage": coverage,
                "min_length": current_min_length,
                "docs_per_cluster": docs_per_cluster
            }

        for docs_per_cluster in docs_per_cluster_options:
            for current_min in min_length_candidates:
                selection = _run_selection(current_min, docs_per_cluster)
                if selection is None:
                    continue
                if selection["coverage"] > best_coverage:
                    best_selection = selection
                    best_coverage = selection["coverage"]
                if selection["coverage"] >= coverage_threshold:
                    print(f"  ✓ Coverage requirement met with min_length={current_min}, "
                          f"docs_per_cluster={docs_per_cluster}")
                    break
                else:
                    print(f"  Coverage below threshold ({selection['coverage']:.2f} < {coverage_threshold}); "
                          f"trying more restrictive settings...")
            if best_selection and best_selection["coverage"] >= coverage_threshold:
                break

        if best_selection is None:
            raise ValueError("Failed to select calibration data with the provided configuration.")

        if best_selection["coverage"] < coverage_threshold:
            print(f"  Warning: Coverage remains below {coverage_threshold:.2f}. "
                  f"Consider increasing n_samples or seq_len.")

        selected_texts = best_selection["selected_texts"]
        selected_indices = best_selection["selected_indices"]

        print(f"\nStep 5: Tokenizing to AutoGPTQ format (seq_len={seq_len})...")
        calibration_data = []
        chunk_start_positions = []

        all_text = "\n\n".join(selected_texts)
        all_tokens = self.tokenizer(
            all_text,
            return_tensors='pt',
            add_special_tokens=False
        )['input_ids'].squeeze(0)

        n_tokens = len(all_tokens)
        n_chunks = min(n_samples, n_tokens // seq_len)

        for i in range(n_chunks):
            start_idx = i * seq_len
            end_idx = start_idx + seq_len
            chunk = all_tokens[start_idx:end_idx]

            calibration_data.append({
                'input_ids': chunk,
                'attention_mask': torch.ones_like(chunk)
            })
            chunk_start_positions.append(int(start_idx))

        if len(calibration_data) < n_samples:
            missing = n_samples - len(calibration_data)
            print(f"  Note: Corpus only yielded {n_chunks} non-overlapping chunks")
            print(f"  Generating {missing} additional overlapping chunks...")

            for _ in range(missing):
                max_start = max(0, n_tokens - seq_len)
                start_idx = int(self.rng.randint(0, max_start + 1)) if max_start > 0 else 0
                end_idx = start_idx + seq_len
                chunk = all_tokens[start_idx:end_idx]

                calibration_data.append({
                    'input_ids': chunk,
                    'attention_mask': torch.ones_like(chunk)
                })
                chunk_start_positions.append(start_idx)

        if not calibration_data:
            raise ValueError("No calibration chunks could be generated. Check input data and parameters.")

        print(f"  Tokenization complete. Sample shape: {calibration_data[0]['input_ids'].shape}")
        print(f"\n{'='*60}")
        print(f"Selection complete: {len(calibration_data)} samples ready")
        print(f"{'='*60}\n")

        metadata = {
            "doc_indices": selected_indices,
            "filtered_doc_indices": best_selection.get("selected_filtered_indices", []),
            "chunk_starts": chunk_start_positions,
            "seq_len": seq_len,
            "coverage": best_selection["coverage"],
            "min_length": best_selection["min_length"],
            "docs_per_cluster": best_selection["docs_per_cluster"],
            "total_tokens": best_selection["total_tokens"],
            "n_valid_documents": best_selection["valid_texts_len"]
        }

        return calibration_data, metadata
