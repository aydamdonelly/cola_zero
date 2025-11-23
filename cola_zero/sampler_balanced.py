"""
COLA-Zero Sampler with BALANCED features.

KEY CHANGES from original sampler.py:
1. TF-IDF reduced from 5000 → 100 dims via SVD
2. Explicit feature weighting to balance contributions
3. All other logic identical to original
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from typing import List, Dict, Tuple


def extract_tfidf_reduced(texts: List[str], n_components: int = 100) -> np.ndarray:
    """Extract TF-IDF and reduce to n_components dimensions."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    # Reduce dimensionality
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    return tfidf_reduced


def compute_lengths(texts: List[str], tokenizer) -> np.ndarray:
    """Compute sequence lengths."""
    return np.array([len(tokenizer.encode(t, add_special_tokens=True)) for t in texts])


def compute_diversity(texts: List[str], tokenizer) -> np.ndarray:
    """Compute vocabulary diversity."""
    scores = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) == 0:
            scores.append(0.0)
        else:
            scores.append(len(set(tokens)) / len(tokens))
    return np.array(scores)


class COLAZeroBalancedSampler:
    """
    COLA-Zero with balanced features.

    Changes:
    - TF-IDF: 5000 → 100 dims
    - Feature weighting: Equal contribution from each feature TYPE
    """

    def __init__(
        self,
        tokenizer,
        device: str = 'cuda',
        random_state: int = 42,
        tfidf_dims: int = 100,
        feature_weights: dict = None
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            device: Device (not used without perplexity)
            random_state: Random seed
            tfidf_dims: TF-IDF dimensionality after SVD (default: 100)
            feature_weights: Dict of weights for each feature type
                           Default: {'tfidf': 1.0, 'length': 100.0, 'diversity': 100.0}
        """
        self.tokenizer = tokenizer
        self.device = device
        self.random_state = random_state
        self.tfidf_dims = tfidf_dims
        self.rng = np.random.RandomState(self.random_state)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Default: Equal L2 contribution (sqrt-dim rule)
        # After StandardScaler, L2 distance = sqrt(sum of squared weighted features)
        # So we need: sqrt(n_dims × weight²) to be equal for all feature groups
        if feature_weights is None:
            self.feature_weights = {
                'tfidf': 0.1,      # sqrt(100 × 0.1²) = 1.0
                'length': 1.0,     # sqrt(1 × 1.0²) = 1.0
                'diversity': 1.0   # sqrt(1 × 1.0²) = 1.0
            }
        else:
            self.feature_weights = feature_weights

    def select_samples(
        self,
        texts: List[str],
        n_samples: int = 128,
        seq_len: int = 2048,
        min_length: int = 175
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, object]]:
        """
        Select diverse calibration samples with BALANCED features.

        Args:
            texts: Candidate documents
            n_samples: Number of samples (typically 128)
            seq_len: Sequence length (typically 2048)
            min_length: Min document length in tokens

        Returns:
            (calibration_data, metadata)
        """
        print(f"\n{'='*60}")
        print("COLA-Zero BALANCED Sample Selection")
        print(f"{'='*60}")
        print(f"TF-IDF dims: {self.tfidf_dims}")
        print(f"Feature weights: {self.feature_weights}")

        target_tokens = n_samples * seq_len
        coverage_threshold = 1.10

        base_min_length = min_length

        # Try different min_length thresholds
        min_length_candidates = sorted({min_length, 200, 225})
        docs_per_cluster_options = [3, 2]

        best_selection = None
        best_coverage = -1.0

        def _run_selection(current_min_length: int, docs_per_cluster: int):
            print(f"\nStep 1: Filtering (min_length={current_min_length})...")
            valid_texts = []
            valid_indices = []
            for original_idx, t in enumerate(texts):
                n_tokens = len(self.tokenizer.encode(t, add_special_tokens=False))
                if n_tokens >= current_min_length:
                    valid_texts.append(t)
                    valid_indices.append(original_idx)
            print(f"  Valid: {len(valid_texts)} / {len(texts)}")

            if len(valid_texts) < n_samples:
                return None

            # Extract BALANCED features
            print(f"\nStep 2: Extracting BALANCED features...")
            print(f"  - TF-IDF ({self.tfidf_dims} dims, weight={self.feature_weights['tfidf']})...")
            tfidf = extract_tfidf_reduced(valid_texts, n_components=self.tfidf_dims)
            tfidf *= self.feature_weights['tfidf']

            print(f"  - Length (1 dim, weight={self.feature_weights['length']})...")
            lengths = compute_lengths(valid_texts, self.tokenizer).reshape(-1, 1).astype(np.float64)
            lengths *= self.feature_weights['length']

            print(f"  - Diversity (1 dim, weight={self.feature_weights['diversity']})...")
            diversity = compute_diversity(valid_texts, self.tokenizer).reshape(-1, 1).astype(np.float64)
            diversity *= self.feature_weights['diversity']

            # Combine and normalize
            combined = np.column_stack([tfidf, lengths, diversity])
            print(f"  Combined shape: {combined.shape}")

            scaler = StandardScaler()
            features = scaler.fit_transform(combined)

            # K-Means clustering
            print(f"\nStep 3: K-Means clustering (k={n_samples})...")
            kmeans = KMeans(
                n_clusters=n_samples,
                random_state=self.random_state,
                init='k-means++',
                n_init=50,
                max_iter=300,
                verbose=0
            )
            cluster_labels = kmeans.fit_predict(features)
            centroids = kmeans.cluster_centers_
            print(f"  Inertia: {kmeans.inertia_:.2f}")

            # Select representatives
            print(f"\nStep 4: Selecting representatives...")
            selected_texts = []
            selected_indices = []
            selected_filtered_indices = []

            for cluster_id in range(n_samples):
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]

                if len(cluster_indices) == 0:
                    continue

                cluster_features = features[cluster_indices]
                centroid = centroids[cluster_id]

                distances = np.linalg.norm(cluster_features - centroid, axis=1)
                top_k = min(docs_per_cluster, len(cluster_indices))
                closest = np.argsort(distances)[:top_k]

                for idx in closest:
                    filtered_idx = cluster_indices[idx]
                    original_idx = valid_indices[filtered_idx]
                    selected_filtered_indices.append(int(filtered_idx))
                    selected_indices.append(int(original_idx))
                    selected_texts.append(valid_texts[filtered_idx])

            print(f"  Selected: {len(selected_texts)} documents")

            # Coverage check
            total_tokens = sum(
                len(self.tokenizer.encode(text, add_special_tokens=False))
                for text in selected_texts
            )

            coverage = total_tokens / max(1, target_tokens)
            print(f"  Coverage: {coverage * 100:.1f}%")

            return {
                "selected_texts": selected_texts,
                "selected_indices": selected_indices,
                "selected_filtered_indices": selected_filtered_indices,
                "total_tokens": total_tokens,
                "coverage": coverage,
                "min_length": current_min_length,
                "docs_per_cluster": docs_per_cluster
            }

        # Try combinations
        for docs_per_cluster in docs_per_cluster_options:
            for current_min in min_length_candidates:
                selection = _run_selection(current_min, docs_per_cluster)
                if selection is None:
                    continue
                if selection["coverage"] > best_coverage:
                    best_selection = selection
                    best_coverage = selection["coverage"]
                if selection["coverage"] >= coverage_threshold:
                    break
            if best_selection and best_selection["coverage"] >= coverage_threshold:
                break

        if best_selection is None:
            raise ValueError("Failed to select calibration data")

        # COVERAGE GUARD: Enforce ≥110% coverage before tokenization
        if best_selection["coverage"] < coverage_threshold:
            print(f"\n⚠️  Coverage below threshold ({best_selection['coverage']:.2%} < {coverage_threshold:.2%})")
            print(f"  Applying coverage guard: adding more documents until ≥{coverage_threshold:.2%}...")

            # Re-extract features for the selected min_length
            current_min = min(base_min_length, best_selection["min_length"])
            valid_texts = []
            valid_indices = []
            for original_idx, t in enumerate(texts):
                n_tokens = len(self.tokenizer.encode(t, add_special_tokens=False))
                if n_tokens >= current_min:
                    valid_texts.append(t)
                    valid_indices.append(original_idx)

            best_selection["min_length"] = current_min
            best_selection["valid_texts_len"] = len(valid_texts)

            # Re-extract features
            tfidf = extract_tfidf_reduced(valid_texts, n_components=self.tfidf_dims)
            tfidf *= self.feature_weights['tfidf']
            lengths = compute_lengths(valid_texts, self.tokenizer).reshape(-1, 1).astype(np.float64)
            lengths *= self.feature_weights['length']
            diversity = compute_diversity(valid_texts, self.tokenizer).reshape(-1, 1).astype(np.float64)
            diversity *= self.feature_weights['diversity']
            combined = np.column_stack([tfidf, lengths, diversity])
            scaler = StandardScaler()
            features = scaler.fit_transform(combined)

            # Re-run clustering
            kmeans = KMeans(
                n_clusters=n_samples,
                random_state=self.random_state,
                init='k-means++',
                n_init=50,  # Increased from 10 for consistency
                max_iter=300,
                verbose=0
            )
            cluster_labels = kmeans.fit_predict(features)
            centroids = kmeans.cluster_centers_

            # Iteratively add more documents per cluster until coverage ≥ 110%
            selected_texts = []
            selected_indices = []
            selected_filtered_indices = []
            max_docs_per_cluster = max(1, min(32, len(valid_texts)))  # Allow deeper sampling for short corpora

            for docs_per_cluster in range(1, max_docs_per_cluster + 1):
                selected_texts = []
                selected_indices = []
                selected_filtered_indices = []

                for cluster_id in range(n_samples):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_indices = np.where(cluster_mask)[0]

                    if len(cluster_indices) == 0:
                        continue

                    cluster_features = features[cluster_indices]
                    centroid = centroids[cluster_id]
                    distances = np.linalg.norm(cluster_features - centroid, axis=1)
                    top_k = min(docs_per_cluster, len(cluster_indices))
                    closest = np.argsort(distances)[:top_k]

                    for idx in closest:
                        filtered_idx = cluster_indices[idx]
                        original_idx = valid_indices[filtered_idx]
                        selected_filtered_indices.append(int(filtered_idx))
                        selected_indices.append(int(original_idx))
                        selected_texts.append(valid_texts[filtered_idx])

                # Check coverage
                total_tokens = sum(
                    len(self.tokenizer.encode(text, add_special_tokens=False))
                    for text in selected_texts
                )
                coverage = total_tokens / max(1, target_tokens)

                print(f"    Trying {docs_per_cluster} docs/cluster: {len(selected_texts)} docs, coverage={coverage:.2%}")

                if coverage >= coverage_threshold:
                    print(f"  ✓ Coverage guard satisfied with {docs_per_cluster} docs per cluster")
                    best_selection["selected_texts"] = selected_texts
                    best_selection["selected_indices"] = selected_indices
                    best_selection["selected_filtered_indices"] = selected_filtered_indices
                    best_selection["total_tokens"] = total_tokens
                    best_selection["coverage"] = coverage
                    best_selection["docs_per_cluster"] = docs_per_cluster
                    break

            if coverage < coverage_threshold:
                print(f"  ⚠️  Could not reach {coverage_threshold:.2%} coverage even with {max_docs_per_cluster} docs/cluster")
                print(f"  Proceeding with best effort: {coverage:.2%}")
                # Use the last iteration (max docs) as best effort
                best_selection["selected_texts"] = selected_texts
                best_selection["selected_indices"] = selected_indices
                best_selection["selected_filtered_indices"] = selected_filtered_indices
                best_selection["total_tokens"] = total_tokens
                best_selection["coverage"] = coverage
                best_selection["docs_per_cluster"] = max_docs_per_cluster

        selected_texts = best_selection["selected_texts"]
        selected_indices = best_selection["selected_indices"]

        # Tokenize
        print(f"\nStep 5: Tokenizing (seq_len={seq_len})...")
        calibration_data = []

        all_text = "\n\n".join(selected_texts)
        all_tokens = self.tokenizer(
            all_text,
            return_tensors='pt',
            add_special_tokens=False
        )['input_ids'].squeeze(0)

        n_tokens = len(all_tokens)
        n_chunks = min(n_samples, n_tokens // seq_len)

        # Non-overlapping chunks
        for i in range(n_chunks):
            start = i * seq_len
            chunk = all_tokens[start:start + seq_len]
            calibration_data.append({
                'input_ids': chunk,
                'attention_mask': torch.ones_like(chunk)
            })

        # Overlapping chunks if needed
        if len(calibration_data) < n_samples:
            missing = n_samples - len(calibration_data)
            print(f"  Adding {missing} overlapping chunks...")
            for _ in range(missing):
                max_start = max(0, n_tokens - seq_len)
                start = int(self.rng.randint(0, max_start + 1)) if max_start > 0 else 0
                chunk = all_tokens[start:start + seq_len]
                calibration_data.append({
                    'input_ids': chunk,
                    'attention_mask': torch.ones_like(chunk)
                })

        print(f"\n{'='*60}")
        print(f"BALANCED selection complete: {len(calibration_data)} samples")
        print(f"{'='*60}\n")

        metadata = {
            "doc_indices": selected_indices,
            "filtered_doc_indices": best_selection.get("selected_filtered_indices", []),
            "seq_len": seq_len,
            "coverage": best_selection["coverage"],
            "method": "balanced",
            "tfidf_dims": self.tfidf_dims,
            "feature_weights": self.feature_weights
        }

        return calibration_data, metadata
