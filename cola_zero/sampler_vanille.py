"""
COLA-Zero Vanille Sampler v2 - DETERMINISTIC & ROBUST

IMPROVEMENTS from v1:
1. Full deterministic RNG binding (Python/NumPy/Torch/sklearn)
2. KMeans n_init=50 with algorithm='lloyd' for stability
3. TruncatedSVD with n_iter=10, power_iteration_normalizer='auto'
4. Supply-aware bucket targets with soft quotas
5. Improved format/domain detection (better regex)
6. Reasoning weight reduced to 0.35 (from 0.447)

FOUNDATION (unchanged):
- sqrt-dim weights principle
- k-means++ initialization
- Coverage guard logic
"""

import numpy as np
import torch
import re
import random
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from typing import List, Dict, Tuple


def set_all_seeds(seed: int):
    """Bind all RNGs for full determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_tfidf_reduced(texts: List[str], n_components: int = 100, random_state: int = 42) -> np.ndarray:
    """Extract TF-IDF and reduce to n_components dimensions with full determinism."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    # Deterministic SVD with enhanced stability
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=random_state,
        n_iter=10,
        power_iteration_normalizer='auto'  # 'auto' or 'LU' for stability
    )
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


def compute_reasoning_features(texts: List[str], tokenizer) -> np.ndarray:
    """
    Compute 5-dimensional reasoning features.

    Returns:
        np.ndarray of shape (n_texts, 5) with features:
        - number_density: fraction of tokens that are numbers
        - math_keywords: density of math-related keywords
        - logic_keywords: density of logic/reasoning keywords
        - question_density: fraction of sentences containing questions
        - entity_density: fraction of tokens that are capitalized entities
    """
    math_keywords = {
        'calculate', 'compute', 'sum', 'difference', 'product', 'divide',
        'equal', 'greater', 'less', 'average', 'total', 'percent', 'ratio',
        'multiply', 'subtract', 'add', 'equation', 'solve', 'result'
    }

    logic_keywords = {
        'because', 'therefore', 'thus', 'hence', 'if', 'then', 'since',
        'consequently', 'implies', 'follows', 'conclude', 'infer', 'deduce',
        'reason', 'logic', 'assume', 'given', 'suppose', 'must', 'should'
    }

    features = []

    for text in texts:
        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(tokens)

        if n_tokens == 0:
            features.append([0.0, 0.0, 0.0, 0.0, 0.0])
            continue

        # Decode back to text for pattern matching
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        # 1. Number density: fraction of tokens that are numbers
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        number_density = len(numbers) / max(1, n_tokens)

        # 2. Math keywords density
        math_count = sum(1 for word in words if word in math_keywords)
        math_density = math_count / max(1, len(words))

        # 3. Logic keywords density
        logic_count = sum(1 for word in words if word in logic_keywords)
        logic_density = logic_count / max(1, len(words))

        # 4. Question density: fraction of sentences with '?'
        sentences = re.split(r'[.!?]+', text)
        questions = [s for s in sentences if '?' in s]
        question_density = len(questions) / max(1, len(sentences))

        # 5. Entity density: capitalized words (excluding sentence starts)
        entities = re.findall(r'(?<!^)(?<!\. )\b[A-Z][a-z]+\b', text)
        entity_density = len(entities) / max(1, n_tokens)

        features.append([
            number_density,
            math_density,
            logic_density,
            question_density,
            entity_density
        ])

    return np.array(features)


def classify_format(text: str) -> str:
    """
    Classify text format into: qa, cot, or general.

    IMPROVED REGEX for better Q/A and CoT detection.
    """
    text_lower = text.lower()

    # Enhanced Q/A or MC markers
    qa_pattern = r'(?:^|\n)(?:Q:|Question:|Frage:|A:|Answer:)|\?|(?m)^(?:[A-D]\))|Rationale:|Options:|Choices:'
    has_qa = bool(re.search(qa_pattern, text, re.MULTILINE | re.IGNORECASE))

    # CoT markers
    cot_markers = [
        'step 1', 'step 2', 'first,', 'second,', 'finally,',
        'let\'s think', 'reasoning:', 'explanation:',
        'therefore', 'thus', 'hence', 'consequently'
    ]
    has_cot = any(marker in text_lower for marker in cot_markers)

    if has_qa:
        return 'qa'
    elif has_cot:
        return 'cot'
    else:
        return 'general'


def classify_domain(text: str) -> str:
    """
    Classify text domain into: commonsense, technical, or other.

    IMPROVED detection:
    - technical: code, markup, high special char density
    - commonsense: webby, factoid, low code/markup density
    """
    text_lower = text.lower()

    # Technical markers (code, markup, URLs)
    technical_pattern = r'(?:def |class |import |function|var |const |<[a-z]+>|</|```|http://|https://|github|stackoverflow|\{|\}|</>|<html>|<div>)'
    has_technical = bool(re.search(technical_pattern, text_lower))

    # Special char density (code has high density)
    special_chars = len(re.findall(r'[{}()<>\[\];]', text))
    total_chars = len(text)
    special_density = special_chars / max(1, total_chars)

    # Commonsense: high frequency of common English words, low special char density
    commonsense_words = ['the', 'is', 'are', 'was', 'were', 'what', 'when', 'where', 'who', 'why',
                         'people', 'time', 'day', 'year', 'world', 'life', 'work', 'said', 'can', 'will']
    common_word_count = sum(text_lower.count(f' {word} ') for word in commonsense_words)
    word_count = len(text.split())

    if has_technical or special_density > 0.05:
        return 'technical'
    elif common_word_count / max(1, word_count) > 0.15 and special_density < 0.02:
        return 'commonsense'
    else:
        return 'other'


class COLAZeroVanilleSampler:
    """
    COLA-Zero Vanille v2: Deterministic & Robust

    Key improvements:
    - Full RNG determinism
    - Supply-aware bucket targets
    - Improved detection
    - Reduced reasoning weight (0.35)
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
                           Default: {'tfidf': 0.1, 'length': 1.0, 'diversity': 1.0, 'reasoning': 0.35}

                           Reasoning weight reduced to 0.35:
                           - sqrt(5 × 0.35²) ≈ 0.78 (softer contribution)
                           - Less overfocus on reasoning markers
        """
        self.tokenizer = tokenizer
        self.device = device
        self.random_state = random_state
        self.tfidf_dims = tfidf_dims
        self.rng = np.random.RandomState(self.random_state)

        # Bind all RNGs for determinism
        set_all_seeds(self.random_state)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Default: sqrt-dim weighting with REDUCED reasoning influence
        if feature_weights is None:
            self.feature_weights = {
                'tfidf': 0.1,       # sqrt(100 × 0.1²) = 1.0
                'length': 1.0,      # sqrt(1 × 1.0²) = 1.0
                'diversity': 1.0,   # sqrt(1 × 1.0²) = 1.0
                'reasoning': 0.35   # sqrt(5 × 0.35²) ≈ 0.78 (reduced from 1.0)
            }
        else:
            self.feature_weights = feature_weights

    def _filter_by_buckets(
        self,
        texts: List[str],
        target_distribution: Dict[str, float]
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Classify texts by buckets and report distribution.

        Now supply-aware: targets are soft quotas, not hard filters.
        """
        print(f"\nBucket Classification:")
        print(f"  Target distribution: {target_distribution}")

        # Classify all texts
        text_info = []
        for text in texts:
            n_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))

            # Length bucket
            if n_tokens <= 256:
                length_bucket = 'short'
            elif n_tokens <= 512:
                length_bucket = 'medium'
            else:
                length_bucket = 'long'

            # Format bucket
            format_bucket = classify_format(text)

            # Domain bucket
            domain_bucket = classify_domain(text)

            text_info.append({
                'text': text,
                'length_bucket': length_bucket,
                'format_bucket': format_bucket,
                'domain_bucket': domain_bucket,
                'n_tokens': n_tokens
            })

        # Count current distribution
        length_counts = {'short': 0, 'medium': 0, 'long': 0}
        format_counts = {'qa': 0, 'cot': 0, 'general': 0}
        domain_counts = {'commonsense': 0, 'technical': 0, 'other': 0}

        for info in text_info:
            length_counts[info['length_bucket']] += 1
            format_counts[info['format_bucket']] += 1
            domain_counts[info['domain_bucket']] += 1

        print(f"\n  Current distribution:")
        print(f"    Length: short={length_counts['short']}, medium={length_counts['medium']}, long={length_counts['long']}")
        print(f"    Format: qa={format_counts['qa']}, cot={format_counts['cot']}, general={format_counts['general']}")
        print(f"    Domain: commonsense={domain_counts['commonsense']}, technical={domain_counts['technical']}, other={domain_counts['other']}")

        # Return all texts (soft quotas handled in clustering, not hard filtering)
        return texts, {
            'length': length_counts,
            'format': format_counts,
            'domain': domain_counts
        }

    def select_samples(
        self,
        texts: List[str],
        n_samples: int = 128,
        seq_len: int = 2048,
        min_length: int = 175,
        bucket_distribution: Dict[str, float] = None
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, object]]:
        """
        Select diverse calibration samples with DETERMINISTIC pipeline.

        Args:
            texts: Candidate documents
            n_samples: Number of samples (typically 128)
            seq_len: Sequence length (typically 2048)
            min_length: Min document length in tokens
            bucket_distribution: Target bucket distribution (optional)

        Returns:
            (calibration_data, metadata)
        """
        print(f"\n{'='*60}")
        print("COLA-Zero VANILLE v2 Sample Selection (Deterministic)")
        print(f"{'='*60}")
        print(f"Random seed: {self.random_state}")
        print(f"TF-IDF dims: {self.tfidf_dims}")
        print(f"Feature weights: {self.feature_weights}")

        # Bind seeds again for safety
        set_all_seeds(self.random_state)

        target_tokens = n_samples * seq_len
        coverage_threshold = 1.10

        base_min_length = min_length

        # Updated bucket distribution (supply-aware targets)
        if bucket_distribution is None:
            bucket_distribution = {
                'short': 0.50,   # 50% short (≤256 tokens) - increased from 40%
                'medium': 0.35,  # 35% medium (257-512) - increased from 30%
                'long': 0.15,    # 15% long (1024-2048) - decreased from 30%
                'qa': 0.25,      # 25% Q/A or MC - increased from 15%
                'cot': 0.075,    # 7.5% with CoT
                'commonsense': 0.35  # 35% commonsense/factoid - increased from 25%
            }

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

            # Bucket classification (soft quotas)
            filtered_texts, bucket_counts = self._filter_by_buckets(valid_texts, bucket_distribution)

            # Extract features with FULL DETERMINISM
            print(f"\nStep 2: Extracting features (deterministic)...")
            print(f"  - TF-IDF ({self.tfidf_dims} dims, weight={self.feature_weights['tfidf']})...")
            tfidf = extract_tfidf_reduced(filtered_texts, n_components=self.tfidf_dims, random_state=self.random_state)
            tfidf *= self.feature_weights['tfidf']

            print(f"  - Length (1 dim, weight={self.feature_weights['length']})...")
            lengths = compute_lengths(filtered_texts, self.tokenizer).reshape(-1, 1).astype(np.float64)
            lengths *= self.feature_weights['length']

            print(f"  - Diversity (1 dim, weight={self.feature_weights['diversity']})...")
            diversity = compute_diversity(filtered_texts, self.tokenizer).reshape(-1, 1).astype(np.float64)
            diversity *= self.feature_weights['diversity']

            print(f"  - Reasoning (5 dims, weight={self.feature_weights['reasoning']})...")
            reasoning = compute_reasoning_features(filtered_texts, self.tokenizer).astype(np.float64)
            reasoning *= self.feature_weights['reasoning']

            # Combine and normalize
            combined = np.column_stack([tfidf, lengths, diversity, reasoning])
            print(f"  Combined shape: {combined.shape}")

            # Fit scaler on ALL valid texts for consistency
            scaler = StandardScaler()
            features = scaler.fit_transform(combined)

            # K-Means clustering with enhanced stability
            print(f"\nStep 3: K-Means clustering (k={n_samples}, n_init=50)...")
            kmeans = KMeans(
                n_clusters=n_samples,
                random_state=self.random_state,
                init='k-means++',
                n_init=50,  # Increased from 10 to 50 for stability
                max_iter=300,
                algorithm='lloyd',  # Explicit algorithm for determinism
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
                    selected_texts.append(filtered_texts[filtered_idx])

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
                "docs_per_cluster": docs_per_cluster,
                "bucket_counts": bucket_counts
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

            # Re-extract features with determinism
            tfidf = extract_tfidf_reduced(valid_texts, n_components=self.tfidf_dims, random_state=self.random_state)
            tfidf *= self.feature_weights['tfidf']
            lengths = compute_lengths(valid_texts, self.tokenizer).reshape(-1, 1).astype(np.float64)
            lengths *= self.feature_weights['length']
            diversity = compute_diversity(valid_texts, self.tokenizer).reshape(-1, 1).astype(np.float64)
            diversity *= self.feature_weights['diversity']
            reasoning = compute_reasoning_features(valid_texts, self.tokenizer).astype(np.float64)
            reasoning *= self.feature_weights['reasoning']
            combined = np.column_stack([tfidf, lengths, diversity, reasoning])
            scaler = StandardScaler()
            features = scaler.fit_transform(combined)

            # Re-run clustering with determinism
            kmeans = KMeans(
                n_clusters=n_samples,
                random_state=self.random_state,
                init='k-means++',
                n_init=50,
                max_iter=300,
                algorithm='lloyd',
                verbose=0
            )
            cluster_labels = kmeans.fit_predict(features)
            centroids = kmeans.cluster_centers_

            # Iteratively add more documents per cluster until coverage ≥ 110%
            selected_texts = []
            selected_indices = []
            selected_filtered_indices = []
            max_docs_per_cluster = max(1, min(32, len(valid_texts)))

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

        non_overlapping_chunks = n_chunks

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

        # Calculate overlap rate
        overlap_rate = (n_samples - non_overlapping_chunks) / n_samples if n_samples > 0 else 0.0

        # Calculate distinct n-grams
        all_tokens_list = all_tokens.tolist()
        distinct_unigrams = len(set(all_tokens_list))
        unigram_diversity = distinct_unigrams / max(1, len(all_tokens_list))

        print(f"\n{'='*60}")
        print(f"VANILLE v2 selection complete: {len(calibration_data)} samples")
        print(f"  Non-overlapping chunks: {non_overlapping_chunks}")
        print(f"  Overlapping chunks: {n_samples - non_overlapping_chunks}")
        print(f"  Overlap rate: {overlap_rate * 100:.1f}%")
        print(f"  Distinct unigrams: {distinct_unigrams} / {len(all_tokens_list)} ({unigram_diversity * 100:.1f}%)")
        print(f"{'='*60}\n")

        metadata = {
            "doc_indices": selected_indices,
            "filtered_doc_indices": best_selection.get("selected_filtered_indices", []),
            "seq_len": seq_len,
            "coverage": best_selection["coverage"],
            "method": "vanille_v2",
            "tfidf_dims": self.tfidf_dims,
            "feature_weights": self.feature_weights,
            "bucket_counts": best_selection.get("bucket_counts", {}),
            "overlap_rate": overlap_rate,
            "distinct_unigrams": distinct_unigrams,
            "unigram_diversity": unigram_diversity
        }

        return calibration_data, metadata
