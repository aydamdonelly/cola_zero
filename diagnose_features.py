"""
Diagnostic script to understand why COLA-Zero = Random.

Tests:
1. Feature contribution to K-Means distances
2. Ablation: What happens with only non-TF-IDF features?
3. Cluster quality metrics
4. Actual diversity of selected samples
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from transformers import AutoTokenizer
from datasets import load_dataset
from cola_zero.features import extract_all_features, extract_tfidf_features, estimate_perplexity
from cola_zero.features import compute_sequence_lengths, compute_vocabulary_diversity
from cola_zero.sampler import COLAZeroSampler
from cola_zero.baselines import RandomSampler
import matplotlib.pyplot as plt


def analyze_feature_importance(features_dict, n_clusters=128, seed=42):
    """
    Analyze how much each feature type contributes to K-Means clustering.
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    results = {}

    for name, feature_matrix in features_dict.items():
        print(f"\nTesting: {name}")
        print(f"  Shape: {feature_matrix.shape}")

        # Normalize
        scaler = StandardScaler()
        normalized = scaler.fit_transform(feature_matrix)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(normalized)

        # Metrics
        inertia = kmeans.inertia_
        unique_labels = len(set(labels))
        silhouette_approx = -inertia / len(normalized)  # Rough proxy

        print(f"  Inertia: {inertia:.2f}")
        print(f"  Unique clusters: {unique_labels}/{n_clusters}")
        print(f"  Avg distance to centroid: {inertia/len(normalized):.4f}")

        results[name] = {
            "inertia": inertia,
            "unique_clusters": unique_labels,
            "labels": labels
        }

    return results


def compare_selected_samples(tokenizer, texts, n_samples=128, seq_len=2048):
    """
    Compare actual diversity of COLA-Zero vs Random selections.
    """
    print("\n" + "="*60)
    print("SAMPLE DIVERSITY COMPARISON")
    print("="*60)

    # COLA-Zero selection
    print("\n1. COLA-Zero Selection:")
    cola_sampler = COLAZeroSampler(tokenizer, device='cpu', include_perplexity=False)
    cola_data, cola_meta = cola_sampler.select_samples(texts, n_samples=n_samples, seq_len=seq_len)
    cola_indices = set(cola_meta['doc_indices'])

    # Random selection
    print("\n2. Random Selection:")
    random_sampler = RandomSampler(tokenizer, random_state=42)
    random_data, random_meta = random_sampler.select_samples(texts, n_samples=n_samples, seq_len=seq_len)
    random_indices = set(random_meta['doc_indices'])

    # Overlap analysis
    overlap = cola_indices & random_indices
    print(f"\n3. Overlap Analysis:")
    print(f"  COLA-Zero selected: {len(cola_indices)} unique documents")
    print(f"  Random selected: {len(random_indices)} unique documents")
    print(f"  Overlap: {len(overlap)} documents ({len(overlap)/len(cola_indices)*100:.1f}%)")

    # If >50% overlap, selections are basically the same!
    if len(overlap) / len(cola_indices) > 0.5:
        print("\n⚠️  WARNING: >50% overlap! COLA-Zero is barely different from random!")

    return cola_indices, random_indices, overlap


def test_feature_weights():
    """
    Main diagnostic entry point.
    """
    print("Loading WikiText-2 train split...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [ex['text'] for ex in dataset if ex['text'].strip()]
    texts = texts[:1000]  # Use first 1000 docs for speed

    print(f"Loaded {len(texts)} documents\n")

    # Use GPT-2 (no auth needed) or pass your model name as arg
    import sys
    tokenizer_name = sys.argv[1] if len(sys.argv) > 1 else 'gpt2'
    print(f"Using tokenizer: {tokenizer_name}\n")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Extract features individually
    print("Extracting features...")
    tfidf = extract_tfidf_features(texts, max_features=5000)
    print(f"TF-IDF: {tfidf.shape}")

    # NOTE: Perplexity is slow, skip for diagnostics
    # ppl = estimate_perplexity(texts, device='cpu')
    # For speed, use random values as proxy
    ppl = np.random.randn(len(texts), 1)
    print(f"Perplexity (mocked): {ppl.shape}")

    lengths = compute_sequence_lengths(texts, tokenizer).reshape(-1, 1)
    print(f"Lengths: {lengths.shape}")

    diversity = compute_vocabulary_diversity(texts, tokenizer).reshape(-1, 1)
    print(f"Diversity: {diversity.shape}")

    # Test different feature combinations
    features_dict = {
        "Full (5000+3)": np.column_stack([tfidf, ppl, lengths, diversity]),
        "TF-IDF only (5000)": tfidf,
        "Non-TF-IDF only (3)": np.column_stack([ppl, lengths, diversity]),
        "TF-IDF reduced (100+3)": np.column_stack([tfidf[:, :100], ppl, lengths, diversity]),
        "TF-IDF reduced (10+3)": np.column_stack([tfidf[:, :10], ppl, lengths, diversity]),
    }

    # Analyze feature importance
    results = analyze_feature_importance(features_dict, n_clusters=128)

    # Compare actual selections
    print("\n" + "="*60)
    print("Testing actual sample selection overlap...")
    compare_selected_samples(tokenizer, texts, n_samples=128, seq_len=2048)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nIf you see:")
    print("1. Similar inertia for 'Full' and 'TF-IDF only' → TF-IDF dominates")
    print("2. High overlap (>50%) → COLA-Zero ≈ Random")
    print("3. Very different inertia for reduced TF-IDF → Dimensionality is the issue")
    print("\nNext steps:")
    print("- Try PCA to reduce TF-IDF to ~100 dims")
    print("- Try different feature weighting schemes")
    print("- Consider activation-based methods (need forward passes)")
    print("="*60)


if __name__ == '__main__':
    test_feature_weights()
