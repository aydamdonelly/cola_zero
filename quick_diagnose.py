"""
QUICK diagnostic: Is TF-IDF dominating?

Minimal imports, runs in <1 minute.
Skips perplexity (too slow), uses mock data.
"""

import sys
print("Starting imports...", flush=True)

import numpy as np
print("✓ numpy", flush=True)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
print("✓ sklearn", flush=True)

from datasets import load_dataset
print("✓ datasets", flush=True)

from transformers import AutoTokenizer
print("✓ transformers", flush=True)

from cola_zero.features import extract_tfidf_features, compute_sequence_lengths, compute_vocabulary_diversity
print("✓ cola_zero features", flush=True)

print("\n" + "="*60)
print("QUICK FEATURE DOMINANCE TEST")
print("="*60)

# Load small dataset for speed
print("\n1. Loading data (first 500 docs only)...", flush=True)
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
texts = [ex['text'] for ex in dataset if ex['text'].strip()]
texts = texts[:500]  # SMALL for speed
print(f"   Loaded {len(texts)} documents")

# Use GPT-2 tokenizer (no auth required) or specify your own
# For diagnostics, the specific tokenizer doesn't matter much
import sys
TOKENIZER_NAME = sys.argv[1] if len(sys.argv) > 1 else 'gpt2'
print(f"   Loading tokenizer: {TOKENIZER_NAME}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
print("   ✓ Tokenizer loaded")

# Extract features
print("\n2. Extracting features...", flush=True)

print("   - TF-IDF (5000 dims)...", flush=True)
tfidf = extract_tfidf_features(texts, max_features=5000)
print(f"     Shape: {tfidf.shape}")

print("   - Length (1 dim)...", flush=True)
lengths = compute_sequence_lengths(texts, tokenizer).reshape(-1, 1)
print(f"     Shape: {lengths.shape}")

print("   - Diversity (1 dim)...", flush=True)
diversity = compute_vocabulary_diversity(texts, tokenizer).reshape(-1, 1)
print(f"     Shape: {diversity.shape}")

# Mock perplexity (just use random values for speed)
print("   - Perplexity (1 dim, MOCKED for speed)...", flush=True)
ppl = np.random.randn(len(texts), 1)
print(f"     Shape: {ppl.shape}")

# Test different combinations
print("\n3. Testing K-Means clustering with different feature sets...", flush=True)

n_clusters = 64  # Smaller for speed
seed = 42

configs = [
    ("Full (5000+3)", np.column_stack([tfidf, ppl, lengths, diversity])),
    ("TF-IDF only (5000)", tfidf),
    ("Non-TF-IDF only (3)", np.column_stack([ppl, lengths, diversity])),
]

results = {}

for name, features in configs:
    print(f"\n   Testing: {name}")
    print(f"   Shape: {features.shape}")

    # Normalize
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=5, max_iter=100)
    labels = kmeans.fit_predict(normalized)

    inertia = kmeans.inertia_
    unique_labels = len(set(labels))

    print(f"   Inertia: {inertia:.2f}")
    print(f"   Unique clusters: {unique_labels}/{n_clusters}")
    print(f"   Avg dist to centroid: {inertia/len(normalized):.4f}")

    results[name] = {
        "inertia": inertia,
        "labels": labels
    }

# Compare
print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

full_inertia = results["Full (5000+3)"]["inertia"]
tfidf_only_inertia = results["TF-IDF only (5000)"]["inertia"]
non_tfidf_inertia = results["Non-TF-IDF only (3)"]["inertia"]

diff = abs(full_inertia - tfidf_only_inertia) / full_inertia * 100

print(f"\nInertia comparison:")
print(f"  Full (5000+3):       {full_inertia:.2f}")
print(f"  TF-IDF only (5000):  {tfidf_only_inertia:.2f}")
print(f"  Non-TF-IDF only (3): {non_tfidf_inertia:.2f}")
print(f"\nDifference (Full vs TF-IDF only): {diff:.2f}%")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if diff < 5:
    print("❌ PROBLEM CONFIRMED: TF-IDF is dominating!")
    print("   The 'Full' and 'TF-IDF only' results are nearly identical.")
    print("   Your other 3 features have NEGLIGIBLE impact on clustering.")
    print("\n   This explains why COLA-Zero ≈ Random!")
    print("\n   SOLUTIONS:")
    print("   1. Reduce TF-IDF to ~100 dims via PCA/SVD")
    print("   2. Weight non-TF-IDF features 100x higher")
    print("   3. Use activation-based features instead")
elif diff < 15:
    print("⚠️  TF-IDF is VERY dominant, but other features have SOME effect")
    print("   Consider rebalancing anyway.")
else:
    print("✅ Features seem reasonably balanced")
    print("   TF-IDF is not dominating completely.")
    print("   The problem might be elsewhere.")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("1. If TF-IDF dominates → Use features_v2.py with balanced features")
print("2. Run: python experiments/01_quantize_cola_zero.py --method balanced")
print("3. Compare results with random baseline again")
print("="*60)
