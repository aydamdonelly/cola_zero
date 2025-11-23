"""
Fixed feature extraction with balanced weighting.

Key changes:
1. Reduce TF-IDF to 100 dims via PCA
2. Weight features explicitly
3. Add feature importance tracking
"""

import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Tuple


def extract_tfidf_features_reduced(
    texts: List[str],
    max_features: int = 5000,
    n_components: int = 100
) -> np.ndarray:
    """
    Extract TF-IDF and reduce dimensionality via SVD (PCA for sparse matrices).

    Args:
        texts: List of documents
        max_features: Initial TF-IDF vocabulary size
        n_components: Target dimensionality (default: 100)

    Returns:
        np.ndarray: Shape (n_documents, n_components)
    """
    print(f"  Extracting TF-IDF (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(texts)  # Sparse matrix

    print(f"  Reducing TF-IDF from {tfidf_matrix.shape[1]} to {n_components} dims via SVD...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)

    explained_variance = svd.explained_variance_ratio_.sum()
    print(f"  Explained variance: {explained_variance:.2%}")

    return tfidf_reduced


def estimate_perplexity(
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 256,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Estimate perplexity using GPT-2 small (same as before).
    """
    print(f"  Loading GPT-2 small for perplexity estimation...")

    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    all_perplexities = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            logits = outputs.logits

            for j in range(len(batch_texts)):
                input_ids = inputs['input_ids'][j]
                attention_mask = inputs['attention_mask'][j]
                doc_logits = logits[j]

                shift_logits = doc_logits[:-1, :]
                shift_labels = input_ids[1:]
                shift_attention = attention_mask[1:]

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(shift_logits, shift_labels)

                masked_losses = losses * shift_attention.float()
                num_tokens = shift_attention.sum().item()

                if num_tokens > 0:
                    avg_loss = masked_losses.sum().item() / num_tokens
                    perplexity = np.exp(avg_loss)
                else:
                    perplexity = np.inf

                all_perplexities.append(perplexity)

        if (i // batch_size + 1) % 10 == 0:
            print(f"    Processed {i + len(batch_texts)}/{len(texts)} documents")

    return np.array(all_perplexities)


def compute_sequence_lengths(texts: List[str], tokenizer) -> np.ndarray:
    """Compute tokenized sequence lengths."""
    lengths = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(tokens))

    return np.array(lengths)


def compute_vocabulary_diversity(texts: List[str], tokenizer) -> np.ndarray:
    """Compute vocabulary diversity (unique token ratio)."""
    diversity_scores = []

    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) == 0:
            diversity_scores.append(0.0)
        else:
            unique_tokens = len(set(tokens))
            total_tokens = len(tokens)
            diversity = unique_tokens / total_tokens
            diversity_scores.append(diversity)

    return np.array(diversity_scores)


def extract_all_features_balanced(
    texts: List[str],
    tokenizer,
    device: str = 'cuda',
    include_perplexity: bool = True,
    tfidf_dims: int = 100,
    feature_weights: dict = None
) -> Tuple[np.ndarray, dict]:
    """
    Extract and combine features with balanced weighting.

    KEY CHANGES:
    1. TF-IDF reduced to 100 dims (not 5000!)
    2. Explicit feature weighting
    3. Returns feature importance info

    Args:
        texts: List of documents
        tokenizer: Target model's tokenizer
        device: Device for perplexity computation
        include_perplexity: Whether to include perplexity feature
        tfidf_dims: TF-IDF dimensionality after reduction (default: 100)
        feature_weights: Optional dict with weights for each feature type
                        e.g., {'tfidf': 1.0, 'perplexity': 10.0, 'length': 5.0, 'diversity': 5.0}

    Returns:
        Tuple of:
          - np.ndarray: Shape (n_documents, n_features) - Normalized feature matrix
          - dict: Feature importance metadata
    """
    if feature_weights is None:
        # Default: Equal contribution from each feature TYPE (not dimension!)
        # TF-IDF: 100 dims → weight each by 1.0
        # Others: 1 dim each → weight by 100.0 to match TF-IDF's total mass
        feature_weights = {
            'tfidf': 1.0,
            'perplexity': 100.0,
            'length': 100.0,
            'diversity': 100.0
        }

    print("\n" + "="*60)
    print("BALANCED FEATURE EXTRACTION")
    print("="*60)

    print(f"\n1. TF-IDF (reduced to {tfidf_dims} dims, weight={feature_weights['tfidf']})...")
    tfidf_features = extract_tfidf_features_reduced(texts, n_components=tfidf_dims)
    tfidf_features *= feature_weights['tfidf']  # Apply weight
    print(f"  Shape: {tfidf_features.shape}")

    features_to_combine = [tfidf_features]
    feature_dims = {'tfidf': tfidf_dims}

    if include_perplexity:
        print(f"\n2. Perplexity (1 dim, weight={feature_weights['perplexity']})...")
        perplexity_scores = estimate_perplexity(texts, device=device)
        perplexity_scores = perplexity_scores.reshape(-1, 1) * feature_weights['perplexity']
        features_to_combine.append(perplexity_scores)
        feature_dims['perplexity'] = 1
    else:
        print("\n2. Perplexity: SKIPPED")

    print(f"\n3. Sequence Length (1 dim, weight={feature_weights['length']})...")
    lengths = compute_sequence_lengths(texts, tokenizer)
    lengths = lengths.reshape(-1, 1) * feature_weights['length']
    features_to_combine.append(lengths)
    feature_dims['length'] = 1

    print(f"\n4. Vocabulary Diversity (1 dim, weight={feature_weights['diversity']})...")
    diversity = compute_vocabulary_diversity(texts, tokenizer)
    diversity = diversity.reshape(-1, 1) * feature_weights['diversity']
    features_to_combine.append(diversity)
    feature_dims['diversity'] = 1

    # Combine
    combined = np.column_stack(features_to_combine)
    total_dims = sum(feature_dims.values())
    print(f"\n5. Combined shape: {combined.shape} (expected: (n, {total_dims}))")

    # Normalize (now weights are already applied!)
    print("\n6. Normalizing...")
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(combined)

    # Feature importance metadata
    feature_info = {
        'dimensions': feature_dims,
        'weights': feature_weights,
        'total_dims': total_dims,
        'scaler_mean': scaler.mean_,
        'scaler_std': scaler.scale_
    }

    print(f"\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    for name, dims in feature_dims.items():
        weight = feature_weights.get(name, 1.0)
        effective_contribution = dims * weight
        print(f"  {name:12s}: {dims:3d} dims × {weight:6.1f} weight = {effective_contribution:7.1f} contribution")
    print("="*60)

    return normalized_features, feature_info
