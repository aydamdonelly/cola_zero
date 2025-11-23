"""
Feature extraction module for COLA-Zero.

Extracts text-based features without using the target model:
- TF-IDF features (topic/domain diversity)
- Perplexity scores (text complexity, using small proxy model)
- Sequence lengths (context window coverage)
- Vocabulary diversity (lexical richness)
"""

import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List


def extract_tfidf_features(texts: List[str], max_features: int = 5000) -> np.ndarray:
    """
    Extract TF-IDF features from text documents.

    Args:
        texts: List of documents
        max_features: Maximum number of features (vocabulary size)

    Returns:
        np.ndarray: Shape (n_documents, max_features)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrams + bigrams
        min_df=2,            # Ignore very rare terms
        max_df=0.95,         # Ignore very common terms
        sublinear_tf=True    # Use log scaling for term frequency
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix.toarray()  # Convert sparse to dense


def estimate_perplexity(
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 256,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Estimate perplexity using a small, fast model (GPT-2 small).

    NOTE: This is NOT the target model being quantized!
    We use GPT-2 small as a proxy for text complexity.

    FIXED: Now computes per-document perplexity instead of per-batch.

    Args:
        texts: List of documents
        batch_size: Process multiple texts at once
        max_length: Maximum sequence length for perplexity computation
        device: 'cuda' or 'cpu'

    Returns:
        np.ndarray: Shape (n_documents,) - Perplexity score per document
    """
    print(f"  Loading GPT-2 small for perplexity estimation...")

    # Load small model (NOT the target model!)
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    all_perplexities = []

    # Process in batches to avoid OOM
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(device)

        # Compute perplexity PER DOCUMENT (not per batch)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])

            # Get per-token losses
            # outputs.loss is averaged, so we need to compute manually
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

            # Compute loss for each document in the batch
            for j in range(len(batch_texts)):
                # Get tokens and attention mask for this document
                input_ids = inputs['input_ids'][j]
                attention_mask = inputs['attention_mask'][j]

                # Get logits for this document
                doc_logits = logits[j]  # Shape: (seq_len, vocab_size)

                # Compute loss only on non-padded tokens
                # Shift for next-token prediction
                shift_logits = doc_logits[:-1, :]
                shift_labels = input_ids[1:]
                shift_attention = attention_mask[1:]

                # Compute cross-entropy loss
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(shift_logits, shift_labels)

                # Mask out padding tokens
                masked_losses = losses * shift_attention.float()

                # Average over non-padded tokens
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
    """
    Compute tokenized sequence lengths.

    Args:
        texts: List of documents
        tokenizer: Target model's tokenizer

    Returns:
        np.ndarray: Shape (n_documents,) - Length in tokens
    """
    lengths = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        lengths.append(len(tokens))

    return np.array(lengths)


def compute_vocabulary_diversity(texts: List[str], tokenizer) -> np.ndarray:
    """
    Compute vocabulary diversity (unique token ratio).

    Args:
        texts: List of documents
        tokenizer: Target model's tokenizer

    Returns:
        np.ndarray: Shape (n_documents,) - Diversity scores
    """
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


def extract_all_features(
    texts: List[str],
    tokenizer,
    device: str = 'cuda',
    include_perplexity: bool = True
) -> np.ndarray:
    """
    Extract and combine all text-based features.

    Args:
        texts: List of documents
        tokenizer: Target model's tokenizer
        device: Device for perplexity computation
        include_perplexity: Whether to include perplexity feature (can be disabled for faster iteration)

    Returns:
        np.ndarray: Shape (n_documents, n_features) - Normalized feature matrix
    """
    print("Extracting TF-IDF features...")
    tfidf_features = extract_tfidf_features(texts, max_features=5000)
    print(f"  TF-IDF shape: {tfidf_features.shape}")

    features_to_combine = [tfidf_features]

    if include_perplexity:
        print("Computing perplexity scores (using GPT-2 small as proxy)...")
        perplexity_scores = estimate_perplexity(texts, device=device)
        print(f"  Perplexity shape: {perplexity_scores.shape}")
        features_to_combine.append(perplexity_scores.reshape(-1, 1))
    else:
        print("  Skipping perplexity (include_perplexity=False)")

    print("Computing sequence lengths...")
    lengths = compute_sequence_lengths(texts, tokenizer)
    print(f"  Lengths shape: {lengths.shape}")
    features_to_combine.append(lengths.reshape(-1, 1))

    print("Computing vocabulary diversity...")
    diversity = compute_vocabulary_diversity(texts, tokenizer)
    print(f"  Diversity shape: {diversity.shape}")
    features_to_combine.append(diversity.reshape(-1, 1))

    # Combine all features
    combined = np.column_stack(features_to_combine)

    feature_count = 5000 + (1 if include_perplexity else 0) + 1 + 1
    print(f"  Combined feature shape: {combined.shape} (expected: (n, {feature_count}))")

    # Normalize features to same scale (critical for k-means)
    print("Normalizing features...")
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(combined)
    print(f"  Normalized shape: {normalized_features.shape}")

    return normalized_features
