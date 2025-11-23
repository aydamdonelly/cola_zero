"""
Multi-seed experimental orchestration for COLA-Zero quantization evaluation.

This script runs systematic experiments across:
- One or more models (default: Meta-Llama-3-8B-Instruct)
- Calibration methods: random and COLA-Zero
- Multiple seeds (default: 1-50) for statistical significance

Each experiment produces a JSON file with metrics:
- Perplexity on WikiText-2 test
- Timing measurements (selection, quantization)
- Downstream task evaluation (HellaSwag, WinoGrande, PIQA, ARC-easy)

Usage:
    python -m experiments.runner

Output:
    results/metrics/raw_runs/{model}__{method}__seed{seed}.json
"""

import os
import sys
import json
import time
import torch

# Allow HuggingFace datasets with custom code (needed for math_qa)
os.environ['HF_DATASETS_TRUST_REMOTE_CODE'] = 'true'
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
import datasets as hf_datasets
from transformers import AutoTokenizer
import transformers
from gptqmodel import GPTQModel, QuantizeConfig
import sklearn

# Try to import lm-eval for downstream evaluation
try:
    from lm_eval import evaluator
    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False
    print("[RUNNER] Warning: lm-eval not installed. Downstream evaluation will be skipped.")
    print("[RUNNER] Install with: pip install lm-eval[api]")

# Check for optimum (required for lm-eval to load GPTQ models)
try:
    import optimum
    HAS_OPTIMUM = True
except ImportError:
    HAS_OPTIMUM = False
    if HAS_LM_EVAL:
        print("[RUNNER] Warning: optimum not installed. Required for lm-eval to load GPTQ models.")
        print("[RUNNER] Install with: pip install optimum auto-gptq")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cola_zero.sampler import COLAZeroSampler
from cola_zero.sampler_balanced import COLAZeroBalancedSampler
from cola_zero.sampler_vanille import COLAZeroVanilleSampler
from cola_zero.sampler_proxy import COLAZeroProxySampler
from cola_zero.baselines import RandomSampler
from experiments.calibration_sources import get_calibration_source

# Import calibration doctor for padding-leak prevention (REQUIRED)
try:
    from calibration_doctor import check_and_pack_for_gptq
    HAS_CALIB_DOCTOR = True
except ImportError:
    HAS_CALIB_DOCTOR = False
    raise RuntimeError(
        "[RUNNER] CRITICAL: calibration_doctor.py is REQUIRED to prevent padding-leakage bugs.\n"
        "Please ensure calibration_doctor.py is in the repo root."
    )


def set_all_seeds(seed: int) -> None:
    """
    Set random seeds across libraries for reproducibility.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collect_env_info() -> Dict[str, Optional[str]]:
    """
    Collect library and hardware version information for logging.
    """
    info: Dict[str, Optional[str]] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "datasets": hf_datasets.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "auto_gptq_or_gptqmodel": None
    }

    version_value: Optional[str] = None

    try:
        import auto_gptq  # type: ignore

        version_value = getattr(auto_gptq, "__version__", None)
    except ImportError:
        try:
            import gptqmodel  # type: ignore

            version_value = getattr(gptqmodel, "__version__", None)
        except ImportError:
            version_value = None

    info["auto_gptq_or_gptqmodel"] = str(version_value) if version_value is not None else None

    return info


def load_wikitext2(split='train'):
    """
    Load WikiText-2 dataset.

    Args:
        split: 'train' or 'test'

    Returns:
        List of document strings
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    documents = [text for text in dataset['text'] if len(text.strip()) > 0]
    return documents


def build_calibration_examples(
    method: str,
    tokenizer,
    documents: List[str],
    n_samples: int,
    seq_len: int,
    seed: int,
    config: Dict = None,
    calibration_source: str = "wikitext",
    tokenizer_name: Optional[str] = None
) -> Tuple[List[Dict], float, Dict[str, object]]:
    """
    Build calibration examples using specified method.

    Args:
        method: 'random' or 'cola_zero' or 'cola_zero_proxy'
        tokenizer: HuggingFace tokenizer
        documents: List of text documents
        n_samples: Number of calibration samples
        seq_len: Sequence length per sample
        seed: Random seed
        config: Experiment configuration dict (optional, needed for proxy)
        calibration_source: Calibration data source ('wikitext', 'c4', 'mathqa')
        tokenizer_name: Optional tokenizer identifier for calibration doctor fallback

    Returns:
        (examples_list, selection_time_sec, metadata_dict)
    """
    print(f"[RUNNER] Building calibration data: method={method}, source={calibration_source}, seed={seed}")

    start_time = time.time()
    meta: Dict[str, object] = {}

    # Determine min_length based on calibration source
    # MathQA has shorter documents (avg ~280 tokens), so we use a lower threshold
    if calibration_source == "mathqa":
        min_length_tokens = 50  # Lower threshold for MathQA
        print(f"[RUNNER] Using min_length={min_length_tokens} tokens for {calibration_source} (short corpus)")
    else:
        min_length_tokens = 175  # Default for C4/WikiText
        print(f"[RUNNER] Using min_length={min_length_tokens} tokens for {calibration_source}")

    if method == "random":
        sampler = RandomSampler(tokenizer, random_state=seed)
        examples, meta = sampler.select_samples(documents, n_samples, seq_len)

    elif method == "cola_zero":
        # Use BALANCED sampler with corrected sqrt-dim feature weights
        sampler = COLAZeroBalancedSampler(
            tokenizer=tokenizer,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            random_state=seed,
            tfidf_dims=100  # Reduced from 5000 to 100
            # Using default feature_weights: {'tfidf': 0.1, 'length': 1.0, 'diversity': 1.0}
            # This ensures equal L2 contribution: sqrt(100×0.1²) = sqrt(1×1.0²) = sqrt(1×1.0²) = 1.0
        )
        examples, meta = sampler.select_samples(
            documents,
            n_samples,
            seq_len,
            min_length=min_length_tokens
        )

    elif method == "cola_zero_vanille":
        # Use VANILLE sampler with reasoning features and bucket-based selection
        sampler = COLAZeroVanilleSampler(
            tokenizer=tokenizer,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            random_state=seed,
            tfidf_dims=100  # Reduced from 5000 to 100
            # Using default feature_weights: {'tfidf': 0.1, 'length': 1.0, 'diversity': 1.0, 'reasoning': 0.447}
            # This ensures equal L2 contribution:
            #   sqrt(100×0.1²) = 1.0 (TF-IDF)
            #   sqrt(1×1.0²) = 1.0 (length)
            #   sqrt(1×1.0²) = 1.0 (diversity)
            #   sqrt(5×0.447²) ≈ 1.0 (reasoning: 5 features)
        )
        examples, meta = sampler.select_samples(
            documents,
            n_samples,
            seq_len,
            min_length=min_length_tokens
        )

    elif method == "cola_zero_proxy":
        # Use PROXY sampler with small LLM perplexity as feature
        config = config or {}  # Handle None case
        proxy_model = config.get("proxy_model", "gpt2")  # Default: GPT-2 (124M)
        proxy_max_length = config.get("proxy_max_length", 512)

        sampler = COLAZeroProxySampler(
            tokenizer=tokenizer,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            random_state=seed,
            tfidf_dims=100,
            proxy_model_name=proxy_model
            # Using default feature_weights: {'tfidf': 0.1, 'length': 1.0, 'diversity': 1.0, 'proxy_ppl': 1.0}
            # This ensures equal L2 contribution:
            #   sqrt(100×0.1²) = 1.0 (TF-IDF)
            #   sqrt(1×1.0²) = 1.0 (length)
            #   sqrt(1×1.0²) = 1.0 (diversity)
            #   sqrt(1×1.0²) = 1.0 (proxy_ppl: activation-aware feature)
        )
        examples, meta = sampler.select_samples(
            documents,
            n_samples,
            seq_len,
            min_length=min_length_tokens,
            proxy_max_length=proxy_max_length
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    selection_time = time.time() - start_time
    meta = meta or {}
    meta["method"] = method
    meta["calibration_source"] = calibration_source

    # Apply calibration doctor to prevent padding leakage (CRITICAL for C4/noisy sources!)
    # IMPORTANT: Random baseline MUST skip doctor (sequential packing breaks random sampling!)
    if method == "random":
        print(f"[RUNNER] ⚠️  Skipping calibration doctor for Random (preserves random sampling distribution)")
        print(f"[RUNNER]    Random sampler handles padding internally via attention_mask")
        skip_doctor = True
    else:
        skip_doctor = False

    if HAS_CALIB_DOCTOR and not skip_doctor:
        print(f"[RUNNER] Applying calibration doctor checks...")

        # Extract selected texts based on method
        # For Random: we use all documents (it concatenates them)
        # For COLA-Zero variants: we use the selected doc indices from metadata
        if method == "random":
            selected_texts = documents  # Random concatenates all texts
        elif "cola_zero" in method:
            # Get selected document indices from metadata
            doc_indices = meta.get("doc_indices", [])
            if not doc_indices:
                print(f"[RUNNER] ⚠️  Warning: No doc_indices in metadata, using all documents")
                selected_texts = documents
            else:
                selected_texts = [documents[idx] for idx in doc_indices]
        else:
            # For other methods, try to use all documents
            selected_texts = documents

        print(f"[RUNNER]    Initial pool: {len(selected_texts)} documents")

        # Get tokenizer name from config or provided value for doctor fallback
        config_models = (config or {}).get('models') or []
        tokenizer_name_for_doctor = tokenizer_name or getattr(tokenizer, "name_or_path", None)
        if not tokenizer_name_for_doctor and config_models:
            tokenizer_name_for_doctor = config_models[0]
        if not tokenizer_name_for_doctor:
            tokenizer_name_for_doctor = "facebook/opt-6.7b"

        # For COLA-Zero: Adjust min_total_ratio to avoid bad top-up docs
        # (Long docs from rest-pool are often boilerplate/junk that COLA-Zero rejected)
        min_total_ratio_to_use = 1.10  # Default for Random

        if "cola_zero" in method and len(selected_texts) < len(documents):
            temp_tokenizer = tokenizer

            # Check if selected docs have enough tokens with relaxed ratio
            needed_tokens_strict = int(seq_len * n_samples * 1.10)
            needed_tokens_relaxed = int(seq_len * n_samples * 1.00)

            current_tokens = sum(
                len(temp_tokenizer(t, add_special_tokens=False)['input_ids'])
                for t in selected_texts
            )

            print(f"[RUNNER]    Token budget: {current_tokens:,} / {needed_tokens_strict:,} (strict 1.10)")
            print(f"[RUNNER]    Token budget: {current_tokens:,} / {needed_tokens_relaxed:,} (relaxed 1.00)")

            if current_tokens < needed_tokens_relaxed:
                # Check if Coverage Guard already tried max docs (source is too short)
                docs_per_cluster = meta.get("docs_per_cluster", 0)
                coverage_from_meta = meta.get("coverage", 0.0)

                if docs_per_cluster >= 10:
                    # Coverage Guard already maxed out - source is fundamentally too short
                    # Use ultra-relaxed ratio based on actual coverage
                    min_total_ratio_to_use = max(0.5, coverage_from_meta * 0.9)  # 90% of achieved coverage
                    print(f"[RUNNER]    ⚠️  Source too short (Coverage Guard maxed at {docs_per_cluster} docs/cluster)")
                    print(f"[RUNNER]    ⚠️  Using ultra-relaxed ratio: {min_total_ratio_to_use:.2f} (based on {coverage_from_meta:.1%} coverage)")
                else:
                    # Really not enough and didn't max out - should not happen
                    print(f"[RUNNER]    ❌ Insufficient even with relaxed ratio - would need top-up")
                    print(f"[RUNNER]    ⚠️  This indicates COLA-Zero selection is too restrictive for this source")
                    raise RuntimeError(f"COLA-Zero selected docs have insufficient tokens: {current_tokens} < {needed_tokens_relaxed}")

            elif current_tokens < needed_tokens_strict:
                # Enough with relaxed ratio - use 1.00 instead of 1.10
                print(f"[RUNNER]    ⚠️  Token budget tight, using relaxed ratio (1.00 instead of 1.10)")
                min_total_ratio_to_use = 1.00
            else:
                print(f"[RUNNER]    ✅ Token budget sufficient for strict ratio (1.10)")
                min_total_ratio_to_use = 1.10

        print(f"[RUNNER]    Final pool for repacking: {len(selected_texts)} documents")

        try:
            packed = check_and_pack_for_gptq(
                texts=selected_texts,
                tokenizer_name=tokenizer_name_for_doctor,
                seq_len=seq_len,
                n_samples=n_samples,
                min_total_ratio=min_total_ratio_to_use,  # Use adjusted ratio for COLA-Zero
                preview_dir=None,  # Set to path if you want preview
                enforce_no_pad=True,
                do_smoke_forward=False,
                seed=seed
            )

            # Replace examples with properly packed chunks (no padding!)
            examples = []
            for input_ids, attention_mask in zip(packed["input_ids"], packed["attention_masks"]):
                examples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                })

            # Update metadata with doctor report
            meta["calib_doctor"] = packed["report"]

            print(f"[RUNNER] ✅ Calibration doctor: PASS")
            print(f"[RUNNER]    - Pad ratio: {packed['report']['specials']['frac_pad']:.4f} (must be 0.0)")
            print(f"[RUNNER]    - EOS max run: {packed['report']['specials']['max_run_eos']}")
            print(f"[RUNNER]    - Diversity: {packed['report']['diversity']:.3f}")
            print(f"[RUNNER]    - Total tokens: {packed['report']['total_tokens']:,}")

            # Improved metrics (source/used documents)
            meta_report = packed['report']['meta']
            unique_docs = meta_report.get('unique_docs_used', 0)
            source_docs = meta_report.get('source_text_count', len(selected_texts))
            frac_multi = meta_report.get('frac_multi_doc_windows', 0.0)

            print(f"[RUNNER]    - Documents: {unique_docs}/{source_docs} used ({unique_docs/max(1, source_docs)*100:.1f}%)")
            print(f"[RUNNER]    - Multi-doc windows: {frac_multi*100:.1f}%")

            # Preview first 3 chunks (decoded, first 100 chars each)
            print(f"[RUNNER] Preview of first 3 calibration chunks:")
            for i in range(min(3, len(packed['input_ids']))):
                chunk_text = packed['tokenizer'].decode(packed['input_ids'][i], skip_special_tokens=True)
                preview = chunk_text[:100].replace('\n', ' ')
                print(f"[RUNNER]    Chunk {i+1}: {preview}...")

        except Exception as e:
            print(f"[RUNNER] ⚠️  Calibration doctor FAILED: {e}")
            raise RuntimeError(f"Calibration doctor failed - cannot proceed (padding-leakage risk): {e}")
            # FAIL HARD - don't continue with potentially broken calibration

    print(f"[RUNNER] Calibration data ready: {len(examples)} samples, {selection_time:.2f}s")

    return examples, selection_time, meta


def quantize_and_save(
    model_name: str,
    examples: List[Dict],
    output_dir: str,
    bits: int,
    group_size: int
) -> float:
    """
    Quantize model and save to disk.

    Args:
        model_name: HuggingFace model ID
        examples: Calibration examples
        output_dir: Save directory
        bits: Quantization bits
        group_size: Group size

    Returns:
        quantization_time_sec
    """
    print(f"[RUNNER] Quantizing model: {model_name}")
    print(f"[RUNNER] Output directory: {output_dir}")

    # Configure quantization
    quantize_config = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        damp_percent=0.01
    )

    # Load model
    print(f"[RUNNER] Loading model for quantization...")
    model = GPTQModel.load(
        model_name,
        quantize_config=quantize_config
    )

    # Quantize
    print(f"[RUNNER] Starting quantization (this may take 30-60 minutes)...")
    quant_start = time.time()

    model.quantize(
        calibration_dataset=examples,
        batch_size=1
    )

    quant_time = time.time() - quant_start

    # Save
    print(f"[RUNNER] Saving quantized model...")
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)

    print(f"[RUNNER] Quantization complete: {quant_time:.2f}s")

    return quant_time


def eval_perplexity(model_path: str, corpus: str = 'wikitext2', device: str = 'cuda', seqlen: int = 2048) -> Tuple[float, int]:
    """
    Evaluate perplexity on specified corpus.

    Args:
        model_path: Path to quantized model
        corpus: Corpus to evaluate on ('wikitext2' or 'c4')
        device: 'cuda' or 'cpu'
        seqlen: Sequence length for evaluation

    Returns:
        Tuple (perplexity, evaluated_token_count)
    """
    print(f"[RUNNER] Evaluating perplexity on {corpus}: {model_path}")

    # Load model
    model = GPTQModel.load(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load test data based on corpus
    if corpus == 'wikitext2':
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    elif corpus == 'c4':
        # Load C4 validation split (streaming, take first ~10MB of text)
        testdata = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
        texts = []
        total_chars = 0
        max_chars = 10_000_000  # ~10MB of text
        for example in testdata:
            texts.append(example['text'])
            total_chars += len(example['text'])
            if total_chars >= max_chars:
                break
        testenc = tokenizer("\n\n".join(texts), return_tensors="pt")
    else:
        raise ValueError(f"Unknown corpus: {corpus}. Must be 'wikitext2' or 'c4'")

    # Get model's device
    model_device = next(model.parameters()).device
    testenc = testenc.input_ids.to(model_device)

    # Calculate perplexity
    nsamples = testenc.numel() // seqlen
    model.eval()
    nlls = []

    with torch.no_grad():
        for i in range(nsamples):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)]
            attn = torch.ones_like(batch, dtype=torch.long)
            outputs = model(batch, attention_mask=attn)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            neg_log_likelihood = loss.float() * (seqlen - 1)
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / ((seqlen - 1) * nsamples))
    ppl_value = ppl.item()
    eval_tokens = int(nsamples * seqlen)

    print(f"[RUNNER] Perplexity ({corpus}): {ppl_value:.2f}")
    print(f"[RUNNER] Evaluated tokens: {eval_tokens}")

    return ppl_value, eval_tokens


def eval_downstream(model_path: str, tasks: List[str] = None, batch_size: int = 8) -> Dict[str, float]:
    """
    Evaluate on downstream zero-shot tasks using lm-eval.

    Args:
        model_path: Path to quantized model
        tasks: List of task names to evaluate (default: ["arc_easy", "hellaswag", "piqa", "winogrande"])
        batch_size: Batch size for evaluation (default: 8)

    Returns:
        Dict with task scores (accuracy) or None values if evaluation fails
    """
    # Default tasks if not specified
    if tasks is None:
        tasks = ["arc_easy", "hellaswag", "piqa", "winogrande"]

    # Create default dict with None values
    default_scores = {task: None for task in tasks}
    default_scores['average'] = None

    if not HAS_LM_EVAL:
        print(f"[RUNNER] Downstream evaluation skipped (lm-eval not installed)")
        print(f"[RUNNER] Install with: pip install lm-eval[api]")
        return default_scores

    if not HAS_OPTIMUM:
        print(f"[RUNNER] Downstream evaluation skipped (optimum not installed)")
        print(f"[RUNNER] optimum is required for lm-eval to load GPTQ models")
        print(f"[RUNNER] Install with: pip install optimum auto-gptq")
        return default_scores

    # Pre-cache math_qa dataset if needed (requires trust_remote_code=True)
    if 'mathqa' in tasks:
        try:
            print(f"[RUNNER] Pre-caching math_qa dataset with trust_remote_code=True...")
            _ = load_dataset('math_qa', trust_remote_code=True)
            print(f"[RUNNER] math_qa dataset cached successfully")
        except Exception as e:
            print(f"[RUNNER] Warning: Failed to pre-cache math_qa dataset: {e}")
            print(f"[RUNNER] Continuing with evaluation anyway...")

    print(f"[RUNNER] Starting downstream evaluation on {len(tasks)} tasks...")
    print(f"[RUNNER] Tasks: {tasks}")

    try:
        # Run lm-eval
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path},dtype=float16,trust_remote_code=True",
            tasks=tasks,
            num_fewshot=0,  # Zero-shot
            batch_size=batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Extract scores
        scores = {}
        for task in tasks:
            task_results = results['results'][task]

            # Different tasks use different metric names
            # Most use 'acc_norm' (normalized accuracy), fallback to 'acc'
            score_value = None
            if 'acc_norm' in task_results:
                score_value = task_results['acc_norm']
            elif 'acc' in task_results:
                score_value = task_results['acc']
            else:
                # Fallback: find first numeric value
                for key, val in task_results.items():
                    if isinstance(val, (int, float)):
                        score_value = val
                        break

            # Ensure score is numeric
            if score_value is not None:
                try:
                    scores[task] = float(score_value)
                except (ValueError, TypeError):
                    print(f"[RUNNER] Warning: Could not convert score for {task}: {score_value}")
                    scores[task] = None
            else:
                scores[task] = None

        # Calculate average (only from non-None numeric scores)
        numeric_scores = [s for s in scores.values() if s is not None and isinstance(s, (int, float))]
        if numeric_scores:
            scores['average'] = sum(numeric_scores) / len(numeric_scores)
        else:
            scores['average'] = None

        print(f"[RUNNER] Downstream evaluation complete:")
        for task, score in scores.items():
            if score is not None and isinstance(score, (int, float)):
                print(f"  {task}: {score:.4f}")
            else:
                print(f"  {task}: None")

        return scores

    except Exception as e:
        print(f"[RUNNER] Error during downstream evaluation: {e}")
        import traceback
        traceback.print_exc()
        return default_scores


def run_single_experiment(
    model_name: str,
    method: str,
    seed: int,
    config: Dict,
    random_baseline_time: Optional[float] = None
) -> Dict:
    """
    Run a single (model, method, seed) experiment.

    Args:
        model_name: HuggingFace model ID
        method: Calibration method
        seed: Random seed
        config: Experiment configuration

    Returns:
        Results dictionary
    """
    print(f"\n{'='*80}")
    print(f"[RUNNER] EXPERIMENT: model={model_name}, method={method}, seed={seed}")
    print(f"{'='*80}\n")

    # Seed everything for reproducibility
    set_all_seeds(seed)

    # Prepare paths
    model_clean = model_name.replace("/", "-")
    output_dir = f"./results/quantized_models/{model_clean}__{method}__seed{seed}"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load calibration dataset from specified source
    calibration_source = config.get('calibration_source', 'wikitext')
    print(f"[RUNNER] Loading calibration data from: {calibration_source}")
    documents = get_calibration_source(calibration_source, split='train')

    # Build calibration examples
    examples, selection_time, calib_meta = build_calibration_examples(
        method=method,
        tokenizer=tokenizer,
        documents=documents,
        n_samples=config['n_calibration_samples'],
        seq_len=config['seq_len'],
        seed=seed,
        config=config,
        calibration_source=calibration_source,
        tokenizer_name=model_name
    )

    # Quantize model
    quant_time = quantize_and_save(
        model_name=model_name,
        examples=examples,
        output_dir=output_dir,
        bits=config['quant_bits'],
        group_size=config['group_size']
    )

    total_time = selection_time + quant_time

    # Evaluate perplexity on specified corpora (default: just WikiText-2)
    ppl_corpora = config.get('ppl_corpora', ['wikitext2'])
    ppl_results = {}
    eval_tokens_dict = {}

    for corpus in ppl_corpora:
        ppl_value, tokens = eval_perplexity(output_dir, corpus=corpus, seqlen=config['seq_len'])
        ppl_results[corpus] = ppl_value
        eval_tokens_dict[corpus] = tokens

    # Primary PPL is always wikitext2 (for backwards compatibility)
    ppl = ppl_results.get('wikitext2', ppl_results[ppl_corpora[0]])
    eval_tokens = eval_tokens_dict.get('wikitext2', eval_tokens_dict[ppl_corpora[0]])

    # Evaluate downstream tasks for all seeds if enabled
    if config.get('do_downstream', False):
        downstream_tasks = config.get('downstream_tasks', None)  # None = use default
        downstream = eval_downstream(
            output_dir,
            tasks=downstream_tasks,
            batch_size=config.get('eval_batch_size', 8)
        )
    else:
        downstream = {
            "arc_easy": None,
            "hellaswag": None,
            "piqa": None,
            "winogrande": None,
            "average": None
        }

    # Build results dictionary
    calib_meta = calib_meta or {}
    calib_meta.update({
        "n_calibration_samples": len(examples),
        "seed": seed,
        "selection_time_sec": selection_time,
        "method": method
    })

    if method == "random":
        selection_overhead = 0.0
    elif random_baseline_time is not None:
        selection_overhead = selection_time - random_baseline_time
    else:
        selection_overhead = None

    results = {
        "model": model_name,
        "method": method,
        "seed": seed,
        "calibration_source": calibration_source,
        "n_calibration_samples": config['n_calibration_samples'],
        "seq_len": config['seq_len'],
        "quant_bits": config['quant_bits'],
        "group_size": config['group_size'],
        "selection_time_sec": selection_time,
        "quant_time_sec": quant_time,
        "total_time_sec": total_time,
        "perplexity": ppl,  # Primary PPL (WikiText-2 for backwards compatibility)
        "downstream": downstream,
        "calib_split": "train",
        "eval_split": "test",
        "calib_meta": calib_meta,
        "eval_tokens": int(eval_tokens),
        "selection_overhead_vs_random_sec": selection_overhead,
        "env": collect_env_info()
    }

    # Add individual PPL results for each corpus
    for corpus, ppl_value in ppl_results.items():
        if corpus == 'wikitext2':
            continue  # Already stored as 'perplexity'
        # Store as ppl_c4, ppl_mathqa, etc.
        results[f"ppl_{corpus}"] = ppl_value

    # Save results
    results_dir = "./results/metrics/raw_runs"
    os.makedirs(results_dir, exist_ok=True)

    # Include calibration source in filename for cross-corpus experiments
    if calibration_source != 'wikitext':
        results_path = f"{results_dir}/{model_clean}__{method}__{calibration_source}__seed{seed}.json"
    else:
        results_path = f"{results_dir}/{model_clean}__{method}__seed{seed}.json"

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[RUNNER] Results saved to: {results_path}")
    print(f"[RUNNER] PPL@WikiText2={ppl:.2f}", end="")
    if 'c4' in ppl_results:
        print(f", PPL@C4={ppl_results['c4']:.2f}", end="")
    print(f", Selection={selection_time:.1f}s, Quant={quant_time:.1f}s")

    return results


def run_experiment_suite(config: Dict):
    """
    Run full experimental suite across all models, methods, and seeds.

    Args:
        config: Configuration dictionary with keys:
            - models: List of model names
            - methods: List of methods
            - seeds: List of random seeds
            - n_calibration_samples: Number of calibration samples
            - seq_len: Sequence length
            - quant_bits: Quantization bits
            - group_size: Group size
            - do_downstream: Whether to run downstream evaluation
    """
    print(f"\n{'='*80}")
    print(f"[RUNNER] STARTING EXPERIMENTAL SUITE")
    print(f"{'='*80}")
    print(f"Models: {config['models']}")
    print(f"Methods: {config['methods']}")
    print(f"Seeds: {config['seeds']}")
    print(f"Calibration samples: {config['n_calibration_samples']}")
    print(f"Downstream eval: {config.get('do_downstream', False)}")
    print(f"{'='*80}\n")

    total_experiments = len(config['models']) * len(config['methods']) * len(config['seeds'])

    completed = 0
    random_baseline_times: Dict[Tuple[str, int], float] = {}

    for model_name in config['models']:
        for seed in config['seeds']:
            for method in config['methods']:
                completed += 1

                print(f"\n[RUNNER] Progress: {completed}/{total_experiments} experiments")

                try:
                    baseline_key = (model_name, seed)
                    result = run_single_experiment(
                        model_name=model_name,
                        method=method,
                        seed=seed,
                        config=config,
                        random_baseline_time=random_baseline_times.get(baseline_key)
                    )
                    if method == 'random':
                        random_baseline_times[baseline_key] = result["selection_time_sec"]
                except Exception as e:
                    print(f"\n[RUNNER] ERROR in experiment:")
                    print(f"  Model: {model_name}")
                    print(f"  Method: {method}")
                    print(f"  Seed: {seed}")
                    print(f"  Error: {str(e)}")
                    print(f"[RUNNER] Continuing with next experiment...\n")
                    continue

    print(f"\n{'='*80}")
    print(f"[RUNNER] EXPERIMENTAL SUITE COMPLETE")
    print(f"[RUNNER] Completed: {completed}/{total_experiments} experiments")
    print(f"[RUNNER] Results directory: ./results/metrics/raw_runs/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Experimental configuration
    config = {
        "models": [
            "meta-llama/Meta-Llama-3-8B-Instruct"
        ],
        "methods": ["cola_zero_vanille", "cola_zero", "random"],
        "seeds": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],  # 10 seeds: 1-10
        "n_calibration_samples": 128,
        "seq_len": 2048,
        "quant_bits": 4,
        "group_size": 128,
        "do_downstream": True,  # Enable downstream eval (HellaSwag, WinoGrande, PIQA, ARC-easy)
        "eval_batch_size": 8  # Batch size for downstream evaluation
    }

    # Run full suite
    run_experiment_suite(config)
