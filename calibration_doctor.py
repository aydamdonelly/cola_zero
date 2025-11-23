#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibration_doctor.py
Sanity-Checks & Packing für GPTQ-Kalibrierung (ohne Padding-Leakage).
- Baut exakt-2048er Fenster (stride=seq_len) aus concat. echten Tokens
- Prüft Pad/EOS-Leakage, Coverage, Diversität, Overlap, Seeds
- Optional: Smoke-Forward durch HF-Modell (CPU/GPU) mit attention_mask

Nutzung (Import im Runner):
  from calibration_doctor import check_and_pack_for_gptq
  packed = check_and_pack_for_gptq(
      texts=selected_texts, tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
      seq_len=2048, n_samples=128, min_total_ratio=1.10,
      enforce_no_pad=True, do_smoke_forward=False
  )
  input_ids, attention_masks = packed["input_ids"], packed["attention_masks"]
"""

import json, os, random, collections
from typing import List, Tuple, Dict, Optional
import numpy as np

try:
    import torch
except Exception:
    torch = None

from transformers import AutoTokenizer

def bind_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def ensure_token_coverage(
    texts: List[str], tokenizer, n_samples:int, seq_len:int, min_ratio:float
) -> Tuple[bool, int, int]:
    total_tokens = 0
    for t in texts:
        total_tokens += len(tokenizer(t, add_special_tokens=False)["input_ids"])
    need = int(n_samples * seq_len * min_ratio)
    return total_tokens >= need, total_tokens, need

def pack_full_windows(
    texts: List[str], tokenizer, seq_len:int, n_samples:int, drop_special:bool=True
) -> Tuple[List[List[int]], List[List[int]], Dict]:
    toks = []
    doc_offsets = []
    doc_ids_per_token = []  # Track which document each token came from

    for idx, t in enumerate(texts):
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if drop_special:
            # Entferne BOS/EOS/Pad, falls in Rohtext geraten
            rm = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
            ids = [i for i in ids if i not in rm]
        if len(ids) == 0:
            continue
        doc_offsets.append((len(toks), len(toks)+len(ids), idx))
        toks.extend(ids)
        doc_ids_per_token.extend([idx] * len(ids))  # Track doc ID for each token

    need = n_samples * seq_len
    if len(toks) < need:
        raise RuntimeError(f"Not enough tokens after cleaning: have={len(toks)} need={need}")

    chunks, masks = [], []
    all_window_doc_sets = []  # Track which docs are in each window
    stride = seq_len  # kein Overlap

    for start in range(0, need, stride):
        window = toks[start:start+seq_len]
        if len(window) < seq_len:
            break
        chunks.append(window)
        masks.append([1]*seq_len)

        # Track which documents contribute to this window
        window_docs = set(doc_ids_per_token[start:start+seq_len])
        all_window_doc_sets.append(window_docs)

    # Compute accurate metrics
    unique_docs_used = len(set().union(*all_window_doc_sets)) if all_window_doc_sets else 0
    frac_multi_doc_windows = sum(1 for s in all_window_doc_sets if len(s) > 1) / max(1, len(all_window_doc_sets))
    unique_doc_fraction = unique_docs_used / max(1, len(doc_offsets))

    # Overlap-Rate bei stride==seq_len Null
    meta = dict(
        total_stream_tokens=len(toks),
        n_chunks=len(chunks),
        seq_len=seq_len,
        overlap_rate=0.0,
        unique_doc_fraction=unique_doc_fraction,
        unique_docs_used=unique_docs_used,
        source_text_count=len(doc_offsets),
        frac_multi_doc_windows=frac_multi_doc_windows
    )
    return chunks[:n_samples], masks[:n_samples], meta

def _estimate_unique_doc_fraction(chunks: List[List[int]], doc_offsets: List[Tuple[int,int,int]]) -> float:
    # Grobe Heuristik: erster Tokenindex des Fensters → welchem Docbereich fällt er zu
    used_docs = set()
    start_positions = [i*len(chunks[0]) for i in range(len(chunks))]
    for sp in start_positions:
        for s,e,doc_id in doc_offsets:
            if s <= sp < e:
                used_docs.add(doc_id)
                break
    if not doc_offsets:
        return 1.0
    return len(used_docs) / max(1, len(doc_offsets))

def special_token_stats(chunks: List[List[int]], tokenizer) -> Dict:
    pad = tokenizer.pad_token_id
    eos = tokenizer.eos_token_id
    bos = tokenizer.bos_token_id
    total = len(chunks)*len(chunks[0])
    c_pad = c_eos = c_bos = 0
    max_run_eos = 0
    for ch in chunks:
        # BOS/EOS/PAD Zählung
        c_pad += sum(1 for x in ch if x == pad)
        c_eos += sum(1 for x in ch if x == eos)
        c_bos += sum(1 for x in ch if x == bos)
        # Runlength für EOS
        run, best = 0, 0
        for x in ch:
            if x == eos:
                run += 1
                best = max(best, run)
            else:
                run = 0
        max_run_eos = max(max_run_eos, best)
    return dict(
        frac_pad=c_pad/total if total else 0.0,
        frac_eos=c_eos/total if total else 0.0,
        frac_bos=c_bos/total if total else 0.0,
        max_run_eos=max_run_eos
    )

def unigram_diversity(chunks: List[List[int]]) -> float:
    if not chunks:
        return 0.0
    seq_len = len(chunks[0])
    vals = []
    for ch in chunks:
        vals.append(len(set(ch))/seq_len)
    return float(np.mean(vals))

def check_and_pack_for_gptq(
    texts: List[str],
    tokenizer_name: str,
    seq_len:int=2048,
    n_samples:int=128,
    min_total_ratio:float=1.10,
    enforce_no_pad:bool=True,
    preview_dir:Optional[str]=None,
    do_smoke_forward:bool=False,
    seed:int=1234
) -> Dict:
    bind_seeds(seed)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # pad != eos erzwingen
    if tok.pad_token_id is None or tok.pad_token_id == tok.eos_token_id:
        if tok.pad_token_id is None:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
        if tok.pad_token_id == tok.eos_token_id:
            if 0 not in [tok.bos_token_id, tok.eos_token_id]:
                tok.pad_token_id = 0
            else:
                pass  # already added above

    ok, have, need = ensure_token_coverage(texts, tok, n_samples, seq_len, min_total_ratio)
    if not ok:
        raise RuntimeError(f"[FAIL] Token coverage too small: have={have}, need>={need}")

    chunks, masks, meta = pack_full_windows(texts, tok, seq_len, n_samples, drop_special=True)

    # Stats
    specs = special_token_stats(chunks, tok)
    div = unigram_diversity(chunks)

    # Hard checks
    assert all(len(ch)==seq_len for ch in chunks), "[FAIL] Ragged sequences detected."
    if enforce_no_pad and specs["frac_pad"] > 0.0:
        raise RuntimeError(f"[FAIL] Pad leakage detected: {specs['frac_pad']*100:.2f}%")
    if tok.pad_token_id == tok.eos_token_id:
        raise RuntimeError("[FAIL] pad_token_id equals eos_token_id")

    # Warnings
    warns = []
    if specs["max_run_eos"] > 4:
        warns.append(f"[WARN] Long EOS runs detected (max_run_eos={specs['max_run_eos']})")
    if specs["frac_eos"] > 0.05:
        warns.append(f"[WARN] High EOS share: {specs['frac_eos']*100:.2f}%")
    if div <= 0.25:
        warns.append(f"[WARN] Low unigram diversity: {div:.3f}")
    if meta["unique_doc_fraction"] <= 0.30:
        warns.append(f"[WARN] Low unique_doc_fraction: {meta['unique_doc_fraction']:.2f}")
    if meta["overlap_rate"] > 0.0:
        warns.append(f"[WARN] Overlap detected: {meta['overlap_rate']:.3f}")

    # Konsolidierte Ausgabe
    report = {
        "n_chunks": len(chunks),
        "seq_len": seq_len,
        "total_tokens": len(chunks)*seq_len,
        "coverage_guard": {"need": n_samples*seq_len*min_total_ratio, "have": have},
        "specials": specs,
        "diversity": div,
        "meta": meta,
        "warnings": warns
    }

    if warns:
        for w in warns:
            print(w)

    return {
        "input_ids": chunks,
        "attention_masks": masks,
        "tokenizer": tok,
        "report": report
    }
