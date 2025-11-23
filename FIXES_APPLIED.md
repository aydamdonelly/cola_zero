# Fixes Applied - Padding-Leakage Prevention & Cross-Corpus

## A) COLA-Zero: Automatic Token Budget Top-Up ✅

**Problem:** COLA-Zero selected 384 documents with ~187k tokens, but needed ≥288k tokens (128 × 2048 × 1.10).

**Fix (Variant 1 - Automatic Top-Up):**
- File: `experiments/runner.py` (lines 279-323)
- Logic:
  1. Extract selected documents from metadata (`doc_indices`)
  2. Count tokens in selected pool
  3. If insufficient, add documents from remaining pool (sorted by length, descending)
  4. Continue until token budget ≥288,358 reached
  5. Pass extended pool to calibration doctor

**Expected Result:** COLA-Zero should now PASS doctor validation and generate proper calibration data.

---

## B) Random: Improved Metrics & Better Tracking ✅

**Problem:**
1. `unique_doc_fraction: 0.00` (metric bug)
2. Random PPL = 26.48 (expected ~16)

**Fix 1 - Accurate Document Tracking:**
- File: `calibration_doctor.py` (lines 48-102)
- Changes:
  - Added `doc_ids_per_token` array to track which document each token came from
  - Improved `unique_doc_fraction` calculation:
    - Old: Only checked START token of each window (heuristic)
    - New: Checks ALL tokens in each window, counts unique docs
  - New metrics:
    - `unique_docs_used`: Absolute count of documents used
    - `source_text_count`: Total documents in pool
    - `frac_multi_doc_windows`: % of windows spanning multiple documents

**Fix 2 - Better Logging:**
- File: `experiments/runner.py` (lines 355-362)
- Output now shows:
  ```
  [RUNNER]    - Documents: 256/28501 used (0.9%)
  [RUNNER]    - Multi-doc windows: 45.3%
  ```

**Expected Result:**
- `unique_doc_fraction` will no longer be 0.00
- Better visibility into which documents are actually used in calibration
- Random PPL issue needs further investigation (see Next Steps below)

---

## Changes Summary

### `experiments/runner.py`
1. **Lines 273-323**: COLA-Zero token budget top-up logic
   - Loads tokenizer
   - Counts tokens in selected pool
   - Adds documents from remaining pool if needed
   - Logs before/after token counts

2. **Lines 355-362**: Improved metrics logging
   - Shows `unique_docs_used / source_text_count`
   - Shows `frac_multi_doc_windows`

### `calibration_doctor.py`
1. **Lines 53-102**: Accurate document tracking
   - Added `doc_ids_per_token` list
   - Track which documents contribute to each window
   - Calculate exact `unique_doc_fraction`, `unique_docs_used`, `frac_multi_doc_windows`

---

## Testing Instructions

### Quick Test (4-5h):
```bash
cd /Users/adamkahirov/Desktop/code/cola_zero
python test_all_fixes.py
```

### Expected Output:

**COLA-Zero:**
```
[RUNNER]    Initial pool: 384 documents
[RUNNER]    Token budget: 187854 / 288358 (65.1%)
[RUNNER]    ⚠️  Token budget too small, adding documents from remaining pool...
[RUNNER]    ✅ Added 234 documents (new total: 618 docs, 301245 tokens)
[RUNNER]    Final pool for repacking: 618 documents
[RUNNER] ✅ Calibration doctor: PASS
[RUNNER]    - Pad ratio: 0.0000 (must be 0.0)
[RUNNER]    - Documents: 145/618 used (23.5%)
[RUNNER] Perplexity (wikitext2): ~11-13  ← Expected after fix!
```

**Random:**
```
[RUNNER]    Initial pool: 28501 documents
[RUNNER]    Final pool for repacking: 28501 documents
[RUNNER] ✅ Calibration doctor: PASS
[RUNNER]    - Pad ratio: 0.0000 (must be 0.0)
[RUNNER]    - Documents: 256/28501 used (0.9%)  ← No longer 0.00!
[RUNNER]    - Multi-doc windows: 45.3%
[RUNNER] Perplexity (wikitext2): ???  ← Still investigating
```

---

## Next Steps for Random PPL Issue

If Random still shows high PPL (~26 instead of ~16), investigate:

### 1. **A/B Test Without Doctor** (Quick check)
Temporarily bypass doctor to isolate the issue:
```python
# In runner.py, add before line 253:
if method == "random" and os.environ.get('SKIP_DOCTOR_DEBUG'):
    print("[DEBUG] Skipping doctor for Random (A/B test)")
    # skip doctor block
```

Run:
```bash
SKIP_DOCTOR_DEBUG=1 python test_all_fixes.py
```

If PPL drops to ~16 → Issue is in doctor repacking logic for Random
If PPL stays high → Issue is elsewhere (data distribution, tokenizer, etc.)

### 2. **Check Doctor vs Random Sampling Difference**
- Random baseline: Concatenates all texts, samples RANDOM positions
- Doctor: Concatenates texts, takes SEQUENTIAL windows from start
- This changes the distribution of selected text!

Possible fix: Make doctor respect Random's sampling strategy (sample random windows instead of sequential)

### 3. **Increase n_samples** (Recommended)
```python
# In test_all_fixes.py, line 47:
"n_calibration_samples": 192,  # or 256
```

Often improves PPL by 0.3-0.6 points.

### 4. **Check Tokenizer Consistency**
Ensure doctor and eval use same tokenizer:
```bash
# Check logs for:
[RUNNER] Loading model for quantization...
INFO  Tokenicer: Auto fixed pad_token_id=...
```

Both should use same `pad_token_id`.

### 5. **Kernel Consistency**
Current setup:
- Quantization: `TritonV2QuantLinear`
- Evaluation: `MarlinQuantLinear`

Test with same kernel for both:
```python
# In GPTQModel quantize call:
use_triton=False  # Force Marlin for both
```

---

## What to Report Back

After running `test_all_fixes.py`, send:

1. **COLA-Zero status:**
   - ✅ PASS or ❌ FAIL?
   - Token budget before/after top-up
   - Final PPL (WikiText-2, C4)

2. **Random status:**
   - Final PPL (WikiText-2, C4)
   - `unique_docs_used / source_text_count` ratio
   - `frac_multi_doc_windows` percentage

3. **Logs:**
   - Full output of COLA-Zero experiment (from "Applying calibration doctor" to "Perplexity")
   - Full output of Random experiment

---

## If All Tests PASS

Run full cross-corpus experiment:
```bash
python run_cross_corpus.py  # ~60-80 hours
```

Then analyze:
```bash
python analyse_cross_corpus.py results/metrics/raw_runs
```

---

**Status:** All fixes implemented and ready for testing! 🚀
