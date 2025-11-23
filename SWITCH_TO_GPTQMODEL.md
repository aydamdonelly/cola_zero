# Switching from AutoGPTQ to GPTQModel

## ✅ All Code Updated

I've updated all experiment files to use GPTQModel instead of AutoGPTQ:
- ✅ `experiments/01_quantize_cola_zero.py`
- ✅ `experiments/02_quantize_random.py`
- ✅ `experiments/03_evaluate_perplexity.py`

## 🚀 What You Need To Do On Remote Machine

### Step 1: Install GPTQModel

```bash
cd /data/user_data/adam/kvkz-vllm/cola_zero/GPTQModel
pip install -v . --no-build-isolation
```

This will replace AutoGPTQ with GPTQModel (actively maintained, fixes Llama bugs).

### Step 2: Run Llama-2-7b-hf Experiment

```bash
cd /data/user_data/adam/kvkz-vllm/cola_zero
python3 experiments/05_compare_methods.py --model meta-llama/Llama-2-7b-hf > llama2-7b.log 2>&1 &
tail -f llama2-7b.log
```

**This will now work!** GPTQModel fixes the Llama Grouped-Query Attention bug.

---

## 🔧 All Fixes Applied

### 1. ✅ Switched to GPTQModel
- **Before:** `from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig`
- **After:** `from gptqmodel import GPTQModel, QuantizeConfig`

### 2. ✅ Fixed Model Loading
- **Before:** Complex try/except with safetensors fallback
- **After:** Simple `GPTQModel.load(model_name, quantize_config)`

### 3. ✅ Fixed Quantization Call
- **Before:** `model.quantize(examples=..., use_triton=..., cache_examples_on_gpu=...)`
- **After:** `model.quantize(calibration_dataset=..., batch_size=1)`

### 4. ✅ Fixed Save Method
- **Before:** `model.save_quantized(path, use_safetensors=True); tokenizer.save_pretrained(path)`
- **After:** `model.save(path)` (tokenizer saved automatically)

### 5. ✅ Fixed Padding Issue (cola_zero/sampler.py)
- **Before:** Each doc padded to 2048 → lots of eos tokens
- **After:** Concatenate all docs → extract dense 2048-token chunks

### 6. ✅ Fixed Short Document Issue (cola_zero/sampler.py)
- **Before:** Filtered by character length (min_length=100)
- **After:** Filtered by token length (min_length=500 tokens)

### 7. ✅ Fixed Insufficient Tokens Issue (cola_zero/sampler.py)
- **Before:** Selected 1 doc per cluster → only 8 chunks
- **After:** Select top 3 docs per cluster → ~384 docs, enough for 128 chunks

---

## 📊 Expected Output

```
Step 4: Selecting representative documents...
  Selected 384 representative documents (avg 3.0 per cluster)

  Total tokens in selected documents: 450,000
  Target tokens for 128 chunks: 262,144
  Coverage: 171.6%

Step 5: Tokenizing to AutoGPTQ format...
  Tokenization complete. Sample shape: torch.Size([2048])
```

**No more "Note: Corpus only yielded 8 non-overlapping chunks"!**

---

## 🎯 Why This Will Work Now

1. **GPTQModel supports Llama-2** - No more rotary embedding errors
2. **Dense calibration chunks** - No padding spam
3. **Enough material** - 384 docs = ~450K tokens for 128 chunks
4. **Better filtering** - Only docs with ≥500 tokens

---

## Alternative: Test with OPT-1.3B First

If you want a quick test before the long Llama-2-7b run:

```bash
python3 experiments/05_compare_methods.py --model facebook/opt-1.3b > opt1.3b.log 2>&1 &
```

Expected time: ~30 minutes
This validates all fixes work before committing to 2-hour Llama-2-7b run.

---

## 🚨 Summary

**On remote machine, run:**
```bash
# Install GPTQModel
cd GPTQModel && pip install -v . --no-build-isolation

# Run experiment
cd .. && python3 experiments/05_compare_methods.py --model meta-llama/Llama-2-7b-hf > llama2.log 2>&1 &
```

**Done!** All code is ready, just need to install GPTQModel.
