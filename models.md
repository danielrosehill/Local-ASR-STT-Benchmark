# Available Models for Testing

## Whisper Model Sizes (Multilingual)

| Model | Parameters | Disk Size | RAM Usage | Relative Speed | Typical WER |
|-------|-----------|-----------|-----------|----------------|-------------|
| `tiny` | 39M | ~75MB | ~1GB | Fastest (1x) | ~15-20% |
| `base` | 74M | ~140MB | ~1GB | Very Fast (1.5x) | ~10-15% |
| `small` | 244M | ~460MB | ~2GB | Fast (3x) | ~7-10% |
| `medium` | 769M | ~1.5GB | ~5GB | Moderate (6x) | ~5-7% |
| `large-v2` | 1550M | ~3GB | ~10GB | Slow (10x) | ~4-6% |
| `large-v3` | 1550M | ~3GB | ~10GB | Slow (10x) | ~3-5% (best) |
| `large-v3-turbo` | 809M | ~1.6GB | ~6GB | Fast-ish (5x) | ~4-6% |

*Speeds are approximate relative to `tiny` on CPU*

## Whisper English-Only Models

English-only models are optimized for English and may perform slightly better than multilingual versions for English audio:

| Model | Description |
|-------|-------------|
| `tiny.en` | English-only tiny |
| `base.en` | English-only base |
| `small.en` | English-only small |
| `medium.en` | English-only medium |

**Note:** Large models don't have `.en` variants as the multilingual versions are sufficient.

## Whisper Engine Variants

### 1. faster-whisper (Recommended)
- **Implementation:** CTranslate2 optimization
- **Speed:** 4-8x faster than OpenAI Whisper
- **Memory:** 50-70% less RAM usage
- **Quality:** Same or very similar WER
- **Quantization:** Supports int8, int16, float16, float32
- **Downloads to:** `~/models/stt/faster-whisper/`

### 2. openai-whisper (Original)
- **Implementation:** Original OpenAI PyTorch
- **Speed:** Baseline (1x)
- **Memory:** Standard
- **Quality:** Reference implementation
- **Downloads to:** `~/.cache/whisper/`

### 3. distil-whisper (Future)
- **Implementation:** Distilled/compressed Whisper
- **Speed:** ~6x faster than OpenAI
- **Quality:** Slight accuracy drop (~2% higher WER)
- **Status:** Not yet integrated (requires transformers library)

## Testing Presets

### Quick Test (5 models, ~2.2GB)
```bash
python scripts/test_all_sizes.py --audio audio/test.wav --reference text/test.txt --preset quick
```
Tests: `tiny`, `base`, `small`, `medium`, `large-v3-turbo`

### Full Test (11 models, ~8GB)
```bash
python scripts/test_all_sizes.py --audio audio/test.wav --reference text/test.txt --preset all
```
Tests all sizes including English-only variants

### English Comparison (6 models, ~1GB)
```bash
python scripts/test_all_sizes.py --audio audio/test.wav --reference text/test.txt --preset english
```
Tests multilingual vs `.en` variants for tiny/base/small

## Model Selection Guide

### For Real-Time Transcription
- **Best:** `tiny` or `base` with faster-whisper
- **Acceptable:** `small` if you need better accuracy

### For Offline/Batch Processing
- **Best Quality:** `large-v3` or `large-v3-turbo`
- **Best Balance:** `medium` or `small`

### For English-Only Audio
- Try `.en` variants of tiny/base/small first
- Often 1-2% better WER than multilingual

### For Low-Resource Machines
- Stick with `tiny` or `base`
- Use faster-whisper with int8 quantization

### For Best Accuracy (No Speed Constraints)
- Use `large-v3` with faster-whisper
- Consider `large-v3-turbo` for 2x speedup with minimal quality loss

## Hardware Requirements

### Minimum (CPU-only)
- **RAM:** 2GB free
- **Models:** tiny, base
- **Inference:** 5-10x real-time

### Recommended (CPU-only)
- **RAM:** 4GB free
- **Models:** tiny, base, small
- **Inference:** 3-5x real-time

### High-End (CPU-only)
- **RAM:** 8GB+ free
- **Models:** All sizes
- **Inference:** 1-10x real-time depending on size

### With GPU (CUDA)
- **VRAM:** 2GB+ for tiny/base, 8GB+ for large
- **Speed:** 50-100x real-time for all models
- **Note:** Current setup is CPU-only; GPU support requires CUDA installation

## Adding Models in Future

### Other STT Engines to Consider
- **Vosk:** Offline, lightweight, many languages
- **Coqui STT:** Open-source, trainable
- **Wav2Vec2:** Facebook's transformer-based STT
- **NeMo:** NVIDIA's toolkit with Conformer models
- **SpeechBrain:** Toolkit with multiple models

These can be added by extending the `STTBenchmark` class with new transcription methods.

## Model Download Behavior

- **First run:** Downloads models (can take 5-30 minutes depending on size)
- **Subsequent runs:** Uses cached models (instant)
- **Storage location:** `~/models/stt/faster-whisper/` (organized, reusable)
- **Multiple projects:** All projects share the same model cache

## Updating Models

Whisper models are updated occasionally (e.g., large-v2 → large-v3). To get the latest:

```bash
# Delete cached model
rm -rf ~/models/stt/faster-whisper/models--Systran--faster-whisper-[model-name]

# Re-run benchmark (will download latest)
python scripts/benchmark_stt.py --audio test.wav --models [model-name]
```

## Questions This Helps Answer

1. **How much does size matter?** → Use `scripts/test_all_sizes.py --preset all`
2. **Faster-whisper vs OpenAI?** → Use `scripts/compare_engines.py`
3. **English-only vs multilingual?** → Use `scripts/test_all_sizes.py --preset english`
4. **Best model for my use case?** → Run quick test, analyze WER vs speed tradeoff
