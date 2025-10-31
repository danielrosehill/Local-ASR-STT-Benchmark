# STT Benchmarking Setup

## Directory Structure

```
~/models/
├── stt/
│   ├── openai-whisper/       # OpenAI Whisper models
│   ├── faster-whisper/        # faster-whisper (CTranslate2) models
│   ├── coqui/                 # Coqui TTS/STT models
│   ├── vosk/                  # Vosk models
│   └── other/                 # Other STT models
└── tts/                       # Text-to-speech models
```

## Environment Variables

Added to `~/.bashrc`:

```bash
# STT/TTS model paths
export HF_HOME="$HOME/models/stt/faster-whisper"
export TRANSFORMERS_CACHE="$HOME/models/stt/faster-whisper"
export XDG_CACHE_HOME="$HOME/models"
```

These ensure that:
- **HuggingFace models** download to `~/models/stt/faster-whisper/`
- **Transformers cache** uses the same location
- Models are stored in a centralized, organized location

## Conda Environment

**Environment name:** `stt-benchmark`

**Activation:**
```bash
conda activate stt-benchmark
```

**Installed packages:**
- `faster-whisper` - Optimized Whisper inference using CTranslate2
- `openai-whisper` - Original OpenAI Whisper
- `jiwer` - WER (Word Error Rate) calculation
- `soundfile`, `librosa`, `pydub` - Audio processing
- `pandas` - Data analysis and results output
- `torch` - PyTorch for Whisper inference
- `numpy`, `tqdm` - Utilities

## Reinstalling Dependencies

```bash
conda activate stt-benchmark
pip install -r requirements.txt
```

## Available STT Engines

### 1. faster-whisper (CTranslate2)
- Faster than OpenAI Whisper
- Lower memory usage
- Quantized models available
- Models: tiny, base, small, medium, large-v3, large-v3-turbo

### 2. OpenAI Whisper (Original)
- Original implementation
- Models: tiny, base, small, medium, large-v3

### 3. Coqui STT (Optional)
- TTS also available
- Not installed by default (conflicts with pandas>=2.0)
- Uncomment in `requirements.txt` if needed

## Model Storage

Models will be downloaded to `~/models/stt/` subdirectories on first use. The benchmarking script will automatically download the specified models if they don't exist.

## Usage

```bash
conda activate stt-benchmark
python scripts/benchmark_stt.py --audio samples/test_audio.wav
```

See `readme.md` for full usage instructions.
