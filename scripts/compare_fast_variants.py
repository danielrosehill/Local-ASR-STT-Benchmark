#!/usr/bin/env python3
"""
Compare Speed-Optimized Whisper Variants

Tests different Whisper implementations optimized for speed:
- faster-whisper (CTranslate2 optimization)
- openai-whisper (original)
- distil-whisper (distilled/compressed)

Answers: Which gives the best speed without sacrificing accuracy?
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import jiwer
from tqdm import tqdm


class FastVariantBenchmark:
    """Benchmark fast Whisper variants."""

    def __init__(self):
        self.results = []

    def transcribe_faster_whisper(self, audio_path: str, model_size: str) -> tuple:
        """Transcribe using faster-whisper."""
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        start_time = time.time()
        segments, info = model.transcribe(audio_path, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        inference_time = time.time() - start_time

        return transcription.strip(), inference_time

    def transcribe_openai_whisper(self, audio_path: str, model_size: str) -> tuple:
        """Transcribe using OpenAI Whisper."""
        import whisper

        model = whisper.load_model(model_size)
        start_time = time.time()
        result = model.transcribe(audio_path)
        inference_time = time.time() - start_time

        return result["text"].strip(), inference_time

    def transcribe_distil_whisper(self, audio_path: str) -> tuple:
        """Transcribe using distil-whisper (small.en equivalent)."""
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = "cpu"
        torch_dtype = torch.float32

        model_id = "distil-whisper/distil-small.en"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=30,
            return_timestamps=True
        )

        start_time = time.time()
        result = pipe(audio_path, return_timestamps=True)
        inference_time = time.time() - start_time

        # Extract text from chunks
        if isinstance(result, dict) and "chunks" in result:
            text = " ".join([chunk["text"] for chunk in result["chunks"]])
        else:
            text = result["text"] if isinstance(result, dict) else str(result)

        return text.strip(), inference_time

    def calculate_wer_cer(self, reference: str, hypothesis: str) -> dict:
        """Calculate WER and CER."""
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)
        return {
            'wer': round(wer * 100, 2),
            'cer': round(cer * 100, 2)
        }

    def benchmark(self, audio_path: str, reference_text: str, model_size: str = "base"):
        """Run benchmark on all variants."""

        variants = [
            ("faster-whisper", lambda: self.transcribe_faster_whisper(audio_path, model_size)),
            ("openai-whisper", lambda: self.transcribe_openai_whisper(audio_path, model_size)),
        ]

        # Only test distil-whisper if using base/small
        if model_size in ["base", "small"]:
            print("\n⚠️  Note: distil-whisper only available for small.en size")
            if model_size == "small" or model_size == "base":
                variants.append(("distil-whisper", lambda: self.transcribe_distil_whisper(audio_path)))

        print(f"\n{'='*60}")
        print("SPEED-OPTIMIZED WHISPER VARIANTS COMPARISON")
        print(f"{'='*60}")
        print(f"Model size: {model_size}")
        print(f"Variants: {len(variants)}")
        print(f"Audio: {Path(audio_path).name}\n")

        for variant_name, transcribe_func in tqdm(variants, desc="Variants"):
            try:
                print(f"\n--- Testing {variant_name} ---")
                transcription, inference_time = transcribe_func()

                metrics = self.calculate_wer_cer(reference_text, transcription)

                result = {
                    'variant': variant_name,
                    'model_size': model_size if variant_name != "distil-whisper" else "distil-small.en",
                    'inference_time_sec': round(inference_time, 2),
                    'wer': metrics['wer'],
                    'cer': metrics['cer'],
                    'transcription': transcription
                }

                self.results.append(result)

                print(f"  Time: {inference_time:.2f}s")
                print(f"  WER: {metrics['wer']}%")
                print(f"  CER: {metrics['cer']}%")
                print(f"  Text: {transcription[:80]}...")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

    def analyze_results(self):
        """Print comparison analysis."""
        df = pd.DataFrame(self.results)

        if df.empty:
            print("No results to analyze")
            return df

        print(f"\n{'='*60}")
        print("COMPARISON ANALYSIS")
        print(f"{'='*60}\n")

        # Baseline is faster-whisper
        baseline = df[df['variant'] == 'faster-whisper'].iloc[0]
        baseline_time = baseline['inference_time_sec']
        baseline_wer = baseline['wer']

        print(f"Baseline (faster-whisper):")
        print(f"  Time: {baseline_time:.2f}s")
        print(f"  WER: {baseline_wer}%\n")

        for _, row in df.iterrows():
            if row['variant'] == 'faster-whisper':
                continue

            speedup = baseline_time / row['inference_time_sec']
            wer_diff = row['wer'] - baseline_wer

            print(f"{row['variant']}:")
            print(f"  Time: {row['inference_time_sec']:.2f}s")
            print(f"  WER: {row['wer']}%")
            print(f"  → Speedup vs baseline: {speedup:.2f}x {'(FASTER)' if speedup > 1 else '(SLOWER)'}")
            print(f"  → WER change: {wer_diff:+.2f}% {'(WORSE)' if wer_diff > 0 else '(BETTER)' if wer_diff < 0 else '(SAME)'}")

            # Efficiency score
            if wer_diff <= 1.0:  # Acceptable accuracy drop
                print(f"  ✓ Good trade-off: {speedup:.1f}x faster with minimal accuracy loss")
            elif wer_diff > 1.0:
                print(f"  ⚠️  Accuracy trade-off: {abs(wer_diff):.1f}% worse WER for {speedup:.1f}x speed")
            print()

        print(f"{'='*60}\n")

        return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare speed-optimized Whisper variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python compare_fast_variants.py --audio audio/test.wav --reference text/test.txt --model base
        """
    )

    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--reference', type=str, required=True, help='Reference transcription')
    parser.add_argument(
        '--model',
        type=str,
        default='base',
        choices=['base', 'small'],
        help='Model size to test (default: base)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='fast_variants_comparison.csv',
        help='Output CSV file'
    )

    args = parser.parse_args()

    # Validate
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    # Load reference
    if Path(args.reference).exists():
        reference_text = Path(args.reference).read_text().strip()
    else:
        reference_text = args.reference

    # Run benchmark
    benchmarker = FastVariantBenchmark()
    benchmarker.benchmark(
        audio_path=args.audio,
        reference_text=reference_text,
        model_size=args.model
    )

    # Analyze and save
    df = benchmarker.analyze_results()

    if not df.empty:
        df.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
