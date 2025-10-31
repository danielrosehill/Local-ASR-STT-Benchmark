#!/usr/bin/env python3
"""
STT Benchmarking Script

Benchmarks multiple Speech-to-Text models against audio samples,
calculating Word Error Rate (WER) and inference time.
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
import jiwer

# Set model cache directories
os.environ['HF_HOME'] = str(Path.home() / 'models' / 'stt' / 'faster-whisper')
os.environ['TRANSFORMERS_CACHE'] = str(Path.home() / 'models' / 'stt' / 'faster-whisper')


class STTBenchmark:
    """Main benchmarking class for STT models."""

    def __init__(self, engine: str = "faster-whisper"):
        """
        Initialize benchmarker.

        Args:
            engine: STT engine to use ('faster-whisper' or 'openai-whisper')
        """
        self.engine = engine
        self.results = []

    def transcribe_faster_whisper(self, audio_path: str, model_size: str) -> tuple:
        """
        Transcribe using faster-whisper.

        Returns:
            (transcription_text, inference_time_seconds)
        """
        from faster_whisper import WhisperModel

        # Model will be downloaded to HF_HOME if not present
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        start_time = time.time()
        segments, info = model.transcribe(audio_path, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
        inference_time = time.time() - start_time

        return transcription.strip(), inference_time

    def transcribe_openai_whisper(self, audio_path: str, model_size: str) -> tuple:
        """
        Transcribe using OpenAI Whisper.

        Returns:
            (transcription_text, inference_time_seconds)
        """
        import whisper

        # Load model (will download to ~/.cache/whisper/ if not present)
        model = whisper.load_model(model_size)

        start_time = time.time()
        result = model.transcribe(audio_path)
        inference_time = time.time() - start_time

        return result["text"].strip(), inference_time

    def calculate_metrics(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate WER and CER.

        Args:
            reference: Ground truth transcription
            hypothesis: Model's transcription

        Returns:
            Dictionary with 'wer' and 'cer' keys
        """
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)

        return {
            'wer': round(wer * 100, 2),  # Convert to percentage
            'cer': round(cer * 100, 2)
        }

    def benchmark_audio(self,
                       audio_path: str,
                       models: List[str],
                       reference_text: Optional[str] = None) -> None:
        """
        Benchmark multiple models on a single audio file.

        Args:
            audio_path: Path to audio file
            models: List of model sizes to test (e.g., ['tiny', 'base', 'small'])
            reference_text: Reference transcription for WER calculation
        """
        audio_name = Path(audio_path).stem

        print(f"\nBenchmarking: {audio_name}")
        print(f"Engine: {self.engine}")
        print(f"Models: {', '.join(models)}\n")

        for model_size in tqdm(models, desc="Models"):
            try:
                # Transcribe
                if self.engine == "faster-whisper":
                    transcription, inference_time = self.transcribe_faster_whisper(
                        audio_path, model_size
                    )
                else:  # openai-whisper
                    transcription, inference_time = self.transcribe_openai_whisper(
                        audio_path, model_size
                    )

                # Calculate metrics if reference provided
                metrics = {}
                if reference_text:
                    metrics = self.calculate_metrics(reference_text, transcription)

                # Store results
                result = {
                    'audio_file': audio_name,
                    'engine': self.engine,
                    'model_size': model_size,
                    'inference_time_sec': round(inference_time, 2),
                    'transcription': transcription,
                    'wer': metrics.get('wer', None),
                    'cer': metrics.get('cer', None)
                }

                self.results.append(result)

                print(f"\n{model_size}:")
                print(f"  Time: {inference_time:.2f}s")
                if reference_text:
                    print(f"  WER: {metrics['wer']}%")
                    print(f"  CER: {metrics['cer']}%")
                print(f"  Text: {transcription[:80]}...")

            except Exception as e:
                print(f"\nError with {model_size}: {e}")
                continue

    def save_results(self, output_path: str) -> None:
        """Save results to CSV."""
        df = pd.DataFrame(self.results)

        # Sort by model size and WER
        if 'wer' in df.columns and df['wer'].notna().any():
            df = df.sort_values(['audio_file', 'model_size', 'wer'])

        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # Print summary
        if 'wer' in df.columns and df['wer'].notna().any():
            print("\n=== Summary (sorted by WER) ===")
            summary = df[['model_size', 'wer', 'cer', 'inference_time_sec']].copy()
            print(summary.to_string(index=False))


def load_reference_text(reference: str) -> str:
    """
    Load reference text from file or use as-is.

    Args:
        reference: File path or text string

    Returns:
        Reference text
    """
    if Path(reference).exists():
        return Path(reference).read_text().strip()
    return reference


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark STT models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark single audio file with multiple models
  python benchmark_stt.py --audio audio/test.wav --models tiny base small

  # With reference transcription for WER calculation
  python benchmark_stt.py --audio audio/test.wav --reference text/test.txt --models tiny base small

  # Use OpenAI Whisper instead of faster-whisper
  python benchmark_stt.py --audio audio/test.wav --engine openai-whisper --models base
        """
    )

    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to audio file'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=['tiny', 'base', 'small', 'medium'],
        help='Whisper model sizes to test (default: tiny base small medium)'
    )

    parser.add_argument(
        '--engine',
        choices=['faster-whisper', 'openai-whisper'],
        default='faster-whisper',
        help='STT engine to use (default: faster-whisper)'
    )

    parser.add_argument(
        '--reference',
        type=str,
        help='Reference transcription text or file path for WER calculation'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results.csv',
        help='Output CSV file (default: results.csv)'
    )

    args = parser.parse_args()

    # Validate audio file
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        return

    # Load reference text if provided
    reference_text = None
    if args.reference:
        reference_text = load_reference_text(args.reference)
        print(f"Reference text loaded: {len(reference_text)} characters\n")

    # Create benchmarker and run
    benchmarker = STTBenchmark(engine=args.engine)
    benchmarker.benchmark_audio(
        audio_path=args.audio,
        models=args.models,
        reference_text=reference_text
    )

    # Save results
    benchmarker.save_results(args.output)


if __name__ == '__main__':
    main()
