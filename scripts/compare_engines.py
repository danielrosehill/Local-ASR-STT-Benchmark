#!/usr/bin/env python3
"""
Compare Whisper Engine Implementations

Tests the same model size across different Whisper implementations:
- faster-whisper (CTranslate2 optimization)
- openai-whisper (original implementation)

Answers: How much do the optimized versions impact WER and speed?
"""

import argparse
import sys
from pathlib import Path
from benchmark_stt import STTBenchmark
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Compare faster-whisper vs openai-whisper for same model size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Compare base model across both engines
  python compare_engines.py --audio audio/test.wav --reference text/test.txt --model base

  # Compare multiple sizes
  python compare_engines.py --audio audio/test.wav --reference text/test.txt --models tiny base small
        """
    )

    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--reference', type=str, required=True, help='Reference transcription')
    parser.add_argument(
        '--models',
        nargs='+',
        default=['base'],
        help='Model sizes to compare (default: base)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='engine_comparison.csv',
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

    all_results = []

    print("="*60)
    print("WHISPER ENGINE COMPARISON")
    print("="*60)
    print(f"Models: {', '.join(args.models)}")
    print(f"Audio: {Path(args.audio).name}\n")

    # Test faster-whisper
    print("\n--- Testing faster-whisper ---")
    fw_benchmark = STTBenchmark(engine="faster-whisper")
    fw_benchmark.benchmark_audio(
        audio_path=args.audio,
        models=args.models,
        reference_text=reference_text
    )
    all_results.extend(fw_benchmark.results)

    # Test openai-whisper
    print("\n--- Testing openai-whisper ---")
    ow_benchmark = STTBenchmark(engine="openai-whisper")
    ow_benchmark.benchmark_audio(
        audio_path=args.audio,
        models=args.models,
        reference_text=reference_text
    )
    all_results.extend(ow_benchmark.results)

    # Save combined results
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\n\nResults saved to: {args.output}")

    # Comparison analysis
    print("\n" + "="*60)
    print("ENGINE COMPARISON ANALYSIS")
    print("="*60)

    for model in args.models:
        print(f"\n{model.upper()}:")

        fw_row = df[(df['engine'] == 'faster-whisper') & (df['model_size'] == model)]
        ow_row = df[(df['engine'] == 'openai-whisper') & (df['model_size'] == model)]

        if not fw_row.empty and not ow_row.empty:
            fw_time = fw_row['inference_time_sec'].values[0]
            ow_time = ow_row['inference_time_sec'].values[0]
            fw_wer = fw_row['wer'].values[0]
            ow_wer = ow_row['wer'].values[0]

            speedup = ow_time / fw_time
            wer_diff = fw_wer - ow_wer

            print(f"  faster-whisper: {fw_time:.2f}s, WER: {fw_wer}%")
            print(f"  openai-whisper: {ow_time:.2f}s, WER: {ow_wer}%")
            print(f"  → Speedup: {speedup:.2f}x")
            print(f"  → WER diff: {wer_diff:+.2f}% {'(faster-whisper better)' if wer_diff < 0 else '(openai better)' if wer_diff > 0 else '(identical)'}")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
