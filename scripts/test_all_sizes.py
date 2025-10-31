#!/usr/bin/env python3
"""
Test All Whisper Model Sizes

Specifically answers the question: How much does model size matter?
Tests all Whisper sizes from tiny → large-v3 and compares WER vs inference time.
"""

import argparse
import sys
from pathlib import Path
from benchmark_stt import STTBenchmark

# All available Whisper model sizes in order (smallest to largest)
ALL_WHISPER_SIZES = [
    'tiny',           # 39M params, ~75MB
    'tiny.en',        # English-only
    'base',           # 74M params, ~140MB
    'base.en',        # English-only
    'small',          # 244M params, ~460MB
    'small.en',       # English-only
    'medium',         # 769M params, ~1.5GB
    'medium.en',      # English-only
    'large-v2',       # 1550M params, ~3GB
    'large-v3',       # 1550M params, ~3GB (latest, best)
    'large-v3-turbo'  # Optimized version of large-v3
]

# For quick testing (reasonable coverage without downloading everything)
QUICK_SIZES = [
    'tiny',
    'base',
    'small',
    'medium',
    'large-v3-turbo'
]

# Multilingual vs English-only comparison
ENGLISH_COMPARISON = [
    'tiny',
    'tiny.en',
    'base',
    'base.en',
    'small',
    'small.en'
]


def main():
    parser = argparse.ArgumentParser(
        description="Test all Whisper model sizes to see how size impacts accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all sizes (downloads ~8GB of models)
  python test_all_sizes.py --audio audio/test.wav --reference text/test.txt --preset all

  # Quick test of representative sizes
  python test_all_sizes.py --audio audio/test.wav --reference text/test.txt --preset quick

  # Compare English-only vs multilingual
  python test_all_sizes.py --audio audio/test.wav --reference text/test.txt --preset english

  # Custom size selection
  python test_all_sizes.py --audio audio/test.wav --models tiny base small medium large-v3
        """
    )

    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to audio file'
    )

    parser.add_argument(
        '--reference',
        type=str,
        required=True,
        help='Reference transcription text or file path (required for WER)'
    )

    parser.add_argument(
        '--preset',
        choices=['all', 'quick', 'english'],
        help='Model size preset: all (all sizes), quick (5 representative), english (multilingual vs .en)'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        help='Custom list of model sizes to test (overrides --preset)'
    )

    parser.add_argument(
        '--engine',
        choices=['faster-whisper', 'openai-whisper'],
        default='faster-whisper',
        help='STT engine (default: faster-whisper, recommended for speed)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='size_comparison.csv',
        help='Output CSV file (default: size_comparison.csv)'
    )

    args = parser.parse_args()

    # Validate audio file
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    # Load reference text
    if Path(args.reference).exists():
        reference_text = Path(args.reference).read_text().strip()
    else:
        reference_text = args.reference

    # Determine which models to test
    if args.models:
        models_to_test = args.models
        print(f"Testing custom model list: {', '.join(models_to_test)}")
    elif args.preset == 'all':
        models_to_test = ALL_WHISPER_SIZES
        print(f"Testing ALL Whisper sizes ({len(models_to_test)} models)")
        print("⚠️  Warning: This will download ~8GB of models on first run!")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != 'y':
            sys.exit(0)
    elif args.preset == 'english':
        models_to_test = ENGLISH_COMPARISON
        print(f"Testing English-only vs multilingual comparison ({len(models_to_test)} models)")
    else:  # quick or default
        models_to_test = QUICK_SIZES
        print(f"Testing quick preset ({len(models_to_test)} representative sizes)")

    print(f"\nModels: {', '.join(models_to_test)}")
    print(f"Engine: {args.engine}")
    print(f"Reference: {len(reference_text)} characters\n")

    # Run benchmark
    benchmarker = STTBenchmark(engine=args.engine)
    benchmarker.benchmark_audio(
        audio_path=args.audio,
        models=models_to_test,
        reference_text=reference_text
    )

    # Save results
    benchmarker.save_results(args.output)

    # Print analysis
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    import pandas as pd
    df = pd.DataFrame(benchmarker.results)

    if 'wer' in df.columns and df['wer'].notna().any():
        # Best accuracy
        best_wer = df.loc[df['wer'].idxmin()]
        print(f"\n✓ Best Accuracy: {best_wer['model_size']}")
        print(f"  WER: {best_wer['wer']}%")
        print(f"  Time: {best_wer['inference_time_sec']}s")

        # Fastest
        fastest = df.loc[df['inference_time_sec'].idxmin()]
        print(f"\n✓ Fastest: {fastest['model_size']}")
        print(f"  Time: {fastest['inference_time_sec']}s")
        print(f"  WER: {fastest['wer']}%")

        # Best trade-off (low WER, reasonable speed)
        # Score: lower is better (WER is already percentage, time in seconds)
        df['score'] = df['wer'] + (df['inference_time_sec'] * 2)  # Weight time less
        best_tradeoff = df.loc[df['score'].idxmin()]
        print(f"\n✓ Best Trade-off: {best_tradeoff['model_size']}")
        print(f"  WER: {best_tradeoff['wer']}%")
        print(f"  Time: {best_tradeoff['inference_time_sec']}s")

        # Size impact analysis
        tiny_wer = df[df['model_size'] == 'tiny']['wer'].values[0] if 'tiny' in models_to_test else None
        if tiny_wer and 'large-v3' in models_to_test:
            large_wer = df[df['model_size'] == 'large-v3']['wer'].values[0]
            improvement = tiny_wer - large_wer
            print(f"\n✓ Size Impact:")
            print(f"  tiny → large-v3: {improvement:.1f}% WER improvement")
            print(f"  ({tiny_wer}% → {large_wer}%)")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
