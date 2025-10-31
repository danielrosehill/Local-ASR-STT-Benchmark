#!/usr/bin/env python3
"""
Aggregate and visualize STT benchmark results
Generates charts showing WER, inference time, and accuracy-speed tradeoffs
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR

def load_all_data():
    """Load all CSV result files"""
    data = {}

    # Size comparison
    size_file = RESULTS_DIR / "size_comparison.csv"
    if size_file.exists():
        data['size'] = pd.read_csv(size_file)
        print(f"Loaded size_comparison.csv: {len(data['size'])} rows")

    # Engine comparison
    engine_file = RESULTS_DIR / "engine_comparison_base.csv"
    if engine_file.exists():
        data['engine'] = pd.read_csv(engine_file)
        print(f"Loaded engine_comparison_base.csv: {len(data['engine'])} rows")

    # Fast variants comparison
    variants_file = RESULTS_DIR / "fast_variants_base.csv"
    if variants_file.exists():
        data['variants'] = pd.read_csv(variants_file)
        print(f"Loaded fast_variants_base.csv: {len(data['variants'])} rows")

    return data

def plot_wer_by_size(df):
    """Plot WER by model size"""
    plt.figure(figsize=(12, 6))

    # Order models by size - only use models that exist in data
    all_models = ['tiny', 'base', 'small', 'medium', 'large-v3-turbo', 'large-v3']
    model_order = [m for m in all_models if m in df['model_size'].values]
    df_sorted = df.set_index('model_size').loc[model_order].reset_index()

    bars = plt.bar(df_sorted['model_size'], df_sorted['wer'], color='steelblue', alpha=0.8)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)

    plt.xlabel('Model Size', fontsize=12, fontweight='bold')
    plt.ylabel('Word Error Rate (%)', fontsize=12, fontweight='bold')
    plt.title('Word Error Rate by Whisper Model Size', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_file = OUTPUT_DIR / "wer_by_size.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_speed_by_size(df):
    """Plot inference time by model size"""
    plt.figure(figsize=(12, 6))

    # Order models by size - only use models that exist in data
    all_models = ['tiny', 'base', 'small', 'medium', 'large-v3-turbo', 'large-v3']
    model_order = [m for m in all_models if m in df['model_size'].values]
    df_sorted = df.set_index('model_size').loc[model_order].reset_index()

    bars = plt.bar(df_sorted['model_size'], df_sorted['inference_time_sec'],
                   color='coral', alpha=0.8)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=10)

    plt.xlabel('Model Size', fontsize=12, fontweight='bold')
    plt.ylabel('Inference Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Inference Time by Whisper Model Size', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_file = OUTPUT_DIR / "speed_by_size.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_accuracy_speed_tradeoff(df):
    """Plot WER vs inference time scatter"""
    plt.figure(figsize=(12, 8))

    # Create scatter plot
    plt.scatter(df['inference_time_sec'], df['wer'],
               s=200, alpha=0.6, c='steelblue', edgecolors='black', linewidth=1.5)

    # Add labels for each point
    for idx, row in df.iterrows():
        plt.annotate(row['model_size'],
                    (row['inference_time_sec'], row['wer']),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')

    plt.xlabel('Inference Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Word Error Rate (%)', fontsize=12, fontweight='bold')
    plt.title('Accuracy vs Speed Tradeoff\n(Lower and left is better)',
             fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = OUTPUT_DIR / "accuracy_speed_tradeoff.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_engine_comparison(df):
    """Compare faster-whisper vs openai-whisper"""
    if df is None or len(df) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # WER comparison
    engines = df['engine'].unique()
    wer_data = [df[df['engine'] == eng]['wer'].values[0] for eng in engines]

    bars1 = ax1.bar(engines, wer_data, color=['steelblue', 'coral'], alpha=0.8)
    ax1.set_ylabel('Word Error Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('WER Comparison (Base Model)', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, max(wer_data) * 1.2])

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)

    # Speed comparison
    speed_data = [df[df['engine'] == eng]['inference_time_sec'].values[0] for eng in engines]

    bars2 = ax2.bar(engines, speed_data, color=['steelblue', 'coral'], alpha=0.8)
    ax2.set_ylabel('Inference Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Speed Comparison (Base Model)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, max(speed_data) * 1.2])

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=10)

    # Add speedup annotation
    if len(speed_data) == 2:
        speedup = speed_data[1] / speed_data[0]
        ax2.text(0.5, 0.95, f'{speedup:.1f}x faster',
                transform=ax2.transAxes,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                fontsize=11, fontweight='bold')

    plt.tight_layout()

    output_file = OUTPUT_DIR / "engine_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_variants_comparison(df):
    """Compare different Whisper variants"""
    if df is None or len(df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # WER comparison
    variants = df['variant'].tolist()
    wer_data = df['wer'].tolist()
    colors = ['steelblue', 'coral', 'lightgreen'][:len(variants)]

    bars1 = axes[0, 0].bar(variants, wer_data, color=colors, alpha=0.8)
    axes[0, 0].set_ylabel('Word Error Rate (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('WER Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)

    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)

    # CER comparison
    cer_data = df['cer'].tolist()

    bars2 = axes[0, 1].bar(variants, cer_data, color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('Character Error Rate (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('CER Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)

    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)

    # Speed comparison
    speed_data = df['inference_time_sec'].tolist()

    bars3 = axes[1, 0].bar(variants, speed_data, color=colors, alpha=0.8)
    axes[1, 0].set_ylabel('Inference Time (seconds)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Speed Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)

    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}s',
                       ha='center', va='bottom', fontsize=9)

    # Accuracy-Speed scatter
    axes[1, 1].scatter(speed_data, wer_data, s=200, alpha=0.7,
                      c=colors, edgecolors='black', linewidth=1.5)

    for i, variant in enumerate(variants):
        axes[1, 1].annotate(variant, (speed_data[i], wer_data[i]),
                           xytext=(10, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold')

    axes[1, 1].set_xlabel('Inference Time (seconds)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Word Error Rate (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Accuracy vs Speed', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = OUTPUT_DIR / "variants_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def create_summary_stats(data):
    """Generate summary statistics"""
    summary = []

    if 'size' in data:
        df = data['size']
        summary.append("\n=== Model Size Comparison ===")
        summary.append(f"Best accuracy: {df.loc[df['wer'].idxmin(), 'model_size']} "
                      f"(WER: {df['wer'].min():.2f}%)")
        summary.append(f"Fastest: {df.loc[df['inference_time_sec'].idxmin(), 'model_size']} "
                      f"({df['inference_time_sec'].min():.2f}s)")

        # Best balance (lowest WER * time product)
        df['score'] = df['wer'] * df['inference_time_sec']
        best_balance = df.loc[df['score'].idxmin(), 'model_size']
        summary.append(f"Best balance: {best_balance}")

    if 'engine' in data:
        df = data['engine']
        summary.append("\n=== Engine Comparison ===")
        for engine in df['engine'].unique():
            engine_data = df[df['engine'] == engine].iloc[0]
            summary.append(f"{engine}: WER={engine_data['wer']:.2f}%, "
                          f"Time={engine_data['inference_time_sec']:.2f}s")

    if 'variants' in data:
        df = data['variants']
        summary.append("\n=== Variant Comparison ===")
        for _, row in df.iterrows():
            summary.append(f"{row['variant']}: WER={row['wer']:.2f}%, "
                          f"Time={row['inference_time_sec']:.2f}s")

    summary_text = "\n".join(summary)
    print(summary_text)

    # Save summary to file
    output_file = OUTPUT_DIR / "summary_stats.txt"
    with open(output_file, 'w') as f:
        f.write(summary_text)
    print(f"\nSaved: {output_file}")

def main():
    print("Loading benchmark results...")
    data = load_all_data()

    if not data:
        print("No data files found in results/")
        return

    print("\nGenerating visualizations...")

    # Size comparison plots
    if 'size' in data:
        plot_wer_by_size(data['size'])
        plot_speed_by_size(data['size'])
        plot_accuracy_speed_tradeoff(data['size'])

    # Engine comparison
    if 'engine' in data:
        plot_engine_comparison(data['engine'])

    # Variants comparison
    if 'variants' in data:
        plot_variants_comparison(data['variants'])

    # Generate summary stats
    create_summary_stats(data)

    print("\n✓ All visualizations generated successfully!")
    print(f"Charts saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
