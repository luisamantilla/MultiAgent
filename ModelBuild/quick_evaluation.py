"""
magnitude_plot.py - Cell type performance plot with comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_comprehensive_metrics(valid_df):
    """Calculate comprehensive evaluation metrics."""

    # Calculate magnitude accuracy (same order of magnitude)
    valid_df['log_gt'] = np.log10(np.abs(valid_df['ground_truth_value']) + 1e-10)
    valid_df['log_extracted'] = np.log10(np.abs(valid_df['parsed_value']) + 1e-10)
    valid_df['magnitude_accuracy'] = (
            np.abs(valid_df['log_gt'] - valid_df['log_extracted']) <= 1.0
    ).astype(float)

    # Calculate relative error and exact accuracy
    valid_df['relative_error'] = np.abs(
        (valid_df['parsed_value'] - valid_df['ground_truth_value']) /
        (valid_df['ground_truth_value'] + 1e-10)
    )
    valid_df['exact_accuracy'] = np.maximum(0, 1 - valid_df['relative_error'])

    # High accuracy threshold (within 20% of ground truth)
    valid_df['high_accuracy'] = (valid_df['relative_error'] <= 0.2).astype(float)

    # Extract success rate (any numerical value extracted)
    valid_df['extraction_success'] = 1.0  # All valid_df rows have extracted values

    return valid_df


def calculate_metrics_by_cell_type(valid_df):
    """Calculate detailed metrics by cell type."""

    metrics_summary = []

    for cell_type in valid_df['cell_type'].unique():
        cell_data = valid_df[valid_df['cell_type'] == cell_type]

        metrics = {
            'cell_type': cell_type,
            'total_parameters': len(cell_data),
            'extraction_success_rate': cell_data['extraction_success'].mean(),
            'magnitude_accuracy': cell_data['magnitude_accuracy'].mean(),
            'high_accuracy_rate': cell_data['high_accuracy'].mean(),
            'mean_exact_accuracy': cell_data['exact_accuracy'].mean(),
            'median_exact_accuracy': cell_data['exact_accuracy'].median(),
            'mean_confidence': cell_data['confidence_score'].mean(),
            'std_confidence': cell_data['confidence_score'].std(),
            'mean_relative_error': cell_data['relative_error'].mean(),
            'median_relative_error': cell_data['relative_error'].median(),
            'confidence_accuracy_correlation': cell_data['confidence_score'].corr(cell_data['magnitude_accuracy'])
        }

        # Calculate confidence calibration (how well confidence predicts accuracy)
        high_conf_mask = cell_data['confidence_score'] >= 0.7
        if high_conf_mask.sum() > 0:
            metrics['high_conf_magnitude_acc'] = cell_data[high_conf_mask]['magnitude_accuracy'].mean()
            metrics['high_conf_count'] = high_conf_mask.sum()
        else:
            metrics['high_conf_magnitude_acc'] = 0
            metrics['high_conf_count'] = 0

        low_conf_mask = cell_data['confidence_score'] < 0.5
        if low_conf_mask.sum() > 0:
            metrics['low_conf_magnitude_acc'] = cell_data[low_conf_mask]['magnitude_accuracy'].mean()
            metrics['low_conf_count'] = low_conf_mask.sum()
        else:
            metrics['low_conf_magnitude_acc'] = 0
            metrics['low_conf_count'] = 0

        metrics_summary.append(metrics)

    return pd.DataFrame(metrics_summary)


def print_detailed_metrics(metrics_df, valid_df):
    """Print comprehensive evaluation metrics."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("=" * 80)

    # Overall metrics
    print(f"\nOVERALL PERFORMANCE:")
    print(f"Total valid extractions: {len(valid_df)}")
    print(
        f"Overall magnitude accuracy: {valid_df['magnitude_accuracy'].mean():.3f} ({valid_df['magnitude_accuracy'].mean():.1%})")
    print(
        f"Overall high accuracy rate (±20%): {valid_df['high_accuracy'].mean():.3f} ({valid_df['high_accuracy'].mean():.1%})")
    print(f"Overall mean confidence: {valid_df['confidence_score'].mean():.3f}")
    print(f"Confidence-magnitude correlation: {valid_df['confidence_score'].corr(valid_df['magnitude_accuracy']):.3f}")

    # Cell type breakdown
    print(f"\nPERFORMANCE BY CELL TYPE:")
    print("-" * 100)
    print(
        f"{'Cell Type':<12} {'Count':<6} {'Mag Acc':<8} {'High Acc':<9} {'Mean Conf':<10} {'Conf Corr':<9} {'Med Error':<9}")
    print("-" * 100)

    for _, row in metrics_df.iterrows():
        print(f"{row['cell_type']:<12} {row['total_parameters']:<6} "
              f"{row['magnitude_accuracy']:<8.3f} {row['high_accuracy_rate']:<9.3f} "
              f"{row['mean_confidence']:<10.3f} {row['confidence_accuracy_correlation']:<9.3f} "
              f"{row['median_relative_error']:<9.3f}")

    # Confidence calibration analysis
    print(f"\nCONFIDENCE CALIBRATION ANALYSIS:")
    print("-" * 80)
    print(f"{'Cell Type':<12} {'High Conf':<10} {'High Conf':<10} {'Low Conf':<9} {'Low Conf':<9}")
    print(f"{'':12} {'Count':<10} {'Mag Acc':<10} {'Count':<9} {'Mag Acc':<9}")
    print("-" * 80)

    for _, row in metrics_df.iterrows():
        print(f"{row['cell_type']:<12} {row['high_conf_count']:<10} "
              f"{row['high_conf_magnitude_acc']:<10.3f} {row['low_conf_count']:<9} "
              f"{row['low_conf_magnitude_acc']:<9.3f}")

    # Problem identification
    print(f"\nPROBLEM IDENTIFICATION:")

    # High confidence but wrong extractions
    problem_cases = valid_df[
        (valid_df['confidence_score'] >= 0.7) &
        (valid_df['magnitude_accuracy'] == 0)
        ]

    if len(problem_cases) > 0:
        print(f"High-confidence wrong extractions: {len(problem_cases)}")
        print("Examples:")
        for _, row in problem_cases.head(3).iterrows():
            print(f"  {row['parameter_name']} ({row['cell_type']}): "
                  f"conf={row['confidence_score']:.2f}, "
                  f"extracted={row['parsed_value']:.2e}, truth={row['ground_truth_value']:.2e}")
    else:
        print("No high-confidence wrong extractions found!")

    # Low confidence but correct extractions
    good_low_conf = valid_df[
        (valid_df['confidence_score'] < 0.5) &
        (valid_df['magnitude_accuracy'] == 1)
        ]

    if len(good_low_conf) > 0:
        print(f"Low-confidence correct extractions: {len(good_low_conf)} (potential under-confidence)")

    # Best and worst performers
    print(f"\nBEST PERFORMING PARAMETERS:")
    best_params = valid_df[valid_df['magnitude_accuracy'] == 1].nlargest(5, 'confidence_score')
    for _, row in best_params.iterrows():
        print(f"  {row['parameter_name']} ({row['cell_type']}): "
              f"conf={row['confidence_score']:.3f}")

    print(f"\nWORST PERFORMING PARAMETERS:")
    worst_params = valid_df[valid_df['magnitude_accuracy'] == 0].nlargest(5, 'confidence_score')
    for _, row in worst_params.iterrows():
        print(f"  {row['parameter_name']} ({row['cell_type']}): "
              f"conf={row['confidence_score']:.3f}, error={row['relative_error']:.2f}")


def create_comprehensive_plot(metrics_df):
    """Create comprehensive performance visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    cell_types = metrics_df['cell_type']
    x_pos = range(len(cell_types))
    width = 0.35

    # Plot 1: Magnitude Accuracy vs Confidence
    axes[0, 0].bar([x - width / 2 for x in x_pos],
                   metrics_df['magnitude_accuracy'],
                   width, label='Magnitude Accuracy',
                   color='steelblue', alpha=0.8)
    axes[0, 0].bar([x + width / 2 for x in x_pos],
                   metrics_df['mean_confidence'],
                   width, label='Mean Confidence',
                   color='orange', alpha=0.8)

    axes[0, 0].set_xlabel('Cell Type')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Magnitude Accuracy vs Mean Confidence by Cell Type')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(cell_types)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim(0, 1.0)

    # Add value labels
    for i, (mag_acc, conf) in enumerate(zip(metrics_df['magnitude_accuracy'], metrics_df['mean_confidence'])):
        axes[0, 0].text(i - width / 2, mag_acc + 0.02, f'{mag_acc:.3f}',
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
        axes[0, 0].text(i + width / 2, conf + 0.02, f'{conf:.3f}',
                        ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Plot 2: Multiple Accuracy Metrics
    width2 = 0.25
    axes[0, 1].bar([x - width2 for x in x_pos],
                   metrics_df['magnitude_accuracy'],
                   width2, label='Magnitude Acc', color='steelblue', alpha=0.8)
    axes[0, 1].bar(x_pos,
                   metrics_df['high_accuracy_rate'],
                   width2, label='High Acc (±20%)', color='green', alpha=0.8)
    axes[0, 1].bar([x + width2 for x in x_pos],
                   metrics_df['mean_exact_accuracy'],
                   width2, label='Mean Exact Acc', color='red', alpha=0.8)

    axes[0, 1].set_xlabel('Cell Type')
    axes[0, 1].set_ylabel('Accuracy Rate')
    axes[0, 1].set_title('Multiple Accuracy Metrics by Cell Type')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(cell_types)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim(0, 1.0)

    # Plot 3: Confidence Calibration
    high_conf_acc = metrics_df['high_conf_magnitude_acc'].fillna(0)
    low_conf_acc = metrics_df['low_conf_magnitude_acc'].fillna(0)

    axes[1, 0].bar([x - width / 2 for x in x_pos],
                   high_conf_acc,
                   width, label='High Confidence (≥0.7)',
                   color='darkgreen', alpha=0.8)
    axes[1, 0].bar([x + width / 2 for x in x_pos],
                   low_conf_acc,
                   width, label='Low Confidence (<0.5)',
                   color='darkred', alpha=0.8)

    axes[1, 0].set_xlabel('Cell Type')
    axes[1, 0].set_ylabel('Magnitude Accuracy')
    axes[1, 0].set_title('Confidence Calibration: Accuracy by Confidence Level')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(cell_types)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim(0, 1.0)

    # Plot 4: Sample Count and Correlation
    axes[1, 1].bar(x_pos, metrics_df['total_parameters'],
                   color='purple', alpha=0.7, label='Sample Count')

    # Add correlation as text annotations
    ax2 = axes[1, 1].twinx()
    correlation_line = ax2.plot(x_pos, metrics_df['confidence_accuracy_correlation'],
                                'ro-', linewidth=2, markersize=8, label='Conf-Acc Correlation')
    ax2.set_ylabel('Confidence-Accuracy Correlation', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-1, 1)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    axes[1, 1].set_xlabel('Cell Type')
    axes[1, 1].set_ylabel('Sample Count')
    axes[1, 1].set_title('Sample Count and Confidence-Accuracy Correlation')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(cell_types)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Combined legend
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig('comprehensive_evaluation_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_magnitude_accuracy_plot():
    """Create comprehensive performance analysis with evaluation metrics."""

    # Load data
    df = pd.read_csv("Parameter_Extraction_Results_with_ground_truth.csv")

    # Filter valid data
    valid_df = df[
        (df['ground_truth_value'].notna()) &
        (df['parsed_value'].notna()) &
        (df['confidence_score'].notna())
        ].copy()

    print(f"Valid data: {len(valid_df)}/{len(df)} rows")

    if len(valid_df) == 0:
        print("No valid data found!")
        return

    # Calculate comprehensive metrics
    valid_df = calculate_comprehensive_metrics(valid_df)

    # Calculate metrics by cell type
    metrics_df = calculate_metrics_by_cell_type(valid_df)

    # Print detailed evaluation metrics
    print_detailed_metrics(metrics_df, valid_df)

    # Create comprehensive visualization
    create_comprehensive_plot(metrics_df)

    # Save metrics to CSV
    metrics_df.to_csv('evaluation_metrics_by_cell_type.csv', index=False)
    print(f"\nDetailed metrics saved to: evaluation_metrics_by_cell_type.csv")

    return metrics_df, valid_df


if __name__ == "__main__":
    metrics_df, valid_df = create_magnitude_accuracy_plot()
    print("Comprehensive evaluation complete!")
    print("Visualization saved as: comprehensive_evaluation_metrics.png")

