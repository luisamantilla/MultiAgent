#!/usr/bin/env python3
"""
evaluation_analysis.py - Comprehensive evaluation of parameter extraction results

This script analyzes the relationship between confidence scores and accuracy,
determines optimal confidence thresholds, and provides evaluation metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class ParameterExtractionEvaluator:
    """Comprehensive evaluator for parameter extraction results."""
    
    def __init__(self, csv_file_path):
        """Initialize with CSV file."""
        self.df = pd.read_csv(csv_file_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare and clean the data for analysis."""
        print("Preparing data for evaluation...")
        
        # Filter rows with both ground truth and extracted values
        self.valid_df = self.df[
            (self.df['ground_truth_value'].notna()) & 
            (self.df['parsed_value'].notna()) &
            (self.df['confidence_score'].notna())
        ].copy()
        
        print(f"Valid rows for analysis: {len(self.valid_df)}/{len(self.df)}")
        
        if len(self.valid_df) == 0:
            print("No valid rows found for analysis!")
            return
        
        # Calculate accuracy metrics
        self.calculate_accuracy_metrics()
        
        # Categorize confidence scores
        self.categorize_confidence()
        
        print(f"Cell type distribution in valid data:")
        print(self.valid_df['cell_type'].value_counts())
        
    def calculate_accuracy_metrics(self):
        """Calculate various accuracy metrics."""
        
        # Relative error
        self.valid_df['relative_error'] = np.abs(
            (self.valid_df['parsed_value'] - self.valid_df['ground_truth_value']) / 
            self.valid_df['ground_truth_value']
        )
        
        # Handle division by zero (when ground truth is 0)
        zero_gt_mask = self.valid_df['ground_truth_value'] == 0
        self.valid_df.loc[zero_gt_mask, 'relative_error'] = np.abs(self.valid_df.loc[zero_gt_mask, 'parsed_value'])
        
        # Accuracy (1 - relative_error, capped at 0)
        self.valid_df['accuracy'] = np.maximum(0, 1 - self.valid_df['relative_error'])
        
        # Order of magnitude accuracy (same order of magnitude = 1, else 0)
        self.valid_df['log_gt'] = np.log10(np.abs(self.valid_df['ground_truth_value']) + 1e-10)
        self.valid_df['log_extracted'] = np.log10(np.abs(self.valid_df['parsed_value']) + 1e-10)
        self.valid_df['magnitude_accuracy'] = (
            np.abs(self.valid_df['log_gt'] - self.valid_df['log_extracted']) <= 1.0
        ).astype(int)
        
        # Binary success metrics
        self.valid_df['high_accuracy'] = (self.valid_df['accuracy'] > 0.8).astype(int)
        self.valid_df['moderate_accuracy'] = (self.valid_df['accuracy'] > 0.5).astype(int)
        self.valid_df['same_magnitude'] = self.valid_df['magnitude_accuracy']
        
    def categorize_confidence(self):
        """Categorize confidence scores."""
        self.valid_df['confidence_category'] = pd.cut(
            self.valid_df['confidence_score'],
            bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
            labels=['Very Low (0-0.3)', 'Low (0.3-0.5)', 'Medium (0.5-0.7)', 
                   'High (0.7-0.9)', 'Very High (0.9-1.0)']
        )
    
    def analyze_confidence_accuracy_relationship(self):
        """Analyze the relationship between confidence and accuracy."""
        print("\n" + "="*60)
        print("CONFIDENCE-ACCURACY RELATIONSHIP ANALYSIS")
        print("="*60)
        
        # Overall correlation
        corr_accuracy = self.valid_df['confidence_score'].corr(self.valid_df['accuracy'])
        corr_magnitude = self.valid_df['confidence_score'].corr(self.valid_df['magnitude_accuracy'])
        
        print(f"Correlation between confidence and accuracy: {corr_accuracy:.3f}")
        print(f"Correlation between confidence and magnitude accuracy: {corr_magnitude:.3f}")
        
        # By confidence category
        print(f"\nAccuracy by confidence category:")
        conf_analysis = self.valid_df.groupby('confidence_category').agg({
            'accuracy': ['mean', 'std', 'count'],
            'magnitude_accuracy': 'mean',
            'high_accuracy': 'mean'
        }).round(3)
        print(conf_analysis)
        
        # By cell type
        print(f"\nAccuracy by cell type:")
        cell_analysis = self.valid_df.groupby('cell_type').agg({
            'accuracy': ['mean', 'std', 'count'],
            'confidence_score': 'mean',
            'magnitude_accuracy': 'mean'
        }).round(3)
        print(cell_analysis)
        
        return corr_accuracy, corr_magnitude
    
    def find_optimal_confidence_threshold(self):
        """Find optimal confidence threshold for different criteria."""
        print(f"\n" + "="*60)
        print("OPTIMAL CONFIDENCE THRESHOLD ANALYSIS")
        print("="*60)
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        results = []
        
        for threshold in thresholds:
            reliable_mask = self.valid_df['confidence_score'] >= threshold
            reliable_data = self.valid_df[reliable_mask]
            
            if len(reliable_data) == 0:
                continue
                
            # Calculate metrics for reliable predictions
            precision_high = reliable_data['high_accuracy'].mean()
            precision_magnitude = reliable_data['magnitude_accuracy'].mean()
            recall_high = reliable_data['high_accuracy'].sum() / self.valid_df['high_accuracy'].sum()
            recall_magnitude = reliable_data['magnitude_accuracy'].sum() / self.valid_df['magnitude_accuracy'].sum()
            coverage = len(reliable_data) / len(self.valid_df)
            
            # F1 scores
            f1_high = 2 * (precision_high * recall_high) / (precision_high + recall_high) if (precision_high + recall_high) > 0 else 0
            f1_magnitude = 2 * (precision_magnitude * recall_magnitude) / (precision_magnitude + recall_magnitude) if (precision_magnitude + recall_magnitude) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'precision_high_acc': precision_high,
                'recall_high_acc': recall_high,
                'f1_high_acc': f1_high,
                'precision_magnitude': precision_magnitude,
                'recall_magnitude': recall_magnitude,
                'f1_magnitude': f1_magnitude,
                'coverage': coverage,
                'n_reliable': len(reliable_data)
            })
        
        threshold_df = pd.DataFrame(results)
        
        # Find optimal thresholds
        best_f1_high = threshold_df.loc[threshold_df['f1_high_acc'].idxmax()]
        best_f1_magnitude = threshold_df.loc[threshold_df['f1_magnitude'].idxmax()]
        best_precision_magnitude = threshold_df.loc[threshold_df['precision_magnitude'].idxmax()]
        
        print(f"Best threshold for high accuracy F1: {best_f1_high['threshold']:.2f} (F1: {best_f1_high['f1_high_acc']:.3f})")
        print(f"Best threshold for magnitude F1: {best_f1_magnitude['threshold']:.2f} (F1: {best_f1_magnitude['f1_magnitude']:.3f})")
        print(f"Best threshold for magnitude precision: {best_precision_magnitude['threshold']:.2f} (Precision: {best_precision_magnitude['precision_magnitude']:.3f})")
        
        # Recommended threshold based on magnitude accuracy (most practical)
        recommended_threshold = best_f1_magnitude['threshold']
        
        print(f"\nRECOMMENDED CONFIDENCE THRESHOLD: {recommended_threshold:.2f}")
        print(f"At this threshold:")
        print(f"  - {best_f1_magnitude['precision_magnitude']:.1%} of confident predictions are same magnitude")
        print(f"  - Covers {best_f1_magnitude['coverage']:.1%} of all predictions")
        print(f"  - Captures {best_f1_magnitude['recall_magnitude']:.1%} of all correct magnitude predictions")
        
        return threshold_df, recommended_threshold
    
    def evaluate_by_cell_type(self):
        """Detailed evaluation by cell type."""
        print(f"\n" + "="*60)
        print("CELL TYPE SPECIFIC EVALUATION")
        print("="*60)
        
        cell_types = self.valid_df['cell_type'].unique()
        
        for cell_type in cell_types:
            cell_data = self.valid_df[self.valid_df['cell_type'] == cell_type]
            
            print(f"\n--- {cell_type.upper()} CELLS ---")
            print(f"Total parameters: {len(cell_data)}")
            print(f"Mean accuracy: {cell_data['accuracy'].mean():.3f}")
            print(f"Mean confidence: {cell_data['confidence_score'].mean():.3f}")
            print(f"Same magnitude rate: {cell_data['magnitude_accuracy'].mean():.3f}")
            print(f"High accuracy rate (>80%): {cell_data['high_accuracy'].mean():.3f}")
            
            # Confidence-accuracy correlation for this cell type
            corr = cell_data['confidence_score'].corr(cell_data['accuracy'])
            print(f"Confidence-accuracy correlation: {corr:.3f}")
            
            # Most/least accurate parameters
            best_params = cell_data.nlargest(3, 'accuracy')[['parameter_name', 'accuracy', 'confidence_score']]
            worst_params = cell_data.nsmallest(3, 'accuracy')[['parameter_name', 'accuracy', 'confidence_score']]
            
            print(f"Best parameters:")
            for _, row in best_params.iterrows():
                print(f"  {row['parameter_name']}: acc={row['accuracy']:.3f}, conf={row['confidence_score']:.3f}")
            
            print(f"Worst parameters:")
            for _, row in worst_params.iterrows():
                print(f"  {row['parameter_name']}: acc={row['accuracy']:.3f}, conf={row['confidence_score']:.3f}")
    
    def create_visualizations(self, recommended_threshold):
        """Create comprehensive visualizations."""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Confidence vs Accuracy Scatter Plot
        plt.subplot(3, 4, 1)
        plt.scatter(self.valid_df['confidence_score'], self.valid_df['accuracy'], 
                   alpha=0.6, c=self.valid_df['cell_type'].astype('category').cat.codes, cmap='tab10')
        plt.xlabel('Confidence Score')
        plt.ylabel('Accuracy')
        plt.title('Confidence vs Accuracy')
        plt.axvline(recommended_threshold, color='red', linestyle='--', label=f'Recommended threshold: {recommended_threshold:.2f}')
        plt.legend()
        
        # 2. Confidence Distribution by Cell Type
        plt.subplot(3, 4, 2)
        for cell_type in self.valid_df['cell_type'].unique():
            cell_data = self.valid_df[self.valid_df['cell_type'] == cell_type]
            plt.hist(cell_data['confidence_score'], alpha=0.5, label=cell_type, bins=20)
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution by Cell Type')
        plt.legend()
        
        # 3. Accuracy Distribution by Cell Type
        plt.subplot(3, 4, 3)
        self.valid_df.boxplot(column='accuracy', by='cell_type', ax=plt.gca())
        plt.title('Accuracy by Cell Type')
        plt.suptitle('')  # Remove default title
        
        # 4. Confidence vs Magnitude Accuracy
        plt.subplot(3, 4, 4)
        plt.scatter(self.valid_df['confidence_score'], self.valid_df['magnitude_accuracy'], 
                   alpha=0.6, c=self.valid_df['cell_type'].astype('category').cat.codes, cmap='tab10')
        plt.xlabel('Confidence Score')
        plt.ylabel('Same Magnitude (0/1)')
        plt.title('Confidence vs Magnitude Accuracy')
        plt.axvline(recommended_threshold, color='red', linestyle='--')
        
        # 5. Confidence Category Analysis
        plt.subplot(3, 4, 5)
        conf_acc = self.valid_df.groupby('confidence_category')['accuracy'].mean()
        conf_acc.plot(kind='bar', rot=45)
        plt.title('Mean Accuracy by Confidence Category')
        plt.ylabel('Mean Accuracy')
        
        # 6. Cell Type vs Confidence Heatmap
        plt.subplot(3, 4, 6)
        heatmap_data = self.valid_df.pivot_table(values='accuracy', index='cell_type', 
                                                columns='confidence_category', aggfunc='mean')
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn')
        plt.title('Accuracy Heatmap: Cell Type vs Confidence')
        
        # 7. Relative Error Distribution
        plt.subplot(3, 4, 7)
        self.valid_df.boxplot(column='relative_error', by='cell_type', ax=plt.gca())
        plt.title('Relative Error by Cell Type')
        plt.yscale('log')
        plt.suptitle('')
        
        # 8. Confidence vs Log Relative Error
        plt.subplot(3, 4, 8)
        plt.scatter(self.valid_df['confidence_score'], np.log10(self.valid_df['relative_error'] + 1e-6), 
                   alpha=0.6, c=self.valid_df['cell_type'].astype('category').cat.codes, cmap='tab10')
        plt.xlabel('Confidence Score')
        plt.ylabel('Log Relative Error')
        plt.title('Confidence vs Log Relative Error')
        plt.axvline(recommended_threshold, color='red', linestyle='--')
        
        # 9. Success Rate at Different Thresholds
        plt.subplot(3, 4, 9)
        thresholds = np.arange(0.1, 1.0, 0.05)
        success_rates = []
        coverage_rates = []
        
        for threshold in thresholds:
            reliable_data = self.valid_df[self.valid_df['confidence_score'] >= threshold]
            if len(reliable_data) > 0:
                success_rate = reliable_data['magnitude_accuracy'].mean()
                coverage_rate = len(reliable_data) / len(self.valid_df)
            else:
                success_rate = 0
                coverage_rate = 0
            success_rates.append(success_rate)
            coverage_rates.append(coverage_rate)
        
        plt.plot(thresholds, success_rates, label='Success Rate', linewidth=2)
        plt.plot(thresholds, coverage_rates, label='Coverage Rate', linewidth=2)
        plt.axvline(recommended_threshold, color='red', linestyle='--', label='Recommended')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Rate')
        plt.title('Success vs Coverage by Threshold')
        plt.legend()
        
        # 10. Parameter-wise Performance
        plt.subplot(3, 4, 10)
        param_performance = self.valid_df.groupby('parameter_name').agg({
            'accuracy': 'mean',
            'confidence_score': 'mean'
        }).reset_index()
        
        plt.scatter(param_performance['confidence_score'], param_performance['accuracy'], alpha=0.7)
        plt.xlabel('Mean Confidence')
        plt.ylabel('Mean Accuracy')
        plt.title('Parameter-wise Performance')
        
        # 11. Ground Truth vs Extracted Values
        plt.subplot(3, 4, 11)
        plt.scatter(np.log10(self.valid_df['ground_truth_value'] + 1e-6), 
                   np.log10(self.valid_df['parsed_value'] + 1e-6),
                   alpha=0.6, c=self.valid_df['confidence_score'], cmap='viridis')
        plt.plot([-6, 6], [-6, 6], 'r--', alpha=0.7)  # Perfect prediction line
        plt.xlabel('Log Ground Truth')
        plt.ylabel('Log Extracted Value')
        plt.title('Ground Truth vs Extracted (log scale)')
        plt.colorbar(label='Confidence Score')
        
        # 12. Cell Type Performance Summary
        plt.subplot(3, 4, 12)
        cell_summary = self.valid_df.groupby('cell_type').agg({
            'accuracy': 'mean',
            'magnitude_accuracy': 'mean',
            'confidence_score': 'mean'
        })
        
        x = range(len(cell_summary))
        width = 0.25
        plt.bar([i - width for i in x], cell_summary['accuracy'], width, label='Accuracy', alpha=0.7)
        plt.bar(x, cell_summary['magnitude_accuracy'], width, label='Magnitude Acc', alpha=0.7)
        plt.bar([i + width for i in x], cell_summary['confidence_score'], width, label='Confidence', alpha=0.7)
        
        plt.xlabel('Cell Type')
        plt.ylabel('Score')
        plt.title('Performance Summary by Cell Type')
        plt.xticks(x, cell_summary.index, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('parameter_extraction_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_critique_agent_guidelines(self, recommended_threshold):
        """Generate guidelines for the CritiqueAgent."""
        print(f"\n" + "="*60)
        print("CRITIQUE AGENT EVALUATION GUIDELINES")
        print("="*60)
        
        print(f"RECOMMENDED CONFIDENCE THRESHOLD: {recommended_threshold:.2f}")
        print(f"\nGuidelines for CritiqueAgent validation:")
        
        # Calculate statistics at recommended threshold
        reliable_data = self.valid_df[self.valid_df['confidence_score'] >= recommended_threshold]
        
        print(f"\n1. CONFIDENCE-BASED FILTERING:")
        print(f"   - Accept extractions with confidence >= {recommended_threshold:.2f}")
        print(f"   - At this threshold: {len(reliable_data)}/{len(self.valid_df)} predictions are considered reliable")
        print(f"   - {reliable_data['magnitude_accuracy'].mean():.1%} of reliable predictions are correct magnitude")
        
        print(f"\n2. MAGNITUDE VALIDATION:")
        print(f"   - Check if extracted value is within same order of magnitude as expected")
        print(f"   - For biological parameters, 1-2 orders of magnitude difference may be acceptable")
        print(f"   - Flag extractions with >2 orders of magnitude difference")
        
        print(f"\n3. CELL-TYPE SPECIFIC EXPECTATIONS:")
        cell_stats = self.valid_df.groupby('cell_type').agg({
            'accuracy': 'mean',
            'confidence_score': 'mean'
        })
        
        for cell_type, stats in cell_stats.iterrows():
            print(f"   - {cell_type}: Expected accuracy ~{stats['accuracy']:.2f}, confidence ~{stats['confidence_score']:.2f}")
        
        print(f"\n4. ADDITIONAL VALIDATION CRITERIA:")
        print(f"   - Biological plausibility check")
        print(f"   - Source citation verification")
        print(f"   - Units consistency check")
        print(f"   - Reasoning quality assessment")
        
        print(f"\n5. CONFIDENCE SCORE INTERPRETATION:")
        conf_interpretation = self.valid_df.groupby('confidence_category').agg({
            'magnitude_accuracy': 'mean',
            'accuracy': 'mean'
        })
        
        for category, stats in conf_interpretation.iterrows():
            print(f"   - {category}: {stats['magnitude_accuracy']:.1%} magnitude accuracy, {stats['accuracy']:.2f} mean accuracy")
    
    def run_full_evaluation(self):
        """Run the complete evaluation pipeline."""
        print("PARAMETER EXTRACTION EVALUATION REPORT")
        print("="*60)
        
        if len(self.valid_df) == 0:
            print("No valid data for evaluation!")
            return
        
        # 1. Basic relationship analysis
        corr_acc, corr_mag = self.analyze_confidence_accuracy_relationship()
        
        # 2. Find optimal threshold
        threshold_df, recommended_threshold = self.find_optimal_confidence_threshold()
        
        # 3. Cell type analysis
        self.evaluate_by_cell_type()
        
        # 4. Generate guidelines
        self.generate_critique_agent_guidelines(recommended_threshold)
        
        # 5. Create visualizations
        self.create_visualizations(recommended_threshold)
        
        return {
            'correlation_accuracy': corr_acc,
            'correlation_magnitude': corr_mag,
            'recommended_threshold': recommended_threshold,
            'threshold_analysis': threshold_df
        }

def main():
    """Main function to run the evaluation."""
    csv_file = "Parameter_Extraction_Results_with_ground_truth.csv"
    
    print("Loading parameter extraction results...")
    evaluator = ParameterExtractionEvaluator(csv_file)
    
    print("Running comprehensive evaluation...")
    results = evaluator.run_full_evaluation()
    
    print(f"\nEvaluation complete!")
    print(f"Visualization saved as: parameter_extraction_evaluation.png")
    
    return results

if __name__ == "__main__":
    results = main()
