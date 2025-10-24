#!/usr/bin/env python3
"""
Comprehensive Evaluation Metrics
Handles imbalanced datasets with appropriate metrics
UPDATED FOR YOUR ACTUAL DATASET DISTRIBUTION
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class ImbalancedDatasetMetrics:
    """Comprehensive metrics for imbalanced Alzheimer's dataset"""
    
    def __init__(self, class_names=None):
        if class_names is None:
            # UPDATED: Match your actual dataset class names
            self.class_names = ['Normal', 'VeryMild', 'Mild', 'Moderate']
        else:
            self.class_names = class_names
        
        self.num_classes = len(self.class_names)
    
    def calculate_all_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive metrics for imbalanced dataset"""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        
        # Per-class and averaged metrics
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['precision_micro'] = float(precision_score(y_true, y_pred, average='micro', zero_division=0))
        metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_micro'] = float(recall_score(y_true, y_pred, average='micro', zero_division=0))
        metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_micro'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class_metrics'] = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(per_class_precision):
                metrics['per_class_metrics'][class_name] = {
                    'precision': float(per_class_precision[i]),
                    'recall': float(per_class_recall[i]),
                    'f1_score': float(per_class_f1[i])
                }
        
        # Advanced metrics for imbalanced data
        metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
        metrics['matthews_corrcoef'] = float(matthews_corrcoef(y_true, y_pred))
        
        # AUC metrics (if probabilities available)
        if y_proba is not None:
            try:
                if self.num_classes == 2:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    metrics['roc_auc_ovr'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
                    metrics['roc_auc_ovo'] = float(roc_auc_score(y_true, y_proba, multi_class='ovo', average='macro'))
                    
                    # Per-class AUC
                    metrics['per_class_auc'] = {}
                    for i, class_name in enumerate(self.class_names):
                        if i < y_proba.shape[1]:
                            # One-vs-rest AUC for each class
                            y_true_binary = (y_true == i).astype(int)
                            metrics['per_class_auc'][class_name] = float(roc_auc_score(y_true_binary, y_proba[:, i]))
            except Exception as e:
                print(f"Warning: Could not calculate AUC metrics: {e}")
                metrics['roc_auc_ovr'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Class-wise accuracy (sensitivity/recall per class)
        metrics['class_wise_accuracy'] = {}
        for i, class_name in enumerate(self.class_names):
            if i < cm.shape[0]:
                if cm[i, :].sum() > 0:
                    class_acc = cm[i, i] / cm[i, :].sum()
                    metrics['class_wise_accuracy'][class_name] = float(class_acc)
                else:
                    metrics['class_wise_accuracy'][class_name] = 0.0
        
        return metrics
    
    def calculate_class_weights(self, y_train):
        """Calculate class weights for imbalanced dataset"""
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        
        weight_dict = {}
        for i, weight in enumerate(weights):
            if i < len(self.class_names):
                weight_dict[self.class_names[i]] = float(weight)
        
        return weights.tolist(), weight_dict
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None, normalize=False):
        """Plot confusion matrix with class names"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        
        # Get actual class names based on confusion matrix size
        actual_classes = self.class_names[:cm.shape[0]]
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=actual_classes,
                   yticklabels=actual_classes)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add class-wise accuracy
        for i in range(cm.shape[0]):
            if cm[i, :].sum() > 0:
                acc = cm[i, i] / cm[i, :].sum()
                plt.text(cm.shape[1] + 0.1, i + 0.5, f'Recall: {acc:.3f}',
                        verticalalignment='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return save_path
        else:
            plt.show()
        
        plt.close()
    
    def plot_class_distribution(self, y_data, title="Class Distribution", save_path=None):
        """Plot class distribution"""
        unique, counts = np.unique(y_data, return_counts=True)
        
        plt.figure(figsize=(12, 6))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        class_names_subset = [self.class_names[i] for i in unique if i < len(self.class_names)]
        bars = plt.bar(class_names_subset, counts, alpha=0.8, color='skyblue')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.title(f'{title} - Counts')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=class_names_subset, autopct='%1.1f%%', startangle=90)
        plt.title(f'{title} - Percentages')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return save_path
        else:
            plt.show()
        
        plt.close()
    
    def generate_classification_report(self, y_true, y_pred, save_path=None):
        """Generate detailed classification report"""
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names,
                                     output_dict=True, zero_division=0)
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        # Print formatted report
        print("Classification Report:")
        print("=" * 60)
        print(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))
        
        return report
    
    def print_imbalance_analysis(self, y_data):
        """Analyze and print class imbalance information"""
        unique, counts = np.unique(y_data, return_counts=True)
        total = len(y_data)
        
        print("📊 Class Imbalance Analysis:")
        print("=" * 50)
        
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        
        for i, (class_idx, count) in enumerate(zip(unique, counts)):
            if class_idx < len(self.class_names):
                percentage = count / total * 100
                ratio = max_count / count
                print(f"{self.class_names[class_idx]}: {count:,} samples ({percentage:.1f}%) - Ratio: 1:{ratio:.1f}")
        
        print(f"\nOverall Imbalance Ratio: 1:{imbalance_ratio:.1f}")
        
        # Recommendations
        if imbalance_ratio > 10:
            print("⚠️  Severe imbalance detected! Recommendations:")
            print("   - Use stratified sampling")
            print("   - Apply aggressive data augmentation to minority classes")
            print("   - Use class weights in loss function")
            print("   - Consider SMOTE or similar oversampling techniques")
            print("   - Focus on balanced accuracy, F1-score, and per-class metrics")
        elif imbalance_ratio > 3:
            print("⚠️  Moderate imbalance detected! Recommendations:")
            print("   - Use class weights")
            print("   - Apply data augmentation")
            print("   - Monitor per-class performance")

def calculate_augmentation_multipliers(class_counts, target_samples=None):
    """Calculate how much to augment each class to balance dataset"""
    
    if target_samples is None:
        target_samples = max(class_counts)
    
    multipliers = {}
    # UPDATED: Match your actual dataset class names
    class_names = ['Normal', 'VeryMild', 'Mild', 'Moderate']
    
    for i, count in enumerate(class_counts):
        if i < len(class_names):
            multiplier = max(1, target_samples // count)
            multipliers[class_names[i]] = {
                'original_count': count,
                'target_count': target_samples,
                'augmentation_multiplier': multiplier,
                'additional_samples_needed': max(0, target_samples - count)
            }
    
    return multipliers

# Example usage and testing
if __name__ == "__main__":
    # UPDATED: Your actual dataset distribution from test results
    print("🧠 Alzheimer's Dataset Metrics Analysis")
    
    # Your actual class distribution from test results
    class_counts = [7811, 8663, 5002, 488]  # Normal, VeryMild, Mild, Moderate
    class_names = ['Normal', 'VeryMild', 'Mild', 'Moderate']
    
    metrics = ImbalancedDatasetMetrics(class_names)
    
    # Simulate labels based on your distribution
    y_example = []
    for i, count in enumerate(class_counts):
        y_example.extend([i] * min(count, 1000))  # Limit for example
    
    y_example = np.array(y_example)
    
    # Print imbalance analysis
    metrics.print_imbalance_analysis(y_example)
    
    # Calculate augmentation strategy
    print(f"\n📈 Augmentation Strategy:")
    multipliers = calculate_augmentation_multipliers(class_counts)
    for class_name, info in multipliers.items():
        print(f"{class_name}:")
        print(f"  Current: {info['original_count']:,} samples")
        print(f"  Target: {info['target_count']:,} samples") 
        print(f"  Augmentation multiplier: {info['augmentation_multiplier']}x")
        print(f"  Additional samples needed: {info['additional_samples_needed']:,}")
        print()
    
    print("✅ Metrics configuration ready for your dataset!")
