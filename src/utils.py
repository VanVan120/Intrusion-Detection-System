import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import os

def evaluate_model(model, X_test, y_test, feature_count, runtime, method_name, save_path=None):
    """
    Evaluates the model and prints/saves standard metrics.
    """
    y_pred = model.predict(X_test)
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    metrics = {
        'Method': method_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Detection Rate (TPR)': tpr,
        'False Positive Rate (FPR)': fpr,
        'Feature Count': feature_count,
        'Runtime (s)': runtime
    }
    
    print(f"\n=== {method_name} Final Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {save_path}")
        
    return metrics, y_pred

def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_pareto_front(history, best_features_len, best_accuracy, title="Pareto Front", save_path=None, show=True):
    if not history:
        print("No history available for Pareto plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history['features'], history['accuracy'], color='gray', linestyle='--', alpha=0.5, label='Evolution Path')
    plt.scatter(history['features'], history['accuracy'], c=history['iteration'], cmap='viridis', s=100, edgecolors='black')
    plt.colorbar(label='Iteration/Generation')
    plt.title(title)
    plt.xlabel('Number of Features (Minimize) ->')
    plt.ylabel('Validation Accuracy (Maximize) ->')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.scatter([best_features_len], [best_accuracy], color='red', marker='*', s=300, label='Best Solution')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Pareto plot saved to {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close()
