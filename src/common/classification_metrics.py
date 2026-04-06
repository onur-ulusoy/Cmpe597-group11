import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Computes standard classification metrics.
    
    Args:
        y_true (np.ndarray): True binary labels (0 or 1).
        y_pred (np.ndarray): Predicted binary labels (0 or 1).
        y_prob (np.ndarray, optional): Predicted probabilities for the positive class.
        
    Returns:
        dict: A dictionary containing Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            metrics["roc_auc"] = auc
        except ValueError:
            # ROC-AUC fails if there's only one class present in y_true
            metrics["roc_auc"] = 0.5
            
    return metrics

def print_classification_report(run_name, metrics):
    """Prints a formatted classification report."""
    print(f"\n=== {run_name} Classification Report ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("=" * (len(run_name) + 26))
