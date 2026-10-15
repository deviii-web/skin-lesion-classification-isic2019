"""
Direct diagnostic & metric evaluation to ensure 100% stable execution.
"""
import numpy as np

def compute_metrics(y_true, y_pred, num_classes=8):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred))

    f1s = []
    class_weights = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        support = np.sum(y_true == c)
        class_weights.append(support)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    total_samples = len(y_true)
    weighted_f1 = float(sum(f1 * (w / total_samples) for f1, w in zip(f1s, class_weights)))
    return acc, macro_f1, weighted_f1, f1s
