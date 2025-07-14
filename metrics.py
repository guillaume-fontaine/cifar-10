import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from typing import Dict, List

class MetricsCalculator:
    """Calculateur de métriques pour la classification"""

    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names

    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Calcule toutes les métriques"""
        # Accuracy globale
        accuracy = np.mean(predictions == targets)

        # Métriques par classe
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )

        # Métriques moyennes
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)

        # Accuracy par classe
        class_accuracy = []
        for i in range(self.num_classes):
            mask = targets == i
            if mask.sum() > 0:
                class_acc = np.mean(predictions[mask] == targets[mask])
                class_accuracy.append(class_acc)
            else:
                class_accuracy.append(0.0)

        # Matrice de confusion
        cm = confusion_matrix(targets, predictions)

        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'accuracy_per_class': class_accuracy,
            'confusion_matrix': cm
        }

    def calculate_top_k_accuracy(self, logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """Calcule la top-k accuracy"""
        _, top_k_pred = torch.topk(logits, k, dim=1)
        targets_expanded = targets.view(-1, 1).expand_as(top_k_pred)
        correct = torch.sum(top_k_pred == targets_expanded, dim=1)
        return torch.mean(correct.float()).item()
