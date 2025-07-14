import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple


class MetricsCalculator:
    """Calculateur de métriques pour la classification"""

    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names

    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray,
                          logits: torch.Tensor = None) -> Dict:
        """Calcule toutes les métriques principales"""

        # 1. Accuracy globale
        accuracy = np.mean(predictions == targets)

        # 2. Accuracy par classe
        class_accuracy = []
        for i in range(self.num_classes):
            mask = targets == i
            if mask.sum() > 0:
                class_acc = np.mean(predictions[mask] == targets[mask])
                class_accuracy.append(class_acc)
            else:
                class_accuracy.append(0.0)

        # 3. Matrice de confusion
        cm = confusion_matrix(targets, predictions)

        # 4. Precision, Recall, F1-Score
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )

        # Métriques moyennes
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)

        # 5. Top-K Accuracy (si les logits sont fournis)
        top_k_accuracies = {}
        if logits is not None:
            targets_tensor = torch.tensor(targets) if isinstance(targets, np.ndarray) else targets
            for k in [1, 3, 5]:
                top_k_acc = self.calculate_top_k_accuracy(logits, targets_tensor, k)
                top_k_accuracies[f'top_{k}_accuracy'] = top_k_acc

        results = {
            # Métriques principales
            'accuracy': accuracy,
            'accuracy_per_class': class_accuracy,
            'confusion_matrix': cm,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,

            # Métriques détaillées par classe
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,

            # Top-K accuracies
            **top_k_accuracies
        }

        return results

    def calculate_top_k_accuracy(self, logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
        """Calcule la top-k accuracy"""
        _, top_k_pred = torch.topk(logits, k, dim=1)
        targets_expanded = targets.view(-1, 1).expand_as(top_k_pred)
        correct = torch.sum(top_k_pred == targets_expanded, dim=1)
        return torch.mean(correct.float()).item()

    def print_detailed_report(self, metrics: Dict) -> None:
        """Affiche un rapport détaillé des métriques"""
        print("=" * 60)
        print("RAPPORT DÉTAILLÉ DES MÉTRIQUES")
        print("=" * 60)

        # 1. Accuracy globale
        print(f"Accuracy globale: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")

        # 2. Accuracy par classe
        print(f"\nAccuracy par classe:")
        for i, (class_name, acc) in enumerate(zip(self.class_names, metrics['accuracy_per_class'])):
            print(f"  {class_name:12}: {acc:.4f} ({acc * 100:.2f}%)")

        # 3. Métriques globales
        print(f"\nMétriques globales (macro-average):")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")
        print(f"  F1-Score:  {metrics['f1_macro']:.4f}")

        # 4. Top-K accuracies
        if 'top_1_accuracy' in metrics:
            print(f"\nTop-K Accuracies:")
            for k in [1, 3, 5]:
                key = f'top_{k}_accuracy'
                if key in metrics:
                    print(f"  Top-{k}: {metrics[key]:.4f} ({metrics[key] * 100:.2f}%)")

        # 5. Matrice de confusion (résumé)
        print(f"\nMatrice de confusion:")
        cm = metrics['confusion_matrix']
        print("Classes les plus confondues:")
        self._print_confusion_summary(cm)

    def _print_confusion_summary(self, cm: np.ndarray) -> None:
        """Affiche un résumé des confusions les plus fréquentes"""
        # Trouve les confusions les plus fréquentes (hors diagonale)
        np.fill_diagonal(cm, 0)  # Ignore la diagonale
        confusion_pairs = []

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append((cm[i, j], i, j))

        # Trie par nombre de confusions décroissant
        confusion_pairs.sort(reverse=True)

        # Affiche les 3 principales confusions
        for count, true_class, pred_class in confusion_pairs[:3]:
            print(f"  {self.class_names[true_class]} → {self.class_names[pred_class]}: {count} erreurs")

    def get_class_performance_summary(self, metrics: Dict) -> List[Tuple[str, float, float, float]]:
        """Retourne un résumé des performances par classe"""
        summary = []
        for i, class_name in enumerate(self.class_names):
            acc = metrics['accuracy_per_class'][i]
            prec = metrics['precision_per_class'][i]
            rec = metrics['recall_per_class'][i]
            f1 = metrics['f1_per_class'][i]
            summary.append((class_name, acc, prec, rec, f1))
        return summary