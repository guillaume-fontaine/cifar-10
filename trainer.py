import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
from metrics import MetricsCalculator

import torch.nn as nn

logger = logging.getLogger(__name__)

class Trainer:
    """Gestionnaire d'entraînement"""
    def __init__(self, model: nn.Module, config, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = nn.CrossEntropyLoss()
        self.metrics_calculator = MetricsCalculator(config.NUM_CLASSES, config.CLASS_NAMES)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f"{config.LOG_DIR}/cifar10_{timestamp}")
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_history = []
        self.val_history = []
    def train_epoch(self, train_loader, epoch: int):
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        avg_loss = running_loss / len(train_loader)
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_predictions), np.array(all_targets)
        )
        return {'loss': avg_loss, **metrics}
    def validate_epoch(self, val_loader, epoch: int):
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        all_logits = []
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_logits.append(outputs.cpu())
        avg_loss = running_loss / len(val_loader)
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_predictions), np.array(all_targets)
        )
        all_logits = torch.cat(all_logits, dim=0)
        all_targets_tensor = torch.tensor(all_targets)
        top5_acc = self.metrics_calculator.calculate_top_k_accuracy(
            all_logits, all_targets_tensor, k=5
        )
        return {'loss': avg_loss, 'top5_accuracy': top5_acc, **metrics}
    def log_metrics(self, train_metrics, val_metrics, epoch: int):
        self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
        self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
        self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
        self.writer.add_scalar('Accuracy/Top5_Validation', val_metrics['top5_accuracy'], epoch)
        self.writer.add_scalar('Precision/Train_Macro', train_metrics['precision_macro'], epoch)
        self.writer.add_scalar('Precision/Validation_Macro', val_metrics['precision_macro'], epoch)
        self.writer.add_scalar('Recall/Train_Macro', train_metrics['recall_macro'], epoch)
        self.writer.add_scalar('Recall/Validation_Macro', val_metrics['recall_macro'], epoch)
        self.writer.add_scalar('F1/Train_Macro', train_metrics['f1_macro'], epoch)
        self.writer.add_scalar('F1/Validation_Macro', val_metrics['f1_macro'], epoch)
        for i, class_name in enumerate(self.config.CLASS_NAMES):
            self.writer.add_scalar(f'Accuracy_PerClass/Train_{class_name}',
                                 train_metrics['accuracy_per_class'][i], epoch)
            self.writer.add_scalar(f'Accuracy_PerClass/Validation_{class_name}',
                                 val_metrics['accuracy_per_class'][i], epoch)
            self.writer.add_scalar(f'Precision_PerClass/Train_{class_name}',
                                 train_metrics['precision_per_class'][i], epoch)
            self.writer.add_scalar(f'Precision_PerClass/Validation_{class_name}',
                                 val_metrics['precision_per_class'][i], epoch)
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
    def save_model(self, filepath: str, epoch: int, metrics):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__
        }, filepath)
    def train(self, train_loader, val_loader):
        logger.info("Début de l'entraînement...")
        for epoch in range(self.config.NUM_EPOCHS):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate_epoch(val_loader, epoch)
            self.log_metrics(train_metrics, val_metrics, epoch)
            self.scheduler.step(val_metrics['loss'])
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                Path(self.config.SAVE_DIR).mkdir(exist_ok=True)
                self.save_model(
                    f"{self.config.SAVE_DIR}/best_model.pth",
                    epoch, val_metrics
                )
                logger.info(f"Nouveau meilleur modèle sauvegardé à l'époque {epoch}")
            else:
                self.patience_counter += 1
            logger.info(f"Époque {epoch+1}/{self.config.NUM_EPOCHS}")
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Top5: {val_metrics['top5_accuracy']:.4f}")
            if self.patience_counter >= self.config.PATIENCE:
                logger.info(f"Early stopping à l'époque {epoch+1}")
                break
        self.writer.close()
        logger.info("Entraînement terminé!")

