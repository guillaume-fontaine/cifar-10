from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import logging
from config import Config

logger = logging.getLogger(__name__)

class DataAugmentation:
    """Gestionnaire des augmentations de données"""

    @staticmethod
    def get_val_transforms() -> transforms.Compose:
        """Transformations pour la validation (sans augmentations)"""
        """https://github.com/kuangliu/pytorch-cifar/issues/19"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])


class DataModule:
    """Module de gestion des données"""

    def __init__(self, config: Config):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

    def setup(self):
        """Initialise les datasets et dataloaders"""
        logger.info("Configuration des données...")

        # Création des datasets
        self.train_dataset = CIFAR10(
            root=self.config.DATA_DIR,
            train=True,
            download=True,
            transform=DataAugmentation.get_val_transforms()
        )

        self.val_dataset = CIFAR10(
            root=self.config.DATA_DIR,
            train=False,
            download=True,
            transform=DataAugmentation.get_val_transforms()
        )

        # Création des dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )

        logger.info(f"Dataset train: {len(self.train_dataset)} images")
        logger.info(f"Dataset validation: {len(self.val_dataset)} images")


