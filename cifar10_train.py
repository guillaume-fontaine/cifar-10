import torch
import logging
from config import Config
from data import DataModule
from model import CIFAR10Model
from trainer import Trainer

# Configuration et logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Utilisation du device: {device}")
    data_module = DataModule(config)
    data_module.setup()
    model = CIFAR10Model(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    )
    trainer = Trainer(model, config, device)
    trainer.train(data_module.train_loader, data_module.val_loader)

if __name__ == "__main__":
    main()
