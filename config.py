class Config:
    """Configuration centralisée du projet"""
    # Données
    DATA_DIR = "./data"
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    # Modèle
    NUM_CLASSES = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.5
    # Entraînement
    NUM_EPOCHS = 1
    PATIENCE = 10
    # Logging
    LOG_DIR = "./runs"
    SAVE_DIR = "./models"
    # Classes CIFAR-10
    CLASS_NAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

