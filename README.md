# README - Installation des dépendances

Ce projet utilise Python et plusieurs bibliothèques pour l'apprentissage automatique et le traitement des données.

## Prérequis
- Python 3.8 ou plus récent
- pip (gestionnaire de paquets Python)

## Installation rapide

1. **Cloner le projet**

```bash
git clone <url-du-repo>
cd PythonProject
```

2. **Installer les dépendances**

Créez un environnement virtuel (optionnel mais recommandé) :

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

Installez les paquets nécessaires :

```bash
pip install -r requirements.txt
```

## Fichier requirements.txt

Si le fichier n'existe pas, créez-le avec le contenu suivant :

```
torch
numpy
scikit-learn
```

## Lancement du projet

Exemple pour lancer l'entraînement :

```bash
python cifar10_train.py
```

## Remarques
- Adaptez les commandes selon votre OS.
- Pour toute question, consultez les fichiers sources ou contactez le responsable du projet.

