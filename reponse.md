# TP : Analyse du dataset CIFAR-10 et augmentation de données

## Question 1 : Quelles sont les informations que vous pouvez extraire du dataset CIFAR-10 ?

CIFAR-10 est un dataset de référence qui contient 60 000 images couleur de 32×32 pixels réparties en 10 classes. Il y a 50 000 images pour l'entraînement et 10 000 pour les tests, avec 6 000 images par classe dans le set d'entraînement. Les classes incluent des objets courants comme airplane, automobile, bird, cat, deer, dog, frog, horse, ship et truck. La petite taille des images (32×32 pixels) est un défi majeur car elle limite les détails visibles, ce qui oblige les modèles à identifier les caractéristiques essentielles de chaque classe.

## Question 2 : Quels sont les biais potentiels présents dans ce dataset ?

CIFAR-10 présente plusieurs biais qui peuvent affecter les performances des modèles. Le biais de représentation est important car certaines classes peuvent manquer de diversité visuelle ou avoir des angles de vue similaires. Il y a aussi un biais temporel puisque les images ont été collectées à une époque donnée. Les automobiles et avions peuvent refléter des styles spécifiques à certaines régions, créant un biais géographique. La faible résolution introduit un biais technique majeur car certaines classes restent plus facilement reconnaissables que d'autres à 32×32 pixels. Enfin, la qualité variable des images (netteté, éclairage, arrière-plan) peut créer des déséquilibres dans l'apprentissage.

## Question 3 : Quelles augmentations de données seraient intéressantes pour ce dataset ?

Les transformations géométriques sont essentielles pour CIFAR-10. La rotation d'images entre ±15° et ±30° simule différents angles de vue sans déformer les objets. Le flip horizontal fonctionne bien pour les classes symétriques comme les avions ou voitures. Le crop aléatoire consiste à découper des zones 32×32 dans des images légèrement agrandies, simulant des variations de position. Pour les transformations de couleur, on peut modifier la luminosité et le contraste de ±20 à 30% pour simuler différents éclairages.

## Question 4 : Comment intégreriez-vous ces augmentations dans le processus d'entraînement ?

Les augmentations doivent être appliquées uniquement pendant l'entraînement, pas pendant la validation ou les tests. On utilise généralement une composition de transformations avec des probabilités contrôlées. L'ordre typique commence par le crop aléatoire avec padding, puis le flip horizontal avec 50% de probabilité, suivi des ajustements de couleur modérés. La normalisation des pixels se fait en dernier. Cette approche "à la volée" calcule les transformations pendant l'entraînement, évitant de stocker des images supplémentaires. On peut aussi ajuster progressivement l'intensité des augmentations selon l'avancement de l'entraînement pour optimiser les résultats.

## Question 5 : Quels sont les avantages de cette intégration d'augmentations ?

L'augmentation de données réduit le surapprentissage car le modèle voit plus de variantes des images d'origine, l'empêchant de mémoriser des détails spécifiques. Cela améliore la généralisation et donne généralement un gain d'accuracy de 2 à 5% sur CIFAR-10. Le modèle devient plus robuste face aux variations naturelles des vraies données. L'augmentation virtuelle du dataset simule un ensemble d'entraînement plus large sans utiliser d'espace de stockage supplémentaire. L'entraînement devient aussi plus stable avec moins de variance entre différents runs. Les transformations sont calculées en parallèle pendant le chargement des données, donc l'impact sur les temps d'entraînement est minimal. Le modèle apprend finalement à reconnaître les objets indépendamment de leur orientation ou éclairage.

# Source
- https://github.com/moritzhambach/Image-Augmentation-in-Keras-CIFAR-10-
- https://www.researchgate.net/figure/Accuracy-results-for-Mixup-CutMix-and-AttentiveCutMix-on-CIFAR-10-reported-in-the-work_fig14_361653978
- https://deeplearning.neuromatch.io/projects/ComputerVision/data_augmentation.html
- https://github.com/chhayac/Machine-Learning-Notebooks/blob/master/Recognizing-CIFAR-10-images-Improved-Model-Data-Augmentation.ipynb