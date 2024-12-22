# Liste des classes

## classifieur.py
Ce fichier contient la classe classifieur, de laquelle héritent les classes DecisionTree et NeuralNetwork.
Elle contient aussi l'implémentation de la fonction evaluate, qui permet d'évaluer un modèle de classement
sur un ensemble de données.

## DecisionTree.py
Ce fichier contient l'implémentation de l'algorithme de l'arbre de décision, à savoir l'entrainement et la
prédiction de la classe d'une nouvelle observation. Les classes qu'il contient sont :
- DecisionTree: qui représente l'arbre de décision global
- TreeNode: qui représente un noeud de l'arbre
Malheureusement, j'ai manqué de temps pour commenter l'intégralité du code, pour faciliter sa compréhension.

## NeuralNetwork.py
Ce fichier contient l'implémentation de l'algorithme du réseau de neurone, à savoir l'entrainement
et la prédiction de la classe d'une nouvelle observation. Les classes qu'il contient sont :
- NeuralNetwork: qui représente le réseau de neurone global
- Neuron: qui représente un neurone dans le réseau
Malheureusement, j'ai manqué de temps pour commenter l'intégralité du code, pour faciliter sa compréhension.


# Autres fichiers et repertoires

## Arbres de décision
Le répertoire "trees" contient les descriptions visuelles des arbres de décision, inspirées de ce que peut ressortir
les libraires d'algorithmes d'apprentissage à l'exemple de Scikit-learn en Python.

## Graphiques
Les graphiques ont été générés par la libraire matplotlib. Il sont tous stockés dans le répertoire "images".


# Libraires utilisées

- numpy : Utilisé pour les calculs numériques.
- pandas : Utilisé uniquement pour un enrobage de certains tableaux et matrices de métadonnées utiles à la compréhension
lors des affichages.
- multiprocess : Utilisé pour l'exécution de tâches en parallèle.
- time : Utilisé pour mesurer le temps d'apprentissage et le temps de prédiction.
- matplotlib : Utilisé pour construire des graphiques.


# Difficultés rencontrées

## Arbres de décision
- Construction de la courbe d'apprentissage : Pour la construction de la courbe d'apprentissage, j'étais parti sur 10
essais (10 apprentissages) pour chaque fraction des données d'entrainement sanity-check. Mais l'apprentissage étant
très chronophage pour les datasets WINE et ABALONE, j'ai du restreindre le nombre d'essais à 3 pour WINE et 1 pour
ABALONE. Sinon j'aurais manqué de temps pour terminer le TP.

## Réseau de neurone
- Validations croisée pour le nombre de périodes d'entrainement, la dimension et la profondeur: L'apprentissage étant
aussi chronophage pour les réseaux de neurone, j'ai dû mettre un place l'exécution en parallèle pour chaque valeur à
tester lors des validations croisées. Sinon, j'aurais manqué de temps pour terminer le TP.
- Hyperparamètres: J'ai aussi un peu galéré à trouver les bonnes valeurs d'hyperparamètres alpha et activation. Surtout
concernant le dataset WINE. Aussi, j'ai finalement opté pour alpha = 0.05 pour IRIS et ABALONE et alpha = 0.001 pour
WINE.
- Algorithme d'optimisation: L'utilisation de l'activation softmax en sortie de réseau étant conseillée pour les
problèmes de classement à plus de deux classes, j'ai tenté de trouver analytiquement la formule de mise à jour par
descente de gradient stochastique mais je n'aurais pas eu le temps de terminer et d'implémenter. Finalement, je
suis resté sur la formule analytique dérivée de l'utilisation de l'activation sigmoide en sortie de réseau. Et elle a
étonnement produit la plupart du temps de meilleurs résultats que l'activation sigmoide lors de mes expérimentations.


# Test / Lancement du fichier entrainer_tester.py
Etant donné que l'apprentissage des algorithmes est chronophages, il est RECOMMANDE de tester chaque section
(ou partie) du fichier entrainer_tester.py une à une, en enveloppant les autres par des # de commentaires. Nous
laissons toutes les sections activées (non commentées).


# Repartition du travail

- Aurélien Simo (100%)

