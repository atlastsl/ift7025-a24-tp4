import math

import numpy as np
import random


def cv_validation_folds(data_len=1, nb_folds=10):
    """
    Cette fonction permet de generer un ordre aleatoire d'indices pour la separation des donnees en ensemble de
    validation pour la validation croisee
    Args:
        data_len: Nombre d'observations/exemples dans le dataset de base
        nb_folds: Nombre d'ensemble de validation pour la validation croisee

    Returns:
        validations: la liste de tableau d'indices pour la validation croisee

    """
    random.seed(1)
    shuffled_index = random.sample(range(data_len), data_len)
    number_splits = min(nb_folds, len(shuffled_index))
    splits_len = math.floor(len(shuffled_index) / number_splits)

    folds = []
    for i in range(number_splits):
        start = i * splits_len
        end = start + splits_len
        if i == number_splits - 1:
            end = data_len
        folds.append(shuffled_index[start:end])

    return folds


def validation_set(data_len=1, split_ratio=0.7):
    random.seed(1)
    train_index = random.sample(range(data_len), int(split_ratio * data_len))
    test_index = list(set(range(data_len)) - set(train_index))
    return train_index, test_index


def learning_curve_training_sets(train_index, nb_trials=20):
    training_sets = dict()
    for i in range(1, 100, 1):
        take_ratio = float(i) / 100.0
        training_sets_i = []
        for trial in range(nb_trials):
            random.seed(None)
            train_ix = random.sample(range(len(train_index)), int(take_ratio * len(train_index)))
            if len(train_ix) > 0:
                training_sets_i.append(train_ix)
        training_sets[i] = training_sets_i
    return training_sets


def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - trainX : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque
        ligne dans cette matrice représente un exemple d'entrainement.

        - trainY : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - testX : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque
        ligne dans cette matrice représente un exemple de test.

        - testY : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """

    random.seed(1)  # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    # Le fichier du dataset est dans le dossier datasets en attaché
    f = open('datasets/bezdekIris.data', 'r')
    data = [line.strip().split(',') for line in f.readlines() if line.strip()]

    X = np.array([row[:-1] for row in data], dtype=float)
    Y = np.array([conversion_labels[row[-1]] for row in data], dtype=int)

    train_index = random.sample(range(len(data)), int(train_ratio * len(data)))
    test_index = list(set(range(len(data))) - set(train_index))

    trainX = X[train_index, :]
    trainY = Y[train_index]
    testX = X[test_index, :]
    testY = Y[test_index]

    # TODO : le code ici pour lire le dataset

    # REMARQUE très importante :
    # remarquez bien comment les exemples sont ordonnés dans
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.

    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    return trainX, trainY, testX, testY


def load_wine_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - trainX : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque
        ligne dans cette matrice représente un exemple d'entrainement.

        - trainY : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - testX : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque
        ligne dans cette matrice représente un exemple de test.

        - testY : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """

    random.seed(1)  # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché
    f = open('datasets/binary-winequality-white.csv', 'r')
    data = [line.strip().split(',') for line in f.readlines() if line.strip()]

    X = np.array([row[:-1] for row in data], dtype=float)
    Y = np.array([row[-1] for row in data], dtype=int)

    train_index = random.sample(range(len(data)), int(train_ratio * len(data)))
    test_index = list(set(range(len(data))) - set(train_index))

    trainX = X[train_index, :]
    trainY = Y[train_index]
    testX = X[test_index, :]
    testY = Y[test_index]

    # TODO : le code ici pour lire le dataset

    # La fonction doit retourner 4 structures de données de type Numpy.
    return trainX, trainY, testX, testY


def load_abalone_dataset(train_ratio):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - trainX : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque
        ligne dans cette matrice représente un exemple d'entrainement.

        - trainY : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - testX : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque
        ligne dans cette matrice représente un exemple de test.

        - testY : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    random.seed(1)
    f = open('datasets/abalone-intervalles.csv', 'r')
    data = [line.strip().split(',') for line in f.readlines() if line.strip()]

    gender_col = [row[0] for row in data]
    gender_col_mapping = {label: idx for idx, label in enumerate(set(gender_col))}
    gender_col_num = [gender_col_mapping[g] for g in gender_col]

    gender_vec = np.array(gender_col_num, dtype=float)
    X = np.array([row[1:-1] for row in data], dtype=float)
    X = np.hstack((np.array(gender_vec).reshape(-1, 1), X))
    Y = np.array([row[-1] for row in data], dtype=float)
    Y = Y.astype(int)

    train_index = random.sample(range(len(data)), int(train_ratio * len(data)))
    test_index = list(set(range(len(data))) - set(train_index))

    trainX = X[train_index, :]
    trainY = Y[train_index]
    testX = X[test_index, :]
    testY = Y[test_index]

    return trainX, trainY, testX, testY
