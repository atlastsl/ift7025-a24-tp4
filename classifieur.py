"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenir au moins les 3 méthodes definies ici bas, 
	* train 	: pour entraîner le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""
import time

import numpy as np
import pandas as pd


# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones

def evaluate_class(y, y_predicted, class_label):
    """
    Evaluate performance of model for a specific class
    Args:
        y: Vector of true classes
        y_predicted: Vector of predicted classes
        class_label: Class Label for evaluation
    Returns:
        List of:
        - Confusion Matrix
        - Accuracy
        - Precision
        - Recall
        - F1 Score
    """
    row_col_names = np.array(["!" + str(class_label), str(class_label)])
    yp_real_ng = y_predicted[y != class_label]
    yp_true_ng = len(yp_real_ng[yp_real_ng != class_label])
    yp_false_ps = len(yp_real_ng[yp_real_ng == class_label])
    yp_real_ps = y_predicted[y == class_label]
    yp_false_ng = len(yp_real_ps[yp_real_ps != class_label])
    yp_true_ps = len(yp_real_ps[yp_real_ps == class_label])
    np_confusion = np.array([[yp_true_ng, yp_false_ng], [yp_false_ps, yp_true_ps]], dtype=int)
    confusion = pd.DataFrame(np_confusion, index=row_col_names, columns=row_col_names)
    confusion.index.name = 'Predicted'
    confusion.columns.name = 'Real'
    accuracy = float((yp_true_ng + yp_true_ps) / (yp_true_ng + yp_false_ps + yp_false_ng + yp_true_ps))
    precision = float(yp_true_ps / (yp_true_ps + yp_false_ps)) if (yp_true_ps + yp_false_ps) > 0 else 0
    recall = float(yp_true_ps / (yp_true_ps + yp_false_ng)) if (yp_true_ps + yp_false_ng) > 0 else 0
    f1_score = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return dict(confusion=confusion, accuracy=accuracy,
                precision=precision, f1_score=f1_score, recall=recall)


def evaluate__confusion_matrix(y, y_predicted):
    """
    Compute confusion matrix for a set of predicted values
    Args:
        y: Vector of true classes
        y_predicted: Vector of predicted classes
    Returns:
        Confusion matrix
    """
    unique_yp = np.unique(y_predicted)
    unique_yt = np.unique(y)
    cross_occurrences = np.zeros((len(unique_yp), len(unique_yt)))
    for i, val1 in enumerate(unique_yp):
        for j, val2 in enumerate(unique_yt):
            cross_occurrences[i, j] = np.sum((y_predicted == val1) & (y == val2))
    df = pd.DataFrame(cross_occurrences, index=unique_yp, columns=unique_yt)
    df.index.name = 'Predicted'
    df.columns.name = 'Real'
    return df


def displayTime(nTime):
    vTime = nTime
    if vTime < 60:
        return f"{vTime:.3f} seconds"
    vTime = vTime / 60.0
    if vTime < 60:
        return f"{vTime:.3f} minutes"
    vTime = vTime / 60.0
    return f"{vTime:.3f} hours"


class Classifier:  #nom de la class à changer

    def __init__(self, **kwargs):
        """
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
        self.trainTime = 0

    def train(self, train, train_labels):  #vous pouvez rajouter d'autres attributs au besoin
        """
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""

    def predict(self, x):
        """
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""

    def evaluate(self, X, y):
        """
		c'est la méthode qui va évaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
        yp_list = []
        startTime = time.time()
        for i in range(X.shape[0]):
            yp = self.predict(X[i, :])
            yp_list.append(yp)
        evalTime = (time.time() - startTime) / float(X.shape[0])
        y_predicted = np.array(yp_list, dtype=int)
        accuracy = np.mean(y == y_predicted)
        error_rate = np.mean(y != y_predicted)
        confusion = evaluate__confusion_matrix(y, y_predicted)
        classes = dict()
        if np.unique(y).size == 2:
            positive_class = np.max(y)
            classes[str(positive_class)] = evaluate_class(y, y_predicted, positive_class)
        else:
            for v in np.unique(y):
                classes[str(v)] = evaluate_class(y, y_predicted, v)
        return dict(classes=classes, accuracy=accuracy, error_rate=error_rate, confusion=confusion,
                    train_time=displayTime(self.trainTime), eval_time=displayTime(evalTime))

# Vous pouvez rajouter d'autres méthodes et fonctions,
# il suffit juste de les commenter.
