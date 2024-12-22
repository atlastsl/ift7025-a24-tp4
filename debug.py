import numpy as np

import load_datasets
from DecisionTree import DecisionTree, decisionTreeScikitLearn, computeImpurity
from entrainer_tester import display_performances, decisionTreeLC
from NeuralNetwork import NeuralNetwork, NNActivationSigmoid, NNActivationSoftmax
import matplotlib.pyplot as plt

## Parametres
train_ratio = 0.7
validation_split_ratio = 0.7
learning_curve_nb_trials = 20
cross_validation_folds = 10
pruning_rate = 0.05

## Charger/lire les datasets
iris_data = load_datasets.load_iris_dataset(train_ratio)
iris_train_X = iris_data[0]
iris_train_Y = iris_data[1]
iris_test_X = iris_data[2]
iris_test_Y = iris_data[3]
wine_data = load_datasets.load_wine_dataset(train_ratio)
wine_train_X = wine_data[0]
wine_train_Y = wine_data[1]
wine_test_X = wine_data[2]
wine_test_Y = wine_data[3]
abalone_data = load_datasets.load_abalone_dataset(train_ratio)
abalone_train_X = abalone_data[0]
abalone_train_Y = abalone_data[1]
abalone_test_X = abalone_data[2]
abalone_test_Y = abalone_data[3]

## 1.1. Instanciation des parametres
dcSplitter = "entropy"
dcStopper = "min-pop"
dcStopperV = 5
dcDiscretizer = "switch-class"
dcDiscretizerV = 1
dcTree = DecisionTree(splitter=dcSplitter, stopper=dcStopper, stopperV=dcStopperV, discretizer=dcDiscretizer,
                      discretizerV=dcDiscretizerV)

nnDim = wine_train_X.shape[1]
nnDepth = 1
hdnActivation = NNActivationSigmoid
outActivation = NNActivationSigmoid
# outActivation = NNActivationSoftmax
alpha = 0.001
nbEpoch = 100
nnInstance = NeuralNetwork(dimension=nnDim, depth=nnDepth, hdnActivation=hdnActivation, outActivation=outActivation,
                           alpha=alpha, nbEpoch=nbEpoch, trace=True)


def debugDC():
    dcTree.train(iris_train_X, iris_train_Y)
    p = dcTree.evaluate(iris_test_X, iris_test_Y)
    p['sk_accuracy'] = decisionTreeScikitLearn((iris_train_X, iris_train_Y, iris_test_X, iris_test_Y))
    display_performances(p, "Debug Iris")
    dcTree.display()
    computeImpurity(iris_train_Y, "entropy")


def learningCurve():
    decisionTreeLC(train_X=iris_train_X, train_Y=iris_train_Y, datasetName="Iris")


def debugDCPrune():
    dcTree.train(iris_train_X, iris_train_Y)
    #dcTree.prune(rate=0.05)


def debugNN():
    nnInstance.train(wine_train_X, wine_train_Y)
    p = nnInstance.evaluate(wine_test_X, wine_test_Y)
    display_performances(p, "Debug Iris")


def debugNNSelectDimension():
    dimensions = range(1, 10)
    d_perf = np.zeros(len(dimensions))
    for i in range(len(dimensions)):
        dimension = dimensions[i]
        nn = NeuralNetwork(dimension=dimension, depth=nnDepth, hdnActivation=hdnActivation, outActivation=outActivation,
                           alpha=alpha, nbEpoch=nbEpoch, trace=False)
        folds = load_datasets.cv_validation_folds(len(iris_train_X))
        perf = []
        for fold in folds:
            test_index = fold
            train_index = list(set(range(iris_train_X.shape[0])) - set(test_index))
            X = iris_train_X[train_index]
            Y = iris_train_Y[train_index]
            Xt = iris_train_X[test_index]
            Yt = iris_train_Y[test_index]
            nn.train(X, Y)
            p = nn.evaluate(Xt, Yt)
            perf.append(p['accuracy'])
        d_perf[i] = np.array(perf).mean()
        print("Completed Dimension " + str(dimension))
    plt.plot(np.array(dimensions), d_perf)
    plt.xlabel('Dimensions')
    plt.ylabel('Accurary with 10-CV')
    plt.title('Neural Network dimension curve')
    plt.show()


def debugNNSelectDepth():
    depths = range(1, 6)
    slDim = 2
    d_perf = np.zeros(len(depths))
    for i in range(len(depths)):
        depth = depths[i]
        nn = NeuralNetwork(dimension=slDim, depth=depth, hdnActivation=hdnActivation, outActivation=outActivation,
                           alpha=alpha, nbEpoch=nbEpoch, trace=False)
        folds = load_datasets.cv_validation_folds(len(iris_train_X))
        perf = []
        for fold in folds:
            test_index = fold
            train_index = list(set(range(iris_train_X.shape[0])) - set(test_index))
            X = iris_train_X[train_index]
            Y = iris_train_Y[train_index]
            Xt = iris_train_X[test_index]
            Yt = iris_train_Y[test_index]
            nn.train(X, Y)
            p = nn.evaluate(Xt, Yt)
            perf.append(p['accuracy'])
        d_perf[i] = np.array(perf).mean()
        print("Completed Depth " + str(depth))
    plt.plot(np.array(depths), d_perf)
    plt.xlabel('Depths')
    plt.ylabel('Accurary with 10-CV')
    plt.title('Neural Network dimension curve')
    plt.show()

def debugMain():
    debugNN()


if __name__ == "__main__":
    debugMain()
