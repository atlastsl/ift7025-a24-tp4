import numpy as np
import load_datasets
from DecisionTree import DecisionTree, decisionTreeScikitLearn
from NeuralNetwork import NeuralNetwork, NNActivationSigmoid, NNActivationSoftmax
import matplotlib.pyplot as plt


#import NeuralNet# importer la classe du Knn
#importer d'autres fichiers et classes si vous en avez développés


def display_performances(performances, title='', display_classes_perf=True):
    print()
    print(title)
    print('------------------------------------------------')
    print(f'--- Training Time: {performances['train_time']}')
    print(f'--- 1Ex Eval Time: {performances['eval_time']}')
    print(f'--- Error rate: {performances['error_rate']:.6f}')
    print(f'--- Accuracy: {performances['accuracy']:.6f}')
    if 'sk_accuracy' in performances.keys():
        print(f'--- ScikitLearn Acc: {performances['sk_accuracy']:.6f}')
    print('--- Confusion Matrix:')
    print(performances['confusion'])
    if display_classes_perf:
        for key in performances['classes'].keys():
            print('-----------------------')
            print('--- Class ' + str(key) + ':')
            print(f'----- Accuracy: {performances['classes'][key]['accuracy']:.6f}')
            print(f'----- Precision: {performances['classes'][key]['precision']:.6f}')
            print(f'----- Recall: {performances['classes'][key]['recall']:.6f}')
            print(f'----- F1-score: {performances['classes'][key]['f1_score']:.6f}')
            #print('----- Confusion Matrix:')
            #print(performances['classes'][key]['confusion'])
    print('------------------------------------------------')
    print()


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entraîner votre classifieur
4- Le tester
"""

# 0. DATASETS

## Parametres
train_ratio = 0.7
validation_split_ratio = 0.7
learning_curve_nb_trials = 5
cross_validation_folds = 1
pruning_rate = 0.05

## Charger/lire les datasets
iris_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
iris_data = load_datasets.load_iris_dataset(train_ratio)
iris_train_X = iris_data[0]
iris_train_Y = iris_data[1]
iris_test_X = iris_data[2]
iris_test_Y = iris_data[3]
wine_columns = ['AcFixe', 'AcVol', 'AcCit', 'AcRes', 'ChloSod', 'DixSfrLib', 'DixSfrTot', 'Density', 'pH', 'SlfPot',
                'Alcohol']
wine_data = load_datasets.load_wine_dataset(train_ratio)
wine_train_X = wine_data[0]
wine_train_Y = wine_data[1]
wine_test_X = wine_data[2]
wine_test_Y = wine_data[3]
abalone_columns = ['Sexe', 'LgCoq', 'DmCoq', 'Hauteur', 'PoidsTot', 'PoidsChr', 'PoidsVis', 'PoidsCoq']
abalone_data = load_datasets.load_abalone_dataset(train_ratio)
abalone_train_X = abalone_data[0]
abalone_train_Y = abalone_data[1]
abalone_test_X = abalone_data[2]
abalone_test_Y = abalone_data[3]

# 1. DECISION TREE

## 1.1. Initialisation

## 1.1.1. Paramètres
dcSplitter = "entropy"
dcStopper = "min-pop"
dcStopperV = 5
dcDiscretizer = "switch-class"
dcDiscretizerV = 1
dcTree = DecisionTree(splitter=dcSplitter, stopper=dcStopper, stopperV=dcStopperV, discretizer=dcDiscretizer,
                      discretizerV=dcDiscretizerV)


## 1.1.2. Fonctions utilitaires
## 1.1.2.1. Courbe d'apprentissage
def decisionTreeLC(train_X, train_Y, datasetName="", pruning=False):
    val_ind = load_datasets.validation_set(len(train_X), validation_split_ratio)
    val_X = train_X[val_ind[1], :]
    val_Y = train_Y[val_ind[1]]
    lc_train = load_datasets.learning_curve_training_sets(val_ind[0], learning_curve_nb_trials)
    curveY = []
    curveX = []
    for i in lc_train:
        perfs = []
        if len(lc_train[i]) > 0:
            for trial_train in lc_train[i]:
                v_train_X = train_X[trial_train, :]
                v_train_Y = train_Y[trial_train]
                dcTree.train(v_train_X, v_train_Y)
                if pruning:
                    dcTree.prune(rate=pruning_rate)
                ev = dcTree.evaluate(val_X, val_Y)
                perfs.append(ev['accuracy'])
        else:
            perfs.append(0)
        curveY.append(np.array(perfs).mean())
        curveX.append(i)
        print(f"Learning curve done step {i}...")
    plt.plot(np.array(curveX), np.array(curveY))
    plt.xlabel('Training set size (%)')
    plt.ylabel('Accurary on validation set')
    plt.title('Decision Tree - Learning curve for ' + datasetName)
    plt.show()


## 1.1.2.1. Entrainement et test
def decisionTreeTrnTst(train_X, train_Y, test_X, test_Y, datasetName="", varNames=None, pruning=False):
    dcTree.train(train_X, train_Y)
    file = f"trees/{datasetName}.txt"
    if pruning:
        dcTree.prune(rate=pruning_rate)
        file = f"trees/{datasetName}_pruned.txt"
    performances = dcTree.evaluate(test_X, test_Y)
    sk_accuracy = decisionTreeScikitLearn((train_X, train_Y, test_X, test_Y))
    performances['sk_accuracy'] = sk_accuracy
    display_performances(performances, title='[Decision Tree] ' + datasetName + ' Dataset', display_classes_perf=True)
    dcTree.display(varNames=varNames, file=file)


# print("DECISION TREE")
# print("--------------------------------------------------------------------------------------------------------------")

## 1.2. Iris Dataset

# print("")
# print("Iris Dataset")
# print("")
# ### 1.2.1. Courbe d'apprentissage
# print("Iris Dataset - Learning Curve")
# print("")
# decisionTreeLC(iris_train_X, iris_train_Y, "IRIS")
# ### 1.2.2. Entrainement et test
# print("Iris Dataset - Training & Test")
# print("")
# decisionTreeTrnTst(iris_train_X, iris_train_Y, iris_test_X, iris_test_Y, "IRIS", iris_columns)
# ### 1.2.3. Courbe d'apprentissage [Avec Elagage]
# print("Iris Dataset - Pruned Tree - Learning Curve")
# print("")
# decisionTreeLC(iris_train_X, iris_train_Y, "IRIS", pruning=True)
# ### 1.2.4. Entrainement et test [Avec Elagage]
# print("Iris Dataset - Pruned Tree - Training & Test")
# print("")
# decisionTreeTrnTst(iris_train_X, iris_train_Y, iris_test_X, iris_test_Y, "IRIS", iris_columns, pruning=True)


## 1.3. Wine Dataset

# print("")
# print("Wine Dataset")
# print("")
### 1.3.1. Courbe d'apprentissage
# print("Wine Dataset - Learning Curve")
# print("")
# decisionTreeLC(wine_train_X, wine_train_Y, "WINE")
# ### 1.3.2. Entrainement et test
# print("Wine Dataset - Training & Test")
# print("")
# decisionTreeTrnTst(wine_train_X, wine_train_Y, wine_test_X, wine_test_Y, "WINE", wine_columns)
### 1.3.3. Courbe d'apprentissage [Avec Elagage]
# print("Wine Dataset - Pruned Tree - Learning Curve")
# print("")
# decisionTreeLC(wine_train_X, wine_train_Y, "WINE", pruning=True)
# ### 1.3.4. Entrainement et test [Avec Elagage]
# print("Wine Dataset - Pruned Tree - Training & Test")
# print("")
# decisionTreeTrnTst(wine_train_X, wine_train_Y, wine_test_X, wine_test_Y, "WINE", wine_columns, pruning=True)


## 1.4. Abalone Dataset

# print("")
# print("Abalone Dataset")
# print("")
# ## 1.4.1. Courbe d'apprentissage
# print("Abalone Dataset - Learning Curve")
# print("")
# decisionTreeLC(abalone_train_X, abalone_train_Y, "ABALONE")
# ## 1.4.2. Entrainement et test
# print("Abalone Dataset - Training & Test")
# print("")
# decisionTreeTrnTst(abalone_train_X, abalone_train_Y, abalone_test_X, abalone_test_Y, "ABALONE",
#                    abalone_columns)
## 1.4.3. Courbe d'apprentissage [Avec Elagage]
# print("Abalone Dataset - Pruned Tree - Learning Curve")
# print("")
# decisionTreeLC(abalone_train_X, abalone_train_Y, "ABALONE", pruning=True)
# ## 1.4.4. Entrainement et test [Avec Elagage]
# print("Abalone Dataset - Pruned Tree - Training & Test")
# print("")
# decisionTreeTrnTst(abalone_train_X, abalone_train_Y, abalone_test_X, abalone_test_Y, "ABALONE",
#                    abalone_columns, pruning=True)


# 2. RESEAU DEN NEURONES

## 2.1. Initialisation

### 2.1.1. Paramètres
nnDepth1 = 1
nnAlpha = 0.05
colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']


### 2.1.2. Fonctions
def nnTrainTest(trainX, trainY, testX, testY, dimension, depth, hdnAct, outAct, alpha, nbEpochs, trace, datasetName):
    nn = NeuralNetwork(dimension=dimension, depth=depth, hdnActivation=hdnAct, outActivation=outAct,
                       alpha=alpha, nbEpoch=nbEpochs, trace=trace)
    nn.train(trainX, trainY)
    p = nn.evaluate(testX, testY)
    display_performances(p, title='[Neural Network] ' + datasetName + ' Dataset', display_classes_perf=True)


def nnCrossValidationStep(nn, trainX, trainY, eval_epoch_perfs=False):
    folds = load_datasets.cv_validation_folds(len(trainX))
    perf = []
    epochPerfs = []
    for fold in folds:
        test_index = fold
        train_index = list(set(range(trainX.shape[0])) - set(test_index))
        X = trainX[train_index]
        Y = trainY[train_index]
        Xt = trainX[test_index]
        Yt = trainY[test_index]
        nn.train(X, Y)
        p = nn.evaluate(Xt, Yt)
        if eval_epoch_perfs:
            epochPerfs.append(nn.epochPerfs)
        perf.append(p['accuracy'])
    if eval_epoch_perfs:
        mm = len(epochPerfs[0])
        epochPerfs = [np.array([ep[i] for ep in epochPerfs]).mean() for i in range(mm)]
    return np.array(perf).mean(), epochPerfs


def nnCvNbEpoch(trainX, trainY, dimension, depth, hdnAct, outAct, alpha, nbEpochsRange, trace, datasetName):
    d_perf = np.zeros(len(nbEpochsRange))
    for i in range(len(nbEpochsRange)):
        nbEpochs = nbEpochsRange[i]
        nn = NeuralNetwork(dimension=dimension, depth=depth, hdnActivation=hdnAct, outActivation=outAct,
                           alpha=alpha, nbEpoch=nbEpochs, trace=trace)
        (perf_i, _) = nnCrossValidationStep(nn, trainX, trainY)
        d_perf[i] = perf_i
        print(">> ------ Completed Nb Epoch " + str(nbEpochs), flush=True)
    plt.plot(np.array(nbEpochsRange), d_perf)
    plt.xlabel('Nb Epochs')
    plt.ylabel('Accurary with 10-CV')
    plt.title(f'Neural Network Nb Epochs Selection for {datasetName} Dataset')
    plt.savefig(f'images/nn_{datasetName.lower()}_sel_ep.png')


def nnCvDimension(trainX, trainY, dimRange, depth, hdnAct, outAct, alpha, nbEpochs, trace, datasetName):
    d_perf = np.zeros(len(dimRange))
    for i in range(len(dimRange)):
        dimension = dimRange[i]
        nn = NeuralNetwork(dimension=dimension, depth=depth, hdnActivation=hdnAct, outActivation=outAct,
                           alpha=alpha, nbEpoch=nbEpochs, trace=trace)
        (perf_i, _) = nnCrossValidationStep(nn, trainX, trainY)
        d_perf[i] = perf_i
        print(">> ------ Completed Dimension " + str(dimension), flush=True)
    plt.plot(np.array(dimRange), d_perf)
    plt.xlabel('Dimensions')
    plt.ylabel('Accurary with 10-CV')
    plt.title(f'Neural Network Dimension Selection for {datasetName} Dataset')
    plt.savefig(f'images/nn_{datasetName.lower()}_sel_di.png')


def nnCvDepth(trainX, trainY, dimension, depthRange, hdnAct, outAct, alpha, nbEpochs, trace, datasetName):
    d_perf = np.zeros(len(depthRange))
    epoch_perfs = []
    for i in range(len(depthRange)):
        depth = depthRange[i]
        nn = NeuralNetwork(dimension=dimension, depth=depth, hdnActivation=hdnAct, outActivation=outAct,
                           alpha=alpha, nbEpoch=nbEpochs, trace=trace)
        (perf_i, epp) = nnCrossValidationStep(nn, trainX, trainY, eval_epoch_perfs=True)
        d_perf[i] = perf_i
        epoch_perfs.append(epp)
        print(">> ------ Completed Depth " + str(depth), flush=True)
    plt.plot(np.array(depthRange), d_perf)
    plt.xlabel('Depths')
    plt.ylabel('Accurary with 10-CV')
    plt.title(f'Neural Network Depth Selection for {datasetName} Dataset')
    plt.savefig(f'images/nn_{datasetName.lower()}_sel_de.png')
    for i in range(len(depthRange)):
        plt.plot(np.array(range(1, len(epoch_perfs[i])+1)), np.array(epoch_perfs[i]), color=colors[i],
                 label=f'Depth {depthRange[i]}')
    plt.xlabel('Learning epoch')
    plt.ylabel('Accurary with 10-CV')
    plt.title(f'Accuracy with learning epoch for {datasetName} Dataset')
    plt.legend()
    plt.savefig(f'images/nn_{datasetName.lower()}_vgr.png')


print("NEURAL NETWORK", flush=True)
print("--------------------------------------------------------------------------------------------------------------", flush=True)

# ## 2.2. Iris Dataset
# nnIrisHdnAct = NNActivationSigmoid
# nnIrisOutAct = NNActivationSoftmax
# print("")
# print("IRIS Dataset")
# print("")
#
# nnIrisNbEpoch = 400
# nnIrisDimension = 2
# nnIrisDepth = 1
# ### 2.2.1. Test arbitraire
# print("IRIS Dataset - Test Arbitraire")
# print("")
# nnTrainTest(iris_train_X, iris_train_Y, iris_test_X, iris_test_Y, iris_train_X.shape[1], 1, nnIrisHdnAct,
#             nnIrisOutAct, nnAlpha, 100, True, "IRIS")
# ### 2.2.2. Selection du nombre d'epoques par CV
# print("IRIS Dataset - Selection Nb Epoques par CV")
# print("")
# nnCvNbEpoch(iris_train_X, iris_train_Y, 4, 1, nnIrisHdnAct, nnIrisOutAct, nnAlpha,
#             [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], False, "IRIS")
# ### 2.2.3. Selection de la dimension par CV
# print("IRIS Dataset - Selection Dimension par CV")
# print("")
# nnCvDimension(iris_train_X, iris_train_Y, range(1, 8, 1), 1, nnIrisHdnAct, nnIrisOutAct, nnAlpha,
#               nnIrisNbEpoch, False, "IRIS")
# ### 2.2.4. Selection de la profondeur par CV
# print("IRIS Dataset - Selection Profondeur par CV")
# print("")
# nnCvDepth(iris_train_X, iris_train_Y, nnIrisDimension, range(1, 6, 1), nnIrisHdnAct, nnIrisOutAct, nnAlpha,
#           nnIrisNbEpoch, False, "IRIS")
# ### 2.2.5. Entrainement et test avec les hyperparametres finaux
# print("IRIS Dataset - Test HyperParam sélectionnés")
# print("")
# nnTrainTest(iris_train_X, iris_train_Y, iris_test_X, iris_test_Y, nnIrisDimension, nnIrisDepth, nnIrisHdnAct,
#             nnIrisOutAct, nnAlpha, nnIrisNbEpoch, True, "IRIS")



## 2.3. WINE Dataset
nnWineHdnAct = NNActivationSigmoid
nnWineOutAct = NNActivationSigmoid
print("", flush=True)
print("WINE Dataset", flush=True)
print("", flush=True)

nnWineNbEpoch = 100
nnWineDimension = 4
nnWineDepth = 1
nnWineAlpha = 0.001
# ### 2.3.1. Test arbitraire
# print("WINE Dataset - Test Arbitraire", flush=True)
# print("", flush=True)
# nnTrainTest(wine_train_X, wine_train_Y, wine_test_X, wine_test_Y, wine_train_X.shape[1], 1, nnWineHdnAct,
#             nnWineOutAct, nnWineAlpha, 100, True, "WINE")
### 2.3.2. Selection du nombre d'epoques par CV
print("WINE Dataset - Selection Nb Epoques par CV", flush=True)
print("", flush=True)
nnCvNbEpoch(wine_train_X, wine_train_Y, 4, 1, nnWineHdnAct, nnWineOutAct, nnWineAlpha,
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], False, "WINE")
# ### 2.3.3. Selection de la dimension par CV
# print("WINE Dataset - Selection Dimension par CV", flush=True)
# print("", flush=True)
# nnCvDimension(wine_train_X, wine_train_Y, range(1, 8, 1), 1, nnWineHdnAct, nnWineOutAct, nnWineAlpha,
#               nnWineNbEpoch, False, "WINE")
# ### 2.3.4. Selection de la profondeur par CV
# print("WINE Dataset - Selection Profondeur par CV", flush=True)
# print("", flush=True)
# nnCvDepth(wine_train_X, wine_train_Y, nnWineDimension, range(1, 6, 1), nnWineHdnAct, nnWineOutAct, nnWineAlpha,
#           nnWineNbEpoch, False, "WINE")
# ### 2.3.5. Entrainement et test avec les hyperparametres finaux
# print("WINE Dataset - Test HyperParam sélectionnés", flush=True)
# print("", flush=True)
# nnTrainTest(wine_train_X, wine_train_Y, iris_test_X, iris_test_Y, nnWineDimension, nnWineDepth, nnWineHdnAct,
#             nnWineOutAct, nnWineAlpha, nnWineNbEpoch, True, "WINE")




## 2.4. ABALONE Dataset
nnAbaloneHdnAct = NNActivationSigmoid
nnAbaloneOutAct = NNActivationSoftmax
print("", flush=True)
print("ABALONE Dataset", flush=True)
print("", flush=True)

nnAbaloneNbEpoch = 100
nnAbaloneDimension = 4
nnAbaloneDepth = 1
### 2.4.1. Test arbitraire
# print("ABALONE Dataset - Test Arbitraire", flush=True)
# print("", flush=True)
# nnTrainTest(abalone_train_X, abalone_train_Y, abalone_test_X, abalone_test_Y, abalone_train_X.shape[1], 1,
#             nnAbaloneHdnAct, nnAbaloneOutAct, nnAlpha, 100, True, "ABALONE")
### 2.4.2. Selection du nombre d'epoques par CV
print("ABALONE Dataset - Selection Nb Epoques par CV", flush=True)
print("", flush=True)
nnCvNbEpoch(abalone_train_X, abalone_train_Y, 4, 1, nnAbaloneHdnAct, nnAbaloneOutAct, nnAlpha,
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], False, "ABALONE")
# ### 2.4.3. Selection de la dimension par CV
# print("ABALONE Dataset - Selection Dimension par CV", flush=True)
# print("", flush=True)
# nnCvDimension(abalone_train_X, abalone_train_Y, range(1, 8, 1), 1, nnAbaloneHdnAct, nnAbaloneOutAct, nnAlpha,
#               nnAbaloneNbEpoch, False, "ABALONE")
# ### 2.2.4. Selection de la profondeur par CV
# print("ABALONE Dataset - Selection Profondeur par CV", flush=True)
# print("", flush=True)
# nnCvDepth(abalone_train_X, abalone_train_Y, nnAbaloneDimension, range(1, 6, 1), nnAbaloneHdnAct, nnAbaloneOutAct,
#           nnAlpha, nnAbaloneNbEpoch, False, "ABALONE")
# ### 2.2.5. Entrainement et test avec les hyperparametres finaux
# print("ABALONE Dataset - Test HyperParam sélectionnés", flush=True)
# print("", flush=True)
# nnTrainTest(abalone_train_X, abalone_train_Y, abalone_test_X, abalone_test_Y, nnAbaloneDimension, nnAbaloneDepth,
#             nnAbaloneHdnAct, nnAbaloneOutAct, nnAlpha, nnAbaloneNbEpoch, True, "ABALONE")