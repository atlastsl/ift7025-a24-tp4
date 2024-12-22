import math
import random
import time

import numpy as np

from classifieur import Classifier

NNLayerTypeInput = 'input'
NNLayerTypeHidden = 'hidden'
NNLayerTypeOutput = 'output'

NNActivationSigmoid = 'sigmoid'
NNActivationRelu = 'relu'
NNActivationSoftmax = 'softmax'
NNActivationId = 'id'


def safeExp(x):
    if x > 709.7:
        return float('inf')
    return np.exp(x)


def activationF(x, activation):
    if activation == NNActivationSigmoid:
        return 1 / (1 + np.exp(-x))
    elif activation == NNActivationRelu:
        return np.maximum(0, x)
    elif activation == NNActivationSoftmax:
        return np.exp(x)
    elif activation == NNActivationId:
        return x
    else:
        return 0


class Neuron:
    def __init__(self, layer, position, pLayer, netDepth, activation, ntwNbIn, ntwInOut):
        self.layer = layer
        self.position = position
        self.pLayer = pLayer
        self.activation = activation
        self.value = 0
        self.bValue = 0
        self.delta = 0
        self.target = 0
        if layer == 0:
            self.weights = None
            self.type = NNLayerTypeInput
        else:
            self.weights = np.random.normal(0, math.sqrt(2.0 / float(ntwNbIn + ntwInOut)), 1 + len(pLayer))
            self.type = NNLayerTypeHidden if layer <= netDepth else NNLayerTypeOutput

    def computeValue(self):
        if self.type != NNLayerTypeInput:
            self.bValue = np.dot(self.weights, np.array([1] + [n.value for n in self.pLayer]))
            # if self.bValue < -709:
            #    print(self.bValue, self.weights, [n.value for n in self.pLayer], self.delta)
            self.value = activationF(self.bValue, self.activation)

    def normalizeValue(self, norm):
        if self.activation == NNActivationSoftmax:
            self.value = self.value / norm

    def setValue(self, x):
        if self.type == NNLayerTypeInput:
            self.bValue = x[self.position]
            self.value = x[self.position]

    def setTarget(self, y):
        if self.type == NNLayerTypeOutput:
            self.target = y

    def computeDelta(self, nLayer=None):
        if self.type == NNLayerTypeOutput:
            self.delta = self.value - self.target
        elif nLayer is not None:
            s = np.array([(neuron.delta * neuron.weights[self.position + 1]) for neuron in nLayer]).sum()
            self.delta = self.value * (1 - self.value) * s
        else:
            self.delta = 0


class NeuralNetwork(Classifier):
    def __init__(self, dimension, depth, hdnActivation, outActivation, alpha, nbEpoch, trace, **kwargs):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.depth = depth
        self.hdnActivation = hdnActivation
        self.outActivation = outActivation
        self.alpha = alpha
        self.nbEpoch = nbEpoch
        self.trace = trace
        self.trainX = None
        self.trainY = None
        self.network = []
        self.classes = []
        self.nb_classes = 0
        self.trainTime = 0
        self.epochPerfs = []

    def start(self):
        np.random.seed(1)
        entryLayer = []
        ntwNbIn = self.trainX.shape[1]
        ntwNbOut = self.nb_classes # self.nb_classes if self.nb_classes > 2 else 1
        for j in range(self.trainX.shape[1]):
            entryLayer.append(Neuron(0, j, None, self.depth, NNActivationId, ntwNbIn, ntwNbOut))
        self.network.append(entryLayer)
        pLayer = entryLayer
        for q in range(self.depth):
            hiddenLayer = []
            for j in range(self.dimension):
                hiddenLayer.append(Neuron(q + 1, j, pLayer, self.depth, self.hdnActivation, ntwNbIn, ntwNbOut))
            self.network.append(hiddenLayer)
            pLayer = hiddenLayer
        if self.nb_classes <= 2:
            self.network.append([Neuron(self.depth + 1, 0, pLayer, self.depth, self.outActivation, ntwNbIn, ntwNbOut)])
        else:
            outputLayer = []
            for j in range(self.nb_classes):
                outputLayer.append(Neuron(self.depth + 1, j, pLayer, self.depth, self.outActivation, ntwNbIn, ntwNbOut))
            self.network.append(outputLayer)

    def forward(self, x):
        for j in range(len(self.network[0])):
            self.network[0][j].setValue(x)
        for q in range(1, self.depth + 2):
            for j in range(len(self.network[q])):
                self.network[q][j].computeValue()
            layerActivation = self.hdnActivation if q < self.depth - 1 else self.outActivation
            if layerActivation == NNActivationSoftmax:
                norm = np.array([n.value for n in self.network[q]]).sum()
                for j in range(len(self.network[q])):
                    self.network[q][j].normalizeValue(norm)

    def target(self, y):
        q = self.depth + 1
        if len(self.network[q]) == 1:
            self.network[q][0].setTarget(y)
        else:
            for j in range(len(self.network[q])):
                self.network[q][j].setTarget(int(self.classes[j] == y))

    def prediction(self):
        q = self.depth + 1
        if len(self.network[q]) == 1:
            return 0 if self.network[q][0].value < 0.5 else 1
        else:
            return np.argmax(np.array([n.value for n in self.network[q]]))

    def verify(self, y):
        pred = self.prediction()
        return y == pred

    def backward(self):
        layers_parc = list(reversed(range(1, self.depth + 2)))
        nLayer = None
        for q in layers_parc:
            for j in range(len(self.network[q])):
                self.network[q][j].computeDelta(nLayer)
            nLayer = self.network[q]

    def update(self):
        for q in range(1, self.depth + 2):
            for j in range(len(self.network[q])):
                for i in range(len(self.network[q][j].weights)):
                    a_i = 1 if i == 0 else self.network[q - 1][i - 1].value
                    self.network[q][j].weights[i] = (self.network[q][j].weights[i] -
                                                     self.alpha * a_i * self.network[q][j].delta)

    def train(self, train, train_labels):
        stTime = time.time()
        self.trainX = train
        self.trainY = train_labels
        self.classes = np.sort(np.unique(train_labels))
        self.nb_classes = np.unique(train_labels).size
        self.start()
        self.epochPerfs = []
        for epoch in range(self.nbEpoch):
            epoch_perf = []
            for i in range(self.trainX.shape[0]):
                x = self.trainX[i]
                y = self.trainY[i]
                self.target(y)
                self.forward(x)
                self.backward()
                self.update()
                yp = self.verify(y)
                epoch_perf.append(int(yp))
            epoch_perf = np.array(epoch_perf).mean()
            if self.trace:
                print(f"Neural Network training Epoch {epoch}, Accuracy : {epoch_perf}")
            self.epochPerfs.append(epoch_perf)
        self.trainTime = time.time() - stTime

    def predict(self, x):
        self.forward(x)
        return self.prediction()

    def evaluate(self, X, y):
        return super().evaluate(X, y)
