import time

import numpy as np

from classifieur import Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

SplEntropy = "entropy"
SplGini = "gini"
StpMinPopulation = "min-pop"
StpMaxDepth = "max-depth"
StpMaxLeafs = "max-leafs"
StpMinPurity = "min-purity"
DscQuantiles = "quantiles"
DscEqWidth = "eq-width"
DscSwitchClass = "switch-class"


def computeSplitVal(left, right, method="mean"):
    if method == "mean":
        return float(left + right) / 2
    if method == "left":
        return left
    if method == "right":
        return right
    return 0


def computeImpurity(Y, splitter) -> float:
    unique_Y = np.unique(Y)
    impurity_s = np.zeros(len(unique_Y), dtype=float)
    for iy in range(unique_Y.shape[0]):
        dataYY = np.where(Y == unique_Y[iy])
        prob = float(len(dataYY[0])) / float(len(Y))
        if splitter == SplEntropy:
            impurity_s[iy] = -prob * np.log2(prob)
        else:
            impurity_s[iy] = 1.0 / float(len(Y)) - prob
    return impurity_s.sum()


def mostCommon(arr):
    """
    Count votes for classes
    """
    counts = np.bincount(arr)
    return np.argmax(counts)


class TreeNode:

    def __init__(self, splitter, stopper, stopperV, discretizer, discretizerV, popX=None, popY=None, popTotal=0,
                 parent=None):
        self.splitter = splitter
        self.stopper = stopper
        self.stopperV = stopperV
        self.discretizer = discretizer
        self.discretizerV = min(discretizerV, len(popX)) if popX is not None else discretizerV
        self.popX = popX
        self.popY = popY
        self.popTotal = popTotal
        self.parent = parent
        self.isRoot = self.parent is None
        self.depth = 0 if self.parent is None else (1 + self.parent.depth)
        self.leafs = 1
        self.n_pop = len(popX) if popX is not None else 0
        self.impurity = (
                float(len(popY)) * computeImpurity(popY, splitter) / float(popTotal)) if popY is not None else 0
        self.splitVar = None
        self.splitVal = None
        self.vote = mostCommon(popY) if popY is not None else None
        self.children = []

    def isSplittable(self) -> bool:
        if len(self.children) > 0:
            return False
        unique_Y = np.unique(self.popY)
        if self.popX is None or self.popY is None:
            return False
        elif len(unique_Y) == 1:
            return False
        elif self.stopper == StpMinPopulation:
            if len(self.popX) < 2 * self.stopperV:
                return False
        elif self.stopper == StpMaxDepth:
            if self.depth == self.stopperV:
                return False
        return True

    def isLeaf(self) -> bool:
        return len(self.children) == 0

    def getLeafsPaths(self):
        if self.isLeaf() and self.isRoot:
            return [[]]
        elif self.isLeaf():
            return []
        else:
            paths = []
            for i in range(len(self.children)):
                child_i_paths = self.children[i].getLeafsPaths()
                if len(child_i_paths) > 0:
                    for child_i_path in child_i_paths:
                        paths.append([i] + child_i_path)
                else:
                    paths.append([i])
            return paths

    def getNode(self, path):
        node = self
        if len(path) > 0:
            for i in path:
                # node = node.children[i]
                if i < len(node.children):
                    node = node.children[i]
                else:
                    node = None
                    break
        return node

    def filterSplittableLeafs(self, leafsPaths):
        leafs = []
        for path in leafsPaths:
            node = self.getNode(path)
            if node.isSplittable():
                leafs.append(path)
        return leafs

    def discretizeVar(self, xi):
        breaks = []
        x_var = self.popX[:, xi]
        y = self.popY
        x_ind = range(len(x_var))
        x_mat = np.transpose(np.array([x_var, y, x_ind]))
        x_nrow = x_mat.shape[0]
        if self.discretizer == DscSwitchClass:
            x_mat = x_mat[x_mat[:, 0].argsort()]
            prev_y = x_mat[0][1]
            for i in range(x_nrow)[1:]:
                curr_y = x_mat[i][1]
                if curr_y != prev_y:
                    _break = computeSplitVal(x_mat[i - 1][0], x_mat[i][0])
                    breaks.append(_break)
                    prev_y = curr_y
        else:
            if self.discretizerV == x_nrow:
                for i in range(x_nrow)[1:]:
                    _break = computeSplitVal(x_mat[i - 1][0], x_mat[i][0])
                    breaks.append(_break)
            elif self.discretizer == DscEqWidth:
                x_min = min(x_var[:, 0])
                x_max = max(x_var[:, 0])
                x_width = (x_max - x_min) / float(self.discretizerV)
                x_curr = x_min
                for i in range(self.discretizerV - 1):
                    x_curr += x_width
                    breaks.append(x_curr)
            else:
                probs = 1.0 / float(self.discretizerV)
                prob_curr = 0
                for i in range(self.discretizerV - 1):
                    prob_curr += probs
                    breaks.append(np.quantile(x_var[:, 0], prob_curr))
        return breaks

    def computeChildrenImpurity(self, xi, xi_break) -> float:
        x_var = self.popX[:, xi]
        left_ind = np.where(x_var < xi_break)[0]
        right_ind = np.where(x_var >= xi_break)[0]
        left_p = float(len(left_ind)) / float(self.n_pop)
        right_p = float(len(right_ind)) / float(self.n_pop)
        left_Y = self.popY[left_ind]
        right_Y = self.popY[right_ind]
        left_impurity = computeImpurity(left_Y, self.splitter)
        right_impurity = computeImpurity(right_Y, self.splitter)
        return left_p * left_impurity + right_p * right_impurity

    def splitNode(self):
        candidates = []
        impurity_gain = []
        for xi in range(self.popX.shape[1]):
            xi_breaks = self.discretizeVar(xi)
            for xi_break in xi_breaks:
                candidates.append((xi, xi_break))
                children_impurity = self.computeChildrenImpurity(xi, xi_break)
                parent_impurity = computeImpurity(self.popY, self.splitter)
                impurity_gain.append(parent_impurity - children_impurity)
        best_impurity_ind = np.argmax(np.array(impurity_gain))
        best_impurity_gain = np.max(np.array(impurity_gain))
        spl_var, spl_val = candidates[best_impurity_ind]
        return spl_var, spl_val, best_impurity_gain

    def computeChildren(self, splitVar, splitVal):
        self.splitVar = splitVar
        self.splitVal = splitVal
        self.vote = None
        x_var = self.popX[:, splitVar]
        left_ind = np.where(x_var < splitVal)[0]
        right_ind = np.where(x_var >= splitVal)[0]
        left_X = self.popX[left_ind, :]
        right_X = self.popX[right_ind, :]
        left_Y = self.popY[left_ind]
        right_Y = self.popY[right_ind]
        left_child = TreeNode(self.splitter, self.stopper, self.stopperV, self.discretizer, self.discretizerV,
                              popX=left_X, popY=left_Y, popTotal=self.popTotal, parent=self)
        right_child = TreeNode(self.splitter, self.stopper, self.stopperV, self.discretizer, self.discretizerV,
                               popX=right_X, popY=right_Y, popTotal=self.popTotal, parent=self)
        self.children.append(left_child)
        self.children.append(right_child)

    def updateNodeParams(self):
        if len(self.children) > 0:
            for i in range(len(self.children)):
                self.children[i].updateNodeParams()
            self.leafs = np.array([child.depth for child in self.children]).sum()
            self.impurity = np.array([child.impurity for child in self.children]).sum()

    def augmentTree(self) -> bool:
        if not self.isRoot:
            return False
        leafsPaths = self.getLeafsPaths()
        flLeafsPaths = self.filterSplittableLeafs(leafsPaths)
        if len(flLeafsPaths) == 0:
            return False
        elif self.stopper == StpMaxLeafs:
            if self.leafs >= self.stopperV:
                return False
        elif self.stopper == StpMinPurity:
            if self.impurity <= self.stopperV:
                return False
        bestLeaf = None
        bestImpurityGain = 0.0
        splVar = None
        splVal = None
        for path in flLeafsPaths:
            leaf = self.getNode(path)
            spl_var, spl_val, best_impurity_gain = leaf.splitNode()
            if best_impurity_gain > bestImpurityGain:
                bestImpurityGain = best_impurity_gain
                bestLeaf = path
                splVar = spl_var
                splVal = spl_val
        if bestLeaf is not None:
            leaf = self.getNode(bestLeaf)
            leaf.computeChildren(splVar, splVal)
            self.updateNodeParams()
        return True

    def getPathByXVar(self, x):
        if len(self.children) == 0:
            return []
        else:
            xSplVal = x[self.splitVar]
            child = 0 if xSplVal < self.splitVal else 1
            child_path = self.children[child].getPathByXVar(x)
            leafPath = [child] + child_path
            return leafPath

    def predictNewX(self, x):
        leafPath = self.getPathByXVar(x)
        leaf = self.getNode(leafPath)
        return leaf.vote

    def getLeafsParents(self):
        leafsPaths = self.getLeafsPaths()
        leafsParents = [",".join([str(v) for v in path[:-1]]) for path in leafsPaths]
        leafsParents = list(set(leafsParents))
        return [[int(s) for s in path.split(",")] if path != "" else [] for path in leafsParents]

    def pruning(self, rate=0.05) -> bool:
        if self.isLeaf() and self.isRoot:
            return False
        leafsParents = self.getLeafsParents()
        hasPruned = False
        for path in leafsParents:
            node = self.getNode(path)
            if node is not None:
                children_impurity = node.computeChildrenImpurity(node.splitVar, node.splitVal)
                parent_impurity = computeImpurity(node.popY, node.splitter)
                gain_impurity_rate = (parent_impurity - children_impurity) / parent_impurity
                mustPrune = (node.children[0].isLeaf() and node.children[1].isLeaf() and
                             node.children[0].vote == node.children[1].vote) or (gain_impurity_rate < rate)
                if mustPrune:
                    node.children = []
                    node.splitVar = None
                    node.splitVal = None
                    node.impurity = (float(len(node.popY)) / float(node.popTotal)) * parent_impurity
                    node.vote = mostCommon(node.popY)
                    hasPruned = True
        self.updateNodeParams()
        return hasPruned

    def displayNode(self, varNames, prefix) -> str:
        if len(self.children) > 0:
            varName = varNames[self.splitVar]
            txt = ""
            for i in range(len(self.children)):
                sign = "<" if i == 0 else ">="
                txt = txt + prefix + "--- " + varName + " " + sign + str(self.splitVal) + "\n"
                child = self.children[i]
                txt = txt + child.displayNode(varNames, prefix + "  |") + "\n"
                txt = txt.replace("\n\n", "\n")
        else:
            txt = prefix + "--- " + "Class: " + str(self.vote) + "\n"
        return txt

    def displayTree(self, varNames) -> str:
        if self.isRoot:
            txt = "-- " + " Root" + "\n"
            return txt + self.displayNode(varNames, "|")
        return ""


class DecisionTree(Classifier):

    def __init__(self, splitter, stopper, stopperV, discretizer, discretizerV, **kwargs):
        super().__init__(**kwargs)
        self.root = None
        self.splitter = splitter
        self.stopper = stopper
        self.stopperV = stopperV
        self.discretizer = discretizer
        self.discretizerV = discretizerV
        self.stTime = 0
        self.trainTime = 0

    def train(self, train, train_labels):
        self.root = TreeNode(self.splitter, self.stopper, self.stopperV, self.discretizer, self.discretizerV,
                             popX=train, popY=train_labels, popTotal=len(train))
        mustContinue = True
        self.stTime = time.time()
        while mustContinue is True:
            mustContinue = self.root.augmentTree()
        self.trainTime = time.time() - self.stTime
        return None

    def predict(self, x):
        return self.root.predictNewX(x)

    def evaluate(self, X, y):
        return super().evaluate(X, y)

    def prune(self, rate=0.05):
        mustContinue = True
        while mustContinue is True:
            mustContinue = self.root.pruning(rate)
        self.trainTime = time.time() - self.stTime
        return None

    def display(self, varNames, file=None):
        txt = self.root.displayTree(varNames)
        if file is not None:
            f = open(file, "w")
            f.write(txt)
            f.close()
        else:
            print(txt)


def decisionTreeScikitLearn(inputs):
    """
    Train and test Scikit learn KNN classifier
    Args:
        inputs: Training and test dataset
    Returns:
        Accuracy of classifier on test data
    """
    clf = DecisionTreeClassifier(random_state=1)
    train_X, train_Y, test_X, test_Y = inputs
    clf.fit(train_X, train_Y)
    y_pred = clf.predict(test_X)
    accuracy = accuracy_score(test_Y, y_pred)
    return accuracy
