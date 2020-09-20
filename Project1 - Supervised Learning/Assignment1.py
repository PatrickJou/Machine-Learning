from DT_classifier import DT_classifier
from MLP_classifier import MLP_classifier
from ADA_classifier import ADA_classifier
from RF_classifier import RF_classifier
from SVM_classifier import SVM_classifier
from KNN_classifier import KNN_classifier
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# Import 7.2. Toy datasets from scikit-learn Library
#https://scikit-learn.org/stable/datasets/index.html
from sklearn.datasets import load_iris
data_iris = load_iris()
X1, Y1 = pd.DataFrame(data_iris["data"],columns=data_iris.feature_names), pd.Series(data_iris["target"])

from sklearn.datasets import load_digits
data_digits = load_digits()
X2, Y2 = pd.DataFrame(data_digits["data"]), pd.Series(data_digits["target"])

from sklearn.datasets import load_wine
data_wine = load_wine()
X3, Y3 = pd.DataFrame(data_wine["data"],columns=data_wine.feature_names), pd.Series(data_wine["target"])

from sklearn.datasets import load_breast_cancer
data_cancer = load_breast_cancer()
X4, Y4 = pd.DataFrame(data_cancer["data"],columns=data_cancer.feature_names), pd.Series(data_cancer["target"])

# First Classification Question: digits dataset
X = X2
Y = Y2
dataset = "digits"

# (1) Decision Tree (DT)classifier - Pruning
# HyperParameter: max tree depth = 13 [Test Score = 0.7830]
DT_classifier(X, Y, dataset)

# (2) Multi-layer Perceptron (MLP) classifier - Neural Networks
# HyperParameter: hidden layer size = 101 [Test Score = 0.9449]
MLP_classifier(X, Y, dataset)

# (3-1) ADABoost (ADA) classifier - Ensemble learning (Boosting methods)
# HyperParameter: n estimators number = 11 [Test Score = 0.2677]
ADA_classifier(X, Y, dataset)

# (3-2) Random Forest (RF) classifier - Ensemble learning (Bagging methods)
# HyperParameter: n estimators number = 71 [Test Score = 0.9383]
RF_classifier(X, Y, dataset)

# (4) Support Vector Machines (SVM) classifier
# HyperParameter: gamma value = 0.01 [Test Score = 0.9722]
SVM_classifier(X, Y, dataset)

# (5) k-nearest neighbors (KNN) classifier
# HyperParameter: K neighbors number = 2 [Test Score = 0.9672]
KNN_classifier(X, Y, dataset)

# Second Classification Question: wine dataset
X = X3
Y = Y3
dataset = "wines"

# (1) Decision Tree (DT)classifier - Pruning
# HyperParameter: max tree depth = 4 [Test Score = 0.9049]
DT_classifier(X, Y, dataset)

# (2) Multi-layer Perceptron (MLP) classifier - Neural Networks
# HyperParameter: hidden layer size = 101 [Test Score = 0.9387]
MLP_classifier(X, Y, dataset)

# (3-1) ADABoost (ADA) classifier - Ensemble learning (Boosting methods)
# HyperParameter: n estimators number = 31 [Test Score = 0.8144]
ADA_classifier(X, Y, dataset)

# (3-2) Random Forest (RF) classifier - Ensemble learning (Bagging methods)
# HyperParameter: n estimators number = 71 [Test Score = 0.9778]
RF_classifier(X, Y, dataset)

# (4) Support Vector Machines (SVM) classifier
# HyperParameter: gamma value = 0.003 [Test Score = 0.7198]
SVM_classifier(X, Y, dataset)

# (5) k-nearest neighbors (KNN) classifier
# HyperParameter: K neighbors number = 1 [Test Score = 0.7251]
KNN_classifier(X, Y, dataset)
