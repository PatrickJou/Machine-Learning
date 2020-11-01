import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV, learning_curve
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, f1_score, homogeneity_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import itertools
import timeit
import time
import os

from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as RCA
from sklearn.ensemble import RandomForestClassifier as RFC

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
from plot_learning_curve import plot_learning_curve

import warnings
warnings.filterwarnings('ignore')

# Import 7.2. Toy datasets from scikit-learn Library
#https://scikit-learn.org/stable/datasets/index.html
from sklearn.datasets import load_digits
data_digits = load_digits()
#X1, Y1 = pd.DataFrame(data_digits["data"]), pd.Series(data_digits["target"])
#Dataset = "digits"

from sklearn.datasets import load_wine
data_wine = load_wine()
X1, Y1 = pd.DataFrame(data_wine["data"],columns=data_wine.feature_names), pd.Series(data_wine["target"])
Dataset = "wine"


Xraw = X1
Xpca = PCA(n_components=5,random_state=5).fit_transform(X1)
Xica = ICA(n_components=5,random_state=5).fit_transform(X1)
Xica /= Xica.std(axis=0)
Xrca = RCA(n_components=5,random_state=5).fit_transform(X1)

# Run RFC
#rfc = RFC(n_estimators=500,min_samples_leaf=round(len(X1)*.01),random_state=5,n_jobs=-1)
#imp = rfc.fit(X1,Y1).feature_importances_
#imp = pd.DataFrame(imp,columns=['Feature Importance'])
#imp.sort_values(by=['Feature Importance'],inplace=True,ascending=False)
#imp['Cum Sum'] = imp['Feature Importance'].cumsum()
#imp = imp[imp['Cum Sum']<=0.35]
#top_cols = imp.index.tolist()
#Xrfc = X1[top_cols]



def MLP_classifier(X, Y, datasource):
    param_range = range(1,201,20)
    train_scores, test_scores = validation_curve(
    MLPClassifier(random_state=626), X, Y, param_name="hidden_layer_sizes", param_range=param_range,
    scoring="accuracy", n_jobs=1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print("train_scores_mean")
    print(train_scores_mean)
    print("test_scores_mean")
    print(test_scores_mean)
    print(np.argmax(test_scores_mean))
    print(test_scores_mean[np.argmax(test_scores_mean)])
    print(param_range[np.argmax(test_scores_mean)])
    hidden_layer_sizes_value = param_range[np.argmax(test_scores_mean)]

    plt.title("MLP Validation Curve on "+ datasource)
    plt.xlabel("hidden_layer")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2

    plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)

    plt.legend(loc="best")
    plt.show()

    MLP_Learning_Curves(X, Y, datasource, hidden_layer_sizes_value)

def MLP_Learning_Curves(X, Y, datasource, hidden_layer_sizes_value):
    title = "MLP Learning Curves on " + datasource
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=626)
    estimator = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes_value, random_state=626)
    plt = plot_learning_curve(estimator, title, X, Y, ylim=(0.0, 1.05), cv=cv)
    plt.show()



MLP_Learning_Curves(Xraw, Y1, "RAW_Digit",101)
MLP_Learning_Curves(Xpca, Y1, "PCA_Digit",101)
MLP_Learning_Curves(Xica, Y1, "ICA_Digit",101)
MLP_Learning_Curves(Xrca, Y1, "RCA_Digit",101)
#MLP_Learning_Curves(Xrfc, Y1, "RAW_Digit",101)






