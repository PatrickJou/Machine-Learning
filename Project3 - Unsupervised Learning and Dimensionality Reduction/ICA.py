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
from itertools import product
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

# Import 7.2. Toy datasets from scikit-learn Library
#https://scikit-learn.org/stable/datasets/index.html
from sklearn.datasets import load_digits
data_digits = load_digits()
X1, Y1 = pd.DataFrame(data_digits["data"]), pd.Series(data_digits["target"])
Dataset = "digits"

from sklearn.datasets import load_wine
data_wine = load_wine()
#X1, Y1 = pd.DataFrame(data_wine["data"],columns=data_wine.feature_names), pd.Series(data_wine["target"])
#Dataset = "wine"

# Run ICA
dims = list(np.arange(2,(X1.shape[1]-1),3))
dims.append(X1.shape[1])
ica = ICA(random_state=5)
kurt = []

for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(X1)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt.append(tmp.abs().mean())

# Plot ICA
plt.figure()
plt.title("ICA Kurtosis: "+ Dataset)
plt.xlabel("Independent Components")
plt.ylabel("Avg Kurtosis Across IC")
plt.plot(dims, kurt, 'b-')
plt.grid(False)
plt.show()

###################################################################
# ICA Dimensional Reduction
X1 = ICA(n_components=5,random_state=5).fit_transform(X1)
X1 /= X1.std(axis=0)
###################################################################

Kclusters = range(2,50,2)
km_sil_scores = []
km_homo_scores = []
km_inertia_scores = []
km_fitness_times = []

for k in Kclusters:
        t1 = time.time()
        km = KMeans(n_clusters=k, n_init=10,random_state=100,n_jobs=-1).fit(X1)
        t2 = time.time()

        km_fitness_times.append(t2 - t1)
        km_sil_scores.append(silhouette_score(X1, km.labels_))
        km_homo_scores.append(homogeneity_score(Y1, km.labels_))
        km_inertia_scores.append(km.inertia_)


em_sil_scores = []
em_homo_scores = []
em_aic_scores = []
em_bic_scores = []
em_fitness_times = []

for k in Kclusters:
        t1 = time.time()
        em = GaussianMixture(n_components=k,covariance_type='diag',n_init=1,warm_start=True,random_state=100).fit(X1)
        t2 = time.time()

        em_fitness_times.append(t2 - t1)
        em_sil_scores.append(silhouette_score(X1, em.predict(X1)))
        em_homo_scores.append(homogeneity_score(Y1, em.predict(X1)))
        em_aic_scores.append(em.aic(X1))
        em_bic_scores.append(em.bic(X1))

# [KM] Plot the Cluster Score over K cluster
plt.title("Cluster Score for K Mean (KM) on ICA " + Dataset)
plt.xlabel("K cluster")
plt.ylabel("Inertia")
#plt.ylim(0.0, 1.1)
lw = 2
plt.plot(Kclusters, km_inertia_scores, label="inertia",color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

# [KM] Plot the Fitness Time over K cluster
plt.title("Fitness Time for K Mean (KM) on ICA " + Dataset)
plt.xlabel("K cluster")
plt.ylabel("Fitness Time (s)")
#plt.ylim(0.0, 1.1)
lw = 2
plt.plot(Kclusters, km_fitness_times, label="fitness time",color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

# [EM] Plot the Cluster Score over K cluster
fig, ax1 = plt.subplots()
ax1.set_xlabel('K cluster')
ax1.plot(Kclusters, em_aic_scores, color="red")
ax1.set_ylabel('AIC_score', color="red")
ax1.tick_params('y', colors="red")
plt.grid(False)

ax2 = ax1.twinx()
ax2.plot(Kclusters, em_bic_scores, color="navy")
ax2.set_ylabel('BIC_score', color="navy")
ax2.tick_params('y', colors="navy")
plt.grid(False)

plt.title("Cluster Score for Expectation Maximization (EM) on ICA " + Dataset)
fig.tight_layout()
plt.show()


# [EM] Plot the Fitness Time over K cluster
plt.title("Fitness Time for Expectation Maximization (EM) on ICA " + Dataset)
plt.xlabel("K cluster")
plt.ylabel("Fitness Time (s)")
#plt.ylim(0.0, 1.1)
lw = 2
plt.plot(Kclusters, em_fitness_times, label="fitness time",color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


