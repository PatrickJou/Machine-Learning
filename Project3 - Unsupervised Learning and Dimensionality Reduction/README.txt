======================
Code Repo
======================
https://github.com/PatrickJou/Machine-Learning
The assignment 3 source codes are in the folder "Project3 - Unsupervised Learning and Dimensionality Reduction"

======================
Environment
======================
Install Anaconda3
Install Pycharm
Install Python 3.7.9

=====================
Python Package
=====================
scikit-learn  0.23.2
pandas        1.1.1
numpy         1.19.1
matplotlib    3.3.1


============================================
1.0 Clustering and Dimensionality Reduction Analysis Intro 
============================================

There are 5 different dimention reduction data from 2 dataset are performed 2 clustering algorithms
Run each python files for each cases.

1. Run "RAW.py"
2. Run "PCA.py"
3. Run "ICA.py"
4. Run "RCA.py"
5. Run "RFC.py"

In each files, comments out the datashet you want to run
#X1, Y1 = pd.DataFrame(data_digits["data"]), pd.Series(data_digits["target"])
#Dataset = "digits"
#X1, Y1 = pd.DataFrame(data_wine["data"],columns=data_wine.feature_names), pd.Series(data_wine["target"])
#Dataset = "wine"

============================================
2.0 Neural Network Analysis Intro 
============================================

The Neural Network using 4 dimension reduction algorithm are all under "ANN.py"
To compare the performance between algorithm, then run the ANN.py directly

