======================
Code Repo
======================
https://github.com/PatrickJou/Machine-Learning
The assignment 2 source codes are in the folder "Project2 - Randomized Optimization"

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
mlrose        1.3.0

============================================
1.0 Optimization Problems Steps Intro 
============================================

The 3 Optimization Problems using 4 algorithm are all under "Bit strings Fitting.py"
To select one of three problems, please uncomments the fitness and ProbelmName for the probelm you select. 

# Select fitness function object
# fitness = mlrose.OneMax()
# ProblemName = "One Max"
# fitness = mlrose.FlipFlop()
# ProblemName = "Flip Flop"
# fitness = mlrose.FourPeaks(t_pct=0.15)
# ProblemName = "Four Peaks"

============================================
2.0 Weights Optimization for Neural Network Steps Intro 
============================================

The Weights Optimization for Neural Network using 3 algorithm are all under "ANN Fitting.py"
To find the best hyperparamter for RNC & SA & GA, uncomment the following function

#ANN_RNC_fitting(X_train_scaled, X_test_scaled, y_train_hot, y_test_hot)
#ANN_SA_fitting(X_train_scaled, X_test_scaled, y_train_hot, y_test_hot)
#ANN_GA_fitting(X_train_scaled, X_test_scaled, y_train_hot, y_test_hot)

To compare the performance between algorithm
Then run the ANN Fitting.py directly

