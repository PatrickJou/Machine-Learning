# Refer to https://github.com/gkhayes/mlrose/blob/master/tutorial_examples.ipynb
import six
import sys
sys.modules['sklearn.externals.six'] = six

import mlrose
import numpy as np
import time
import matplotlib.pyplot as plt

#from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

from ANN_RNC import ANN_RNC_fitting
from ANN_SA import ANN_SA_fitting
from ANN_GA import ANN_GA_fitting

# Load the dataset
#data = load_iris()
data = load_wine()

# Split data into training and test sets
# Training Set Size = 80%
# Testing Set Size = 20%
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2,
                                                    random_state = 3)

# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()
y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()


#ANN_RNC_fitting(X_train_scaled, X_test_scaled, y_train_hot, y_test_hot)
#ANN_SA_fitting(X_train_scaled, X_test_scaled, y_train_hot, y_test_hot)
#ANN_GA_fitting(X_train_scaled, X_test_scaled, y_train_hot, y_test_hot)

iterator = range(100,1000,100)

random_restarts = 10
fitness_score1 = []
fitness_time1 = []
for i in iterator:
    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu',
                                         algorithm ='random_hill_climb',
                                         max_iters = i, bias = True, is_classifier = True,
                                         learning_rate = 0.1, early_stopping = True, restarts = random_restarts,
                                         clip_max = 5, max_attempts = 100, random_state = 3)

    t1 = time.time()
    nn_model1.fit(X_train_scaled, y_train_hot)
    t2 = time.time()

    y_test_pred = nn_model1.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    fitness_score1.append(y_test_accuracy)
    fitness_time1.append(t2-t1)

print(fitness_score1)
print(fitness_time1)

Decay_Schedule = mlrose.GeomDecay(init_temp=1.0, decay=0.99, min_temp=0.001)
fitness_score2 = []
fitness_time2 = []
for i in iterator:
    nn_model2 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu',
                                 algorithm = 'simulated_annealing',
                                 max_iters = i, bias = True, is_classifier = True,
                                 learning_rate = 0.1, early_stopping = True, schedule = Decay_Schedule,
                                 clip_max = 5, max_attempts = 100, random_state = 3)

    t1 = time.time()
    nn_model2.fit(X_train_scaled, y_train_hot)
    t2 = time.time()

    y_test_pred = nn_model2.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    fitness_score2.append(y_test_accuracy)
    fitness_time2.append(t2-t1)

print(fitness_score2)
print(fitness_time2)

pop_size = 100
mutation_prob = 0.3
fitness_score3 = []
fitness_time3 = []
for i in iterator:
    nn_model3 = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu',
                                 algorithm = 'genetic_alg',
                                 max_iters = i, bias = True, is_classifier = True,
                                 learning_rate = 0.1, early_stopping = True, mutation_prob = mutation_prob,
                                 pop_size = pop_size, clip_max = 5, max_attempts = 100, random_state = 3)

    t1 = time.time()
    nn_model3.fit(X_train_scaled, y_train_hot)
    t2 = time.time()

    y_test_pred = nn_model3.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    fitness_score3.append(y_test_accuracy)
    fitness_time3.append(t2-t1)

print(fitness_score3)
print(fitness_time3)

# Plot the Fitness Score over Iteration
plt.title("Fitness Score for ANN_fitting")
plt.xlabel("Iteration")
plt.ylabel("Fitness Score")
lw = 2
plt.plot(iterator, fitness_score1, label="RNC",color="navy", lw=lw)
plt.plot(iterator, fitness_score2, label="SA",color="red", lw=lw)
plt.plot(iterator, fitness_score3, label="GA",color="green", lw=lw)
plt.legend(loc="best")
plt.show()

# Plot the Fitness Score over Iteration
plt.title("Fitness Time for ANN_fitting")
plt.xlabel("Iteration")
plt.ylabel("Fitness Time (s)")
lw = 2
plt.plot(iterator, fitness_time1, label="RNC",color="navy", lw=lw)
plt.plot(iterator, fitness_time2, label="SA",color="red", lw=lw)
plt.plot(iterator, fitness_time3, label="GA",color="green", lw=lw)
plt.legend(loc="best")
plt.show()

