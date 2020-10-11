import six
import sys
sys.modules['sklearn.externals.six'] = six

import mlrose
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# neural network object and fit object - random_hill_climb
def ANN_RNC_fitting(X_train_scaled, X_test_scaled, y_train_hot, y_test_hot):

    iterator = range(100,1000,100)

    random_restarts = 1
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

    random_restarts = 5
    fitness_score2 = []
    fitness_time2 = []
    for i in iterator:
        nn_model2 = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu',
                                         algorithm ='random_hill_climb',
                                         max_iters = i, bias = True, is_classifier = True,
                                         learning_rate = 0.1, early_stopping = True, restarts = random_restarts,
                                         clip_max = 5, max_attempts = 100, random_state = 3)

        t1 = time.time()
        nn_model2.fit(X_train_scaled, y_train_hot)
        t2 = time.time()

        y_test_pred = nn_model2.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

        fitness_score2.append(y_test_accuracy)
        fitness_time2.append(t2-t1)

    #print(fitness_score2)
    #print(fitness_time2)

    random_restarts = 10
    fitness_score3 = []
    fitness_time3 = []
    for i in iterator:
        nn_model3 = mlrose.NeuralNetwork(hidden_nodes = [2], activation ='relu',
                                         algorithm ='random_hill_climb',
                                         max_iters = i, bias = True, is_classifier = True,
                                         learning_rate = 0.1, early_stopping = True, restarts = random_restarts,
                                         clip_max = 5, max_attempts = 100, random_state = 3)

        t1 = time.time()
        nn_model3.fit(X_train_scaled, y_train_hot)
        t2 = time.time()

        y_test_pred = nn_model3.predict(X_test_scaled)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

        fitness_score3.append(y_test_accuracy)
        fitness_time3.append(t2-t1)

    #print(fitness_score3)
    #print(fitness_time3)

    # Plot the Fitness Score over Iteration
    plt.title("Fitness Score for ANN_RNC_fitting")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Score")
    #plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(iterator, fitness_score1, label="restarts = 1",color="navy", lw=lw)
    plt.plot(iterator, fitness_score2, label="restarts = 5",color="red", lw=lw)
    plt.plot(iterator, fitness_score3, label="restarts = 10",color="green", lw=lw)
    plt.legend(loc="best")
    plt.show()

    # Plot the Fitness Score over Iteration
    plt.title("Fitness Time for ANN_RNC_fitting")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Time (s)")
    #plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(iterator, fitness_time1, label="restarts = 1",color="navy", lw=lw)
    plt.plot(iterator, fitness_time2, label="restarts = 5",color="red", lw=lw)
    plt.plot(iterator, fitness_time3, label="restarts = 10",color="green", lw=lw)
    plt.legend(loc="best")
    plt.show()
