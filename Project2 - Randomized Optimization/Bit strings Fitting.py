import six
import sys
sys.modules['sklearn.externals.six'] = six

import mlrose
import numpy as np
import time
import matplotlib.pyplot as plt

# Select Optimization Problems
fitness = mlrose.OneMax()
ProblemName = "One Max"
#fitness = mlrose.FlipFlop()
#ProblemName = "Flip Flop"
#fitness = mlrose.FourPeaks(t_pct=0.15)
#ProblemName = "Four Peaks"

# Define optimization problem object [24 bits String]
problem = mlrose.DiscreteOpt(length = 24, fitness_fn = fitness, maximize=True, max_val=2)

# Define Interation Parameter
#iterator = [5, 10, 15, 20, 25, 50, 100, 150, 200, 250]
iterator = range(5,100,5)
print(iterator)

# Solve using (1) simulated annealing [SA]
fitness_score1 = []
fitness_time1 = []
for i in iterator:
    t1 = time.time()
    best_state, best_fitness = mlrose.simulated_annealing(problem, schedule = mlrose.ExpDecay(), max_attempts = 10, max_iters = i,random_state = 626)
    t2 = time.time()
    fitness_score1.append(best_fitness)
    fitness_time1.append(t2-t1)
print(fitness_score1)
print(fitness_time1)

# Solve using (2) random_hill_climb [RHC]
fitness_score2 = []
fitness_time2 = []
for i in iterator:
    t1 = time.time()
    best_state, best_fitness= mlrose.random_hill_climb(problem, restarts = 10, max_attempts = 10, max_iters = i, random_state = 626)
    t2 = time.time()
    fitness_score2.append(best_fitness)
    fitness_time2.append(t2-t1)
print(fitness_score2)
print(fitness_time2)

# Solve using (3) genetic_alg [GA]
fitness_score3 = []
fitness_time3 = []
for i in iterator:
    t1 = time.time()
    best_state, best_fitness = mlrose.genetic_alg(problem, max_attempts = 10, max_iters = i, random_state = 626)
    t2 = time.time()
    fitness_score3.append(best_fitness)
    fitness_time3.append(t2-t1)
print(fitness_score3)
print(fitness_time3)

# Solve using (4) mimic [MIMIC]
fitness_score4 = []
fitness_time4 = []
for i in iterator:
    t1 = time.time()
    best_state, best_fitness = mlrose.mimic(problem, max_attempts = 10,max_iters = i, random_state = 626)
    t2 = time.time()
    fitness_score4.append(best_fitness)
    fitness_time4.append(t2-t1)
print(fitness_score4)
print(fitness_time4)

# Plot the Fitness Score over Iteration
plt.title("Fitness Score for Problem "+ ProblemName)
plt.xlabel("Iteration")
plt.ylabel("Fitness Score")
#plt.ylim(0.0, 1.1)
lw = 2
plt.plot(iterator, fitness_score1, label="SA",color="darkorange", lw=lw)
plt.plot(iterator, fitness_score2, label="RHC",color="navy", lw=lw)
plt.plot(iterator, fitness_score3, label="GA",color="green", lw=lw)
plt.plot(iterator, fitness_score4, label="MIMIC",color="red", lw=lw)
plt.legend(loc="best")
plt.show()

# Plot the Fitness Score over Iteration
plt.title("Fitness Time for Problem "+ ProblemName)
plt.xlabel("Iteration")
plt.ylabel("Fitness Time (s)")
#plt.ylim(0.0, 1.1)
lw = 2
plt.plot(iterator, fitness_time1, label="SA",color="darkorange", lw=lw)
plt.plot(iterator, fitness_time2, label="RHC",color="navy", lw=lw)
plt.plot(iterator, fitness_time3, label="GA",color="green", lw=lw)
plt.plot(iterator, fitness_time4, label="MIMIC",color="red", lw=lw)
plt.legend(loc="best")
plt.show()








