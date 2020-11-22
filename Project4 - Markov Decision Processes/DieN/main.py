"""
CS 7642 Homework 1
"""
#from mdptoolbox import mdp
#from hiive import mdptoolbox.mdp

from hiive import mdptoolbox
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp

from util import *
from problems import *
from QL import *
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def run_program(N, B):
    # Arbitrary threshold to consider realistic horizon
    threshold = 0.01

    # Iterator variables
    max_timestep = 0

    # Calculate maximum likely horizon using arbitrary threshold
    while True:
        p = pow(1. * (N - sum(B)) / N, max_timestep)
        if p < threshold:
            break
        max_timestep += 1

    # State is bankroll, from 0 to N*max_timesteps (inclusive)
    states = range(N * max_timestep + 1)

    # Extra state for each possible bankroll to indicate terminal state
    #Nstates =len(states) * 2

    # Actions are always roll or quit, encoded to {0, 1}
    actions = [0, 1]

    T = build_transition_matrix(len(states), N, B)
    R = build_reward_matrix(len(states))

    # Gamma is 1 since we don't value future reward any less than immediate
    gamma = 1.0

    # Arbitrary threshold epsilon
    # "epsilon-optimal policy"
    #epsilon = 0.01
    epsilon1 = 1
    epsilon2 = 0.01
    epsilon3 = 0.0001

    vi1_value = []
    vi1_time = []
    pi1_value = []
    pi1_time = []
    vi2_value = []
    vi2_time = []
    pi2_value = []
    pi2_time = []
    vi3_value = []
    vi3_time = []
    pi3_value = []
    pi3_time = []

    iterations = range(1,10,1)
    for i in iterations:

        vi1 = mdptoolbox.mdp.ValueIteration(T, R, gamma, epsilon1, max_iter=i)
        vi1.run()
        vi2 = mdptoolbox.mdp.ValueIteration(T, R, gamma, epsilon2, max_iter=i)
        vi2.run()
        vi3 = mdptoolbox.mdp.ValueIteration(T, R, gamma, epsilon3, max_iter=i)
        vi3.run()

        vi1_time.append(vi1.time)
        vi1_value.append(vi1.V[0])
        vi2_time.append(vi2.time)
        vi2_value.append(vi2.V[0])
        vi3_time.append(vi3.time)
        vi3_value.append(vi3.V[0])


        pi1 = mdptoolbox.mdp.PolicyIterationModified(T, R, gamma, epsilon1, max_iter=i)
        pi1.run()
        pi2 = mdptoolbox.mdp.PolicyIterationModified(T, R, gamma, epsilon2, max_iter=i)
        pi2.run()
        pi3 = mdptoolbox.mdp.PolicyIterationModified(T, R, gamma, epsilon3, max_iter=i)
        pi3.run()

        pi1_time.append(pi1.time)
        pi1_value.append(pi1.V[0])
        pi2_time.append(pi2.time)
        pi2_value.append(pi2.V[0])
        pi3_time.append(pi3.time)
        pi3_value.append(pi3.V[0])

    #print(vi1_value)
    #print(vi2_value)
    #print(vi3_value)
    #print(pi1_value)
    #print(pi2_value)
    #print(pi3_value)

    plt.title("DieN MDP Value")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.ylim(12.8, 13.0)
    lw = 2
    plt.plot(iterations, vi1_value, label="value iteration (epsilon = 1)",color="navy", lw=lw)
    plt.plot(iterations, vi2_value, label="value iteration (epsilon = 0.01)",color="red", lw=lw)
    plt.plot(iterations, vi3_value, label="value iteration (epsilon = 0.0001)",color="green", lw=lw)
    plt.plot(iterations, pi1_value, label="policy iteration (epsilon = 1)",color="blue", lw=lw)
    plt.plot(iterations, pi2_value, label="policy iteration (epsilon = 0.01)",color="purple", lw=lw)
    plt.plot(iterations, pi3_value, label="policy iteration (epsilon = 0.0001)",color="pink", lw=lw)
    plt.legend(loc="best")
    plt.show()

    plt.title("DieN MDP Runtime")
    plt.xlabel("Iteration")
    plt.ylabel("Runtime")
    #plt.ylim(12.8, 13.0)
    lw = 2
    plt.plot(iterations, vi1_time, label="value iteration (epsilon = 1)",color="navy", lw=lw)
    plt.plot(iterations, vi2_time, label="value iteration (epsilon = 0.01)",color="red", lw=lw)
    plt.plot(iterations, vi3_time, label="value iteration (epsilon = 0.0001)",color="green", lw=lw)
    plt.plot(iterations, pi1_time, label="policy iteration (epsilon = 1)",color="blue", lw=lw)
    plt.plot(iterations, pi2_time, label="policy iteration (epsilon = 0.01)",color="purple", lw=lw)
    plt.plot(iterations, pi3_time, label="policy iteration (epsilon = 0.0001)",color="pink", lw=lw)
    plt.legend(loc="best")
    plt.show()

    #np.random.seed(626)
    #ql = mdptoolbox.mdp.QLearning(T, R, gamma, n_iter=10000)
    #ql.setVerbose()
    #ql.run()

    #print ('N={} ... output={}'.format(N, ql.V[0]))
    #print(ql.Q)
    #print (ql.V)
    #print (ql.policy)
    #print (ql.time)

    epsilon = 0.01
    opt = mdptoolbox.mdp.PolicyIterationModified(T, R, gamma, epsilon, max_iter=1000)
    opt.run()

    #np.random.seed(626)
    #ql = QLearning (T, R, gamma, n_iter=1000)
    #ql.run()

    #print ('N={} ... output={}'.format(N, ql.V[0]))
    #print (ql.Q)
    #print (ql.V)
    #print (ql.policy)
    #print (ql.time)

    vi_delta = []
    pi_delta = []
    ql_delta = []
    vi_time = []
    pi_time = []
    ql_time = []

    iterations = range(1,10,1)
    for i in iterations:
        vi = mdptoolbox.mdp.ValueIteration(T, R, gamma, epsilon, max_iter=i)
        vi.run()
        pi = mdptoolbox.mdp.PolicyIterationModified(T, R, gamma, epsilon, max_iter=i)
        pi.run()
        np.random.seed(626)
        ql = QLearning (T, R, gamma, n_iter=i*10000)
        #ql = QLearning (T, R, gamma, n_iter=i*100)
        ql.run()

        vi_delta.append(sum(abs(np.subtract(opt.policy, vi.policy))))
        pi_delta.append(sum(abs(np.subtract(opt.policy, pi.policy))))
        ql_delta.append(sum(abs(np.subtract(opt.policy, ql.policy))))
        vi_time.append(vi.time)
        pi_time.append(pi.time)
        ql_time.append(ql.time)

    #print(vi_time)
    #print(pi_time)
    #print(ql_time)

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('DieN MDP Policy Delta')
    #plt.title("DieN MDP Value")
    #plt.xlabel("Iteration")
    #plt.ylabel("Value")
    #plt.ylim(12.8, 13.0)
    lw = 2
    ax1.plot(iterations, vi_delta, label="value iteration",color="navy", lw=lw)
    ax1.plot(iterations, pi_delta, label="policy iteration",color="red", lw=lw)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Policy Delta")
    ax1.legend(loc="best")
    ax2.plot(range(10000,100000,10000), ql_delta, label="Q learning",color="green", lw=lw)
    #ax2.plot(range(100,1000,100), ql_delta, label="Q learning",color="green", lw=lw)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Policy Delta")
    ax2.legend(loc="best")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('DieN MDP Runtime')
    #plt.title("DieN MDP Value")
    #plt.xlabel("Iteration")
    #plt.ylabel("Value")
    #plt.ylim(12.8, 13.0)
    lw = 2
    ax1.plot(iterations, vi_time, label="value iteration",color="navy", lw=lw)
    ax1.plot(iterations, pi_time, label="policy iteration",color="red", lw=lw)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Runtime")
    ax1.legend(loc="best")
    ax2.plot(range(10000,100000,10000), ql_time, label="Q learning",color="green", lw=lw)
    #ax2.plot(range(100,1000,100), ql_time, label="Q learning",color="green", lw=lw)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Runtime")
    ax2.legend(loc="best")
    plt.show()


    print(ql_delta)
    #print (len(vi.policy))
    #print (sum(abs(np.subtract(vi.policy, vi.policy))))
    #print (sum(abs(np.subtract(vi.policy, pi.policy))))
    #print (sum(abs(np.subtract(vi.policy, ql.policy))))

''' run the test'''
#e = examples[2]
#run_program(e['N'], e['B'])

t = testcase[0]
run_program(t['N'], t['B'])

#for e in examples:
#    run_program(e['N'], e['B'])

#for p in problems:
#    run_program(p['N'], p['B'])

#for t in testcase:
#    run_program(t['N'], t['B'])
