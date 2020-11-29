import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools

matplotlib.use('Agg')

num_states = 2
num_actions = 2
gamma = 0.9

color = np.array([152./255, 225./255, 152./255]).reshape(1,-1)

# define random reward function
r = np.random.uniform(-1, 1, num_states*num_actions)

# define random transition function
alphas = np.ones(num_states)
P = np.zeros((num_states*num_actions, num_states))
for sa in range(num_states*num_actions):
  P[sa] = np.random.dirichlet(alphas) 

# MDP defined in Figure 3,4,5,6 left
#r = np.array([-0.1, -1., 0.1, 0.4, 1.5, 0.1])
r = np.array([0.68, -0.89,  0.9,  -0.9])
P = np.array([[ 0.3,  0.7],
       [ 0.9,  0.1],
       [ 0.1,  0.9],
       [ 0.4,  0.6]])
'''
P = np.array([[ 0.9,  0.1],
       [ 0.2,  0.8],
       [ 0.7,  0.3],
       [ 0.05,  0.95],
       [ 0.25, 0.75],
       [ 0.3, 0.7]])
'''

num_samples = 100000
value_functions = []
for _ in range(num_samples):
    alphas = np.ones(num_actions)
    Pi = np.zeros((num_states, num_states*num_actions))
    for s in range(num_states):
        Pi[s, s*num_actions:(s+1)*num_actions] = np.random.dirichlet(alphas)

    P_pi = np.matmul(Pi, P)
    r_pi = np.matmul(Pi, r)

    V_pi = np.matmul(np.linalg.inv((np.eye(num_states) - gamma*P_pi)), r_pi)
    value_functions.append(V_pi)

xmin = min(V[0] for V in value_functions)
xmax = max(V[0] for V in value_functions)
ymin = min(V[1] for V in value_functions)
ymax = max(V[1] for V in value_functions)
eps = 0.2

  
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(*zip(*value_functions), c=color, edgecolors=color, s=3)
ax = plt.gca()

#ax.set_title('Mountain Value Function')       # title
ax.set_xlabel('V(s1)')                            # x label
ax.set_ylabel('V(s2)')                            # y label
ax.dist = 5






num_states = 2
num_actions = 3
gamma = 0.8

color = np.array([152./255, 225./255, 152./255]).reshape(1,-1)


# MDP defined in Figure 3,4,5,6 left
r = np.array([-0.1, -1., 0.1, 0.4, 1.5, 0.1])

P = np.array([[ 0.9,  0.1],
       [ 0.2,  0.8],
       [ 0.7,  0.3],
       [ 0.05,  0.95],
       [ 0.25, 0.75],
       [ 0.3, 0.7]])


num_samples = 100000
value_functions = []
for _ in range(num_samples):
    alphas = np.ones(num_actions)
    Pi = np.zeros((num_states, num_states*num_actions))
    for s in range(num_states):
        Pi[s, s*num_actions:(s+1)*num_actions] = np.random.dirichlet(alphas)

    P_pi = np.matmul(Pi, P)
    r_pi = np.matmul(Pi, r)

    V_pi = np.matmul(np.linalg.inv((np.eye(num_states) - gamma*P_pi)), r_pi)
    value_functions.append(V_pi)

xmin = min(V[0] for V in value_functions)
xmax = max(V[0] for V in value_functions)
ymin = min(V[1] for V in value_functions)
ymax = max(V[1] for V in value_functions)
eps = 0.2

  


plt.subplot(122)
plt.scatter(*zip(*value_functions), c=color, edgecolors=color, s=3)
ax = plt.gca()

#ax.set_title('Mountain Value Function')       # title
ax.set_xlabel('V(s1)')                            # x label
ax.set_ylabel('V(s2)')                            # y label
ax.dist = 5
plt.savefig('figure2_left.png')

