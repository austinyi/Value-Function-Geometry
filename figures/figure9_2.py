import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools

matplotlib.use('Agg')

num_states = 2
num_actions = 3
gamma = 0.8

start = np.array([0.2, 1])
end = np.array([0.2, 0])

color = np.array([152./255, 225./255, 152./255]).reshape(1,-1)

# define random reward function
r = np.random.uniform(-1, 1, num_states*num_actions)

# define random transition function
alphas = np.ones(num_states)
P = np.zeros((num_states*num_actions, num_states))
for sa in range(num_states*num_actions):
  P[sa] = np.random.dirichlet(alphas) 


r = np.array([-0.1, -1., 0.1, 0.4, 1.5, 0.1])

P = np.array([[ 0.9,  0.1],
       [ 0.2,  0.8],
       [ 0.7,  0.3],
       [ 0.05,  0.95],
       [ 0.25, 0.75],
       [ 0.3, 0.7]])


num_samples = 100000
value_functions = []
sample_policy = []

for _ in range(num_samples):
    alphas = np.ones(num_actions)
    Pi = np.zeros((num_states, num_states*num_actions))
    policy = np.zeros(num_states)
    for s in range(num_states):
        Pi[s, s*num_actions:(s+1)*num_actions] = np.random.dirichlet(alphas)
        
    policy[0] = Pi[0,0]
    policy[1] = Pi[1,2]

    P_pi = np.matmul(Pi, P)
    r_pi = np.matmul(Pi, r)

    V_pi = np.matmul(np.linalg.inv((np.eye(num_states) - gamma*P_pi)), r_pi)

    value_functions.append(V_pi)
    sample_policy.append(policy)

xmin = min(V[0] for V in value_functions)
xmax = max(V[0] for V in value_functions)
ymin = min(V[1] for V in value_functions)
ymax = max(V[1] for V in value_functions)


#Value Functions of Deterministic Policies
set_deter_actions = []
for i in range(num_actions):
    deter_action = np.zeros(num_actions)
    deter_action[i] = 1
    set_deter_actions.append(deter_action)

value_functions_deter = []
for policies in itertools.product(set_deter_actions, repeat=num_states):
    Pi = np.zeros((num_states, num_states*num_actions))
    for s in range(num_states):
        Pi[s, s*num_actions:(s+1)*num_actions] = policies[s]

    P_pi = np.matmul(Pi, P)
    r_pi = np.matmul(Pi, r)

    V_pi = np.matmul(np.linalg.inv((np.eye(num_states) - gamma*P_pi)), r_pi)
    value_functions_deter.append(V_pi)




  
plt.figure(figsize=(10,5))
plt.subplot(122)
plt.scatter(*zip(*value_functions), c=color, edgecolors=color, s=10)
plt.scatter(*zip(*value_functions_deter), color='red', s=1)

ax = plt.gca()
ax.set_xlabel('V(s1)')                            # x label
ax.set_ylabel('V(s2)')                            # y label







plt.savefig('figure9_2.png')

