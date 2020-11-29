### 17, 18, 33, 42번쨰 줄을 바꿔가며 실험해볼 수 있습니다.
### 17, 18번째 줄은 policy 그래프 상에서 어떤 두 점을 이을건지 정하는 부분


import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools

matplotlib.use('Agg')

num_states = 2
num_actions = 2
gamma = 0.9

start = np.array([0.2, 0.8])
end = np.array([0.6, 0.3])

color = np.array([152./255, 225./255, 152./255]).reshape(1,-1)

# random reward function
r = np.random.uniform(-1, 1, num_states*num_actions)

# random transition function
alphas = np.ones(num_states)
P = np.zeros((num_states*num_actions, num_states))
for sa in range(num_states*num_actions):
    P[sa] = np.random.dirichlet(alphas) 


# r = [r(a1, s1), r(a2, s1), r(a1, s2), r(a2, s2)]
r = np.array([0.68, -0.89,  0.9,  -0.9])

# P = [
# ..[P(s1| a1, s1), P(s2| a1, s1)],
# ..[P(s1| a2, s1), P(s2| a2, s1)],
# ..[P(s1| a1, s2), P(s2| a1, s2)],
# ..[P(s1| a2, s2), P(s2| a2, s2)]
#]

P = np.array([[ 0.3,  0.7],
       [ 0.9,  0.1],
       [ 0.1,  0.9],
       [ 0.4,  0.6]])



num_samples = 50000
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



n_samp = 101
pol_samp = np.linspace(start, end, n_samp)
val_samp = []

for i in range(n_samp):
    Pi = np.zeros((num_states, num_states*num_actions))
    Pi[0,0] = pol_samp[i, 0]
    Pi[0,1] = 1 - pol_samp[i, 0]
    Pi[1,2] = pol_samp[i, 1]
    Pi[1,3] = 1 - pol_samp[i, 1]

    P_pi = np.matmul(Pi, P)
    r_pi = np.matmul(Pi, r)

    V_pi = np.matmul(np.linalg.inv((np.eye(num_states) - gamma*P_pi)), r_pi)

    val_samp.append(V_pi)




plt.figure(figsize=(10,5))

plt.subplot(121)
plt.scatter(*zip(*sample_policy), c=color, edgecolors=color, s=1)
plt.scatter([0,0,1,1], [0,1,0,1], color='red', s=10)
plt.scatter(*zip(*pol_samp), color = 'red', s=1)
ax = plt.gca()
ax.set_xlabel('P(a1|s1)')                            # x label
ax.set_ylabel('P(a1|s2)')                            # y label

plt.subplot(122)
plt.scatter(*zip(*value_functions), c=color, edgecolors=color, s=1)
plt.scatter(*zip(*value_functions_deter), color='red', s=10)
plt.scatter(*zip(*val_samp), color = 'red', s=1)
ax = plt.gca()
ax.set_xlabel('V(s1)')                            # x label
ax.set_ylabel('V(s2)')                            # y label

plt.savefig('simul_policy.png')






# Value functions of semi-deterministic policies
'''params_sd_policies = []
for s in range(num_states):
  for a in range(num_actions):
    p = np.zeros(num_actions)
    p[a] = 1
    params_sd_policies.append((s, p))

value_functions_semi_deter = []
for params in params_sd_policies:
  state, state_policy = params
 
  for _ in range(10000):    
    alphas = np.ones(num_actions)
    Pi = np.zeros((num_states, num_states*num_actions))
    for s in range(num_states):
      Pi[s, s*num_actions:(s+1)*num_actions] = np.random.dirichlet(alphas)
    Pi[state, state*num_actions:(state+1)*num_actions] = state_policy    

    P_pi = np.matmul(Pi, P)
    r_pi = np.matmul(Pi, r)
    V_pi = np.matmul(np.linalg.inv((np.eye(num_states) - gamma*P_pi)), r_pi)
    value_functions_semi_deter.append(V_pi)
    

plt.figure(figsize=(5,5))
plt.scatter(*zip(*value_functions), color=color, s=1)
plt.scatter(*zip(*value_functions_semi_deter), color='orange', s=1)
ax = plt.gca()
plt.savefig('simul3_right.png')
'''