import pdb
#import cv2
import gym
import math
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from collections import namedtuple
from collections import deque
from itertools import count
from PIL import Image
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

#matplotlib.use('Agg')


# ### 하이퍼파라미터
# 하이퍼파라미터
EPISODES = 300  # 애피소드 반복횟수
EPS_START = 1  # 학습 시작시 에이전트가 무작위로 행동할 확률
EPS_END = 0.1   # 학습 막바지에 에이전트가 무작위로 행동할 확률
EPS_DECAY = 1000  # 학습 진행시 에이전트가 무작위로 행동할 확률을 감소시키는 값
GAMMA = 0.9
   # 할인계수
LR = 0.002      # 학습률
BATCH_SIZE = 64  # 배치 크기
TARGET_UPDATE = 5


env = gym.make('MountainCar-v0')
env.reset()


# if gpu is to be used
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#device = torch.device("cpu")

## DQN Agent
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
        
    def forward(self, x):
        #x = x / 255.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.position = 0

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state,
                            action,
                            torch.FloatTensor([reward]),
                            torch.FloatTensor([next_state])))


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr = LR)
memory = ReplayMemory(100000)

steps_done = 0


def act(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            # pdb.set_trace()
            return policy_net(state.to(device)).max(1)[1].view(1, 1).cpu()
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)


def learn():
        if len(memory) < BATCH_SIZE:
            return
        batch = memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states).to(device)
        actions = torch.cat(actions).to(device)
        rewards = torch.cat(rewards).to(device)
        next_states = torch.cat(next_states).to(device)

        current_q = policy_net(states).gather(1, actions)
        max_next_q = target_net(next_states).detach().max(1)[0]
        expected_q = rewards + (GAMMA * max_next_q)

        loss = F.mse_loss(current_q.squeeze(), expected_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


score_history = []

# ## 학습 시작

for e in range(EPISODES):
    state = env.reset()
    steps = 0

    
    while True:
        state = torch.FloatTensor([state])
        action = act(state)
        next_state, reward, done, _ = env.step(action.item())
    
        # 게임이 끝났을 경우 플러스 보상주기
        if next_state[0] > 0.5:
             reward += 10

        reward += 15 * (abs(next_state[1])) 
        
        memory.memorize(state, action, reward, next_state)
        learn()

        state = next_state
        steps += 1

        
        #
        # pdb.set_trace()
        if done:
            print("에피소드:{0} 점수: {1}".format(e, steps))
            score_history.append(steps)
            break
    if e % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        #print(target_net.state_dict())
        #print(policy_net.state_dict())
    


plt.plot(score_history)
plt.ylabel('score')
#pdb.set_trace()
plt.savefig('mc_dqnfinal_score.png')


x = np.arange(-1.2, 0.6, 0.05)             # points in the x axis
y = np.arange(-0.07, 0.07, 0.005)             # points in the y axis
X, Y = np.meshgrid(x, y)               # create the "base grid"

X1 = np.ravel(X)
Y1 = np.ravel(Y)
Z = np.dstack([X1,Y1])
Z = torch.FloatTensor([Z]).to(device)
Z = policy_net(Z).detach().max(3)[0].cpu()
Z = Z.numpy().reshape(29,36)



fig = plt.figure()
ax = fig.gca(projection='3d')             # 3d axes instance
surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,cmap=cm.RdPu,linewidth=1,antialiased=True)

#ax.set_title('Mountain Value Function')       # title
ax.set_xlabel('Position')                            # x label
ax.set_ylabel('Velocity')                            # y label
ax.set_zlabel('Value')                        # z label
fig.colorbar(surf)#, shrink=0.5, aspect=0.2)  # colour bar


plt.xticks([-1.2, -0.9, -0.6, -0.3, 0, 0.3 , 0.6]) # x축 단위 바꾸기 
plt.yticks([-0.06, -0.03, 0, 0.03, 0.06]) # y축 단위 바꾸기 
 
ax.view_init(elev=19,azim=34)               # elevation & angle
ax.dist = 9                                             # distance from the plot
fig.savefig('mc_dqnfinal.png')




#dataset = pd.DataFrame(Z)
#sns.heatmap(dataset, annot=True, fmt='d')
#plt.title('Annoteat cell with numeric value', fontsize=20)
#plt.savefig('mc_heatmap.png')



#env.close()
