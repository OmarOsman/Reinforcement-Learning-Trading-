import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from itertools import count
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorCriticNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(np.prod(self.input_dims), self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, n_actions)
        self.v = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.device)
        

        
    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        state = state.flatten()
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v)

class Agent(object):
    """ Agent class for use with a single actor critic network that shares
        the lowest layers. For use with more complex environments
    """
    def __init__(self, alpha, input_dims, gamma=0.99,
                 layer1_size=32, layer2_size=16, n_actions=2):
        self.gamma = gamma
        self.eps =  np.finfo(np.float32).eps.item()
        self.actor_critic = ActorCriticNetwork(alpha, input_dims, layer1_size,layer2_size, n_actions=n_actions)  
        self.saved_actions =  []
        self.rewards = []
        
        self.losses = []


    def choose_action(self, observation):
        probabilities, state_value = self.actor_critic.forward(observation)
        probabilities = F.softmax(probabilities,dim = -1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        
        self.saved_actions.append(SavedAction(log_probs, state_value))
        return action.item()

    def learn(self, state, reward, done):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = [] 
        
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = T.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, T.tensor([R])))
            
        self.actor_critic.optimizer.zero_grad()
        loss = T.stack(policy_losses).sum() + T.stack(value_losses).sum()
        self.losses.append(loss.item())
        loss.backward()
        self.actor_critic.optimizer.step()
        
        del self.rewards[:]
        del self.saved_actions[:]