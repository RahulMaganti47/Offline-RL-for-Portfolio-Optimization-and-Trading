import math, random
import gym
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd  
import torch.nn.functional as F
from Trading_Env import TradingEnvironment 

class DQN_Network(nn.Module):
    def __init__(self): 
        super(DQN_Network, self).__init__() 
        
        # __GRU__ 
        """
        self.input_layer = nn.Linear(8, 128)
        self.hidden_1 = nn.Linear(128, 128)
        self.hidden_2 = nn.Linear(32,31)
        self.hidden_state = torch.tensor(torch.zeros(2,1,32))
        self.rnn = nn.GRU(128, 32, 2)
        self.action_head = nn.Linear(31, 5)
        """
        
        self.input_layer = nn.Linear(8, 32)
        self.hidden_1 = nn.Linear(32, 32)
        self.hidden_2 = nn.Linear(32,31)
        self.output_layer = nn.Linear(31, 5)
    

    def forward(self, state):
        state = state.squeeze()

        """
        out = torch.sigmoid(self.input_layer(state))
        out = torch.tanh(self.hidden_1(out))
        out, self.hidden_state = self.rnn(out.view(1,-1,128), self.hidden_state.data)
        out = F.relu(self.hidden_2(out.squeeze()))
        out = self.action_head(out)
        """
        out = F.relu(self.input_layer(state))
        out = F.relu(self.hidden_1(out))
        out = F.relu(self.hidden_2(out))
        out = F.sigmoid(self.output_layer(out))

        """
        out = F.relu(self.input_layer(state))
        out = F.relu(self.hidden_1(out))
        out = F.relu(self.hidden_2(out))
        out = F.relu(self.output_layer(out))
        """
        return out 

class DQN_Agent(): 
    def __init__(self): 
        #initialize target and policy networks 
        self.target_net = DQN_Network()
        self.policy_net = DQN_Network()

        self.eps_start = .9  
        self.eps_end = .05
        self.eps_decay = 200  
        self.steps_done = 0 

    def select_action(self, state): 
        random_n = random.random() #generate random number
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1  

        if (random_n < eps_threshold): 
        #take random action (random # betwee 0 and 4)
            action = torch.tensor([random.randrange(4)]) 
        else: 
        #take the best action  
            with torch.no_grad(): 
                actions = self.policy_net(state)  
                action = torch.argmax(actions).view(1, 1) 
                
        return action.item() 


