from collections import deque
import torch.optim as optim
import torch 
import torch.nn as nn  
import torch.nn.functional as F
from Trading_Env import TradingEnvironment 
import numpy as np 
from torch.distributions import Categorical
from torch.autograd import Variable

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim 
        self.action_dim = action_dim 

        self.input_layer = nn.Linear(self.state_dim, 128)
        self.hidden_1 = nn.Linear(128, 128)
        self.hidden_2 = nn.Linear(32,31)
        self.hidden_state = torch.tensor(torch.zeros(2,1,32)) 
        self.rnn = nn.GRU(128, 32, 2)
        self.action_head = nn.Linear(31, 5) 

    def forward(self, state):  
        x = torch.sigmoid(self.input_layer(state))
        x = torch.tanh(self.hidden_1(x))
        x, self.hidden_state = self.rnn(x.view(1,-1,128), self.hidden_state.data)
        x = F.relu(self.hidden_2(x.squeeze()))
        action_scores = self.action_head(x) 
        probs = F.softmax(action_scores)
        m = Categorical(probs)
        action = m.sample() 
        return action, m.log_prob(action) 

def compute_rewards(rewards, gamma):
    discounted_rewards = np.zeros(len(rewards))
    moving_add = 0
    for i in reversed(range(0, len(rewards))):
        moving_add = moving_add*gamma + rewards[i]
        discounted_rewards[i] = moving_add

    return discounted_rewards


class Critic(nn.Module): 
    def __init__(self, state_dim, hidden_dim=20, output_dim=1, lambd=.9):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim, output_dim, bias=False) 
        self.lambd = lambd 
 
    def forward(self, state): 
        x = F.relu(self.l1(state)) 
        x = self.l2(x) 
        return F.softmax(x, dim=0) 
    

def train(env, n_episodes, policy, critic, gamma, print_every=4):
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    #optimizer_v = optim.Adam(critic.parameters(), lr=.05)
    scores_deque = deque(maxlen=100)
    log_interval = 10 
    running_reward = 0
    running_rewards = []
    total_rewards = [] 
    for i in range(n_episodes):  
        traj_log_probs = [] 
        rewards = []
        values = [] 
        state = env.reset()
        score = 0 
        done = False
        
        while not done:  
            action, log_prob = policy(state) 
            value = critic.forward(state) 
            values.append(value) 
            traj_log_probs.append(log_prob)
            next_state, reward, done, msg = env.step(action) 
            score += reward 
            rewards.append(reward) 

            state = torch.FloatTensor(next_state) 

        scores_deque.append(score)  
        total_rewards.append(score)
            
        disc_rewards = compute_rewards(rewards, gamma)
        disc_rewards = torch.tensor([disc_rewards]).float()  
    
        values = torch.cat(values) 

        # advantage: discounted_rewards - function approximated values 
        advantage = disc_rewards - values
    
        policy_loss = []  
        for log_prob in traj_log_probs:
            policy_loss.append(-log_prob * advantage) 
         
        policy_loss = torch.cat(policy_loss).sum() 
    
        
        #value_loss = Variable(value_loss, requires_grad = True)

        # MSE loss 
        value_loss_1 = advantage.pow(2).mean()
        value_loss_1 = Variable(value_loss_1, requires_grad = True)
        #create one loss function for actor and critic
        total_loss = torch.stack([policy_loss, value_loss_1]).sum()
       
        # zero gradients from previous iteration 
        optimizer.zero_grad()
        total_loss.backward() 
        optimizer.step()
        s
        running_reward = running_reward * (1 - 1/log_interval) + reward * (1/log_interval)
        running_rewards.append(running_reward) 
        
        #early stopping 
        """
        if msg["msg"] == "done" and env.portfolio_value() > env.starting_portfolio_value * 1.5 and running_reward > 500:
            print("Early Stopping: " + str(int(reward)))
            break
        """
        if i % log_interval == 0:
            print("""Episode {}: started at {:.1f}, finished at {:.1f} because {} @ t={}, \
            last reward {:.1f}, running reward {:.1f}""".format(i, env.starting_portfolio_value, \
                env.portfolio_value(), msg["msg"], env.cur_timestep, reward, running_reward))

   
    
    return total_rewards

policy = Actor(8, 5)
critic = Critic(8)
serieslength = 250 
if __name__ == "__main__": 
    num_episodes = 100
    GAMMA = .9 
    series_length = 250 
    env = TradingEnvironment(max_stride=4, series_length=serieslength, starting_cash_mean=100, randomize_cash_std=100, starting_shares_mean=30, randomize_shares_std=10) 
    train(env, num_episodes, policy, critic, GAMMA, print_every=10)

    total_rewards = 0 
    total_profits = 0
    failed_goes = 0 
    num_goes = 120
    env = TradingEnvironment(max_stride=4, series_length=serieslength,starting_cash_mean=100, randomize_cash_std=100, starting_shares_mean=100,randomize_shares_std=10)

    for i in range(num_goes):
        done = False 
        env.reset()
        reward_this_go = 1e-8 
        for j in range(0, env.series_length+1):  
            action, log_prob = policy(torch.FloatTensor(env.state))
            next_state, reward, done, msg = env.step(action)
            
            if msg['msg'] == "done":
                reward_this_go = env.portfolio_value()  
                break
            if done:
                break 
        total_profits += (env.portfolio_value() - env.starting_portfolio_value) / env.starting_portfolio_value

        #total_profits = (env.portfolio_value() - env.starting_portfolio_value()) / (env.starting_portfolio_value())
        
        if reward_this_go == 1e-8: 
            failed_goes += 1
        else:
            total_rewards += reward_this_go

    if failed_goes == num_goes:
        print("failed all")
    else:
        print("Failed goes: {} / {}, Avg Rewards per successful game: {}".format(failed_goes, num_goes, total_rewards / (num_goes - failed_goes)))
        print("Avg % profit per game: {}".format(total_profits / num_goes))
        print("Avg % profit per finished game: {}".format(total_profits / (num_goes - failed_goes)))
            
    #env = TradingEnvironment(max_stride=4, series_length=250, starting_cash_mean=1000, randomize_cash_std=100, starting_shares_mean=100, randomize_shares_std=10)
    env = TradingEnvironment(max_stride=4, series_length=serieslength,starting_cash_mean=100, randomize_cash_std=100, starting_shares_mean=100,randomize_shares_std=10)

    env.reset()
    print("starting portfolio value {}".format(env.portfolio_value()))
    for i in range(0,env.series_length + 1):
        action, _ = policy(torch.FloatTensor(env.state))
        next_state, reward, done, msg = env.step(action)
        if msg["msg"] == 'bankrupted self':
            print('bankrupted self by 1')
            break
        if msg["msg"] == 'sold more than have':
            print('sold more than have by 1')
            break
        print("{}, have {} aapl and {} msft and {} cash".format(msg["msg"], next_state[0], next_state[1], next_state[2]))
        if msg["msg"] == "done":
            print(next_state, reward)
            print("total portfolio value {}".format(env.portfolio_value()))
            break    
