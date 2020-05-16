import torch 
import torch.nn as nn 
import numpy as np
import random
import math 
import torch.nn.functional as F
import torch.distributions as td
from Trading_Env import TradingEnvironment
from torch.distributions import Categorical 
from Replay_Buffer import Replay_Buffer
from torch.distributions import RelaxedOneHotCategorical
from torch.distributions import Gumbel
import matplotlib.pyplot as plt

torch.manual_seed(51)

class RegularActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RegularActor, self).__init__()
        self.state_dim = state_dim 
        self.action_dim = action_dim 

        self.l1 = nn.Linear(self.state_dim, 128) 
        self.l2 = nn.Linear(128, 32)
        self.l3 = nn.Linear(32, 5)

    def forward(self, state): 
    
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a)) 
        probs = F.softmax(self.l3(a), dim=0)
        m = Categorical(probs)
        action = m.sample()  
        return action
  
    def compute_rewards(self, rewards, gamma):
        discounted_rewards = np.zeros(len(rewards))
        moving_add = 0
        for i in reversed(range(0, len(rewards))):
            moving_add = moving_add*gamma + rewards[i]
            discounted_rewards[i] = moving_add

        return discounted_rewards

class EnsembleCritic(nn.Module):
    def __init__(self, num_qs, state_dim, action_dim): 
        super(EnsembleCritic, self).__init__() 
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.num_qs = num_qs

        self.l1 = nn.Linear(self.state_dim + self.action_dim, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)

        self.input_layer = nn.Linear(self.state_dim + action_dim, 128)
        self.hidden_1 = nn.Linear(128, 128)
        self.hidden_2 = nn.Linear(32,31)
        self.hidden_state = torch.tensor(torch.zeros(2,1,32))
        self.rnn = nn.GRU(128, 32, 2)
        self.value_head = nn.Linear(31, 1)

        self.l4 = nn.Linear(self.state_dim + self.action_dim, 32)
        self.l5 = nn.Linear(32, 32)
        self.l6 = nn.Linear(32, 1)

    def forward(self, state, action, with_var=False):
        all_qs = []
        action = torch.tensor(action, dtype=torch.float)  
        action = action.unsqueeze(1)
        q1 = F.relu(self.l1(torch.cat([state, action], dim=1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        all_qs = torch.cat([q1.unsqueeze(0), q2.unsqueeze(0),], 0) # q3.unsqueeze(0), q4.unsqueeze(0)], 0)   # Num_q x B x 1
        if with_var:
            std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
            return all_qs, std_q
        return all_qs

    def q1(self, state, action): 
        action = action.unsqueeze(0) 
        action = torch.tensor(action, dtype=torch.float)
    
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = torch.sigmoid(self.l3(q1)) 
        return q1 
        
    def q_all(self, state, action, with_var=False):
        all_qs = []
        #action = torch.FloatTensor([action]).unsqueeze(0)
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2)) 
        q2 = self.l6(q2)

        all_qs = torch.cat(
            [q1.unsqueeze(0), q2.unsqueeze(0),], 0) # q3.unsqueeze(0), q4.unsqueeze(0)], 0)  # Num_q x B x 1
        if with_var:
            std_q = torch.std(all_qs, dim=0, keepdim=False, unbiased=False)
            return all_qs, std_q
        return all_qs 
    
class VAE(nn.Module): 
    def __init__(self, state_dim, action_size, latent_dim, categorical_dim, anneal_rate, temp_min, temperature, max_action): 
        super(VAE, self).__init__()
        # action dim: refers to the number of possible actions
        self.categorical_dim = categorical_dim 
        self.state_dim = state_dim 
        self.action_size = action_size 
        self.latent_dim = latent_dim 
        self.temperature = temperature
        self.max_action = max_action
  
        #encoder network 
        self.e1 = nn.Linear(self.state_dim + self.action_size, 120)
        self.e2 = nn.Linear(120, 120)  
        self.e3 = nn.Linear(120, self.latent_dim * self.categorical_dim)

        #decoder network    
        self.d1 = nn.Linear(self.state_dim + (self.latent_dim * self.categorical_dim), 256) 
        self.d2 = nn.Linear(256, 512)
        self.d3 = nn.Linear(512, self.action_size)  
         
    def forward(self, state, action):  
        #encoding 
        state = state.unsqueeze(0)
        action = action.unsqueeze(0)
        q = F.relu(self.e1(torch.cat([state, action], dim=1))) 
        q = F.relu(self.e2(q))
        q = F.relu(self.e3(q))  
        q_y = q.view(q.size(0), self.latent_dim, self.categorical_dim)  # 1 x 2 x 5  

        # decoding  
        z = RelaxedOneHotCategorical(torch.tensor([self.temperature]), logits=q)  
        log_prob = z.log_prob(action)
        sample = z.sample()  
        recon = self.decode(state, sample)     
     
        return recon, F.softmax(q_y, dim=1).reshape(*q.size()))
    
    def kl_loss(self, q_y): 
        # want to do: capture the distribution of our input (the q function) through a VAE 
        # kl di
        # vergence between the r
        log_ratio = torch.log(q_y * self.categorical_dim + 1e-20)
        kl_diverge = torch.sum(q_y * log_ratio, dim=-1).mean()
        return kl_diverge
        
    def decode(self, state, z):
        state_sample = torch.cat([state, z], 1)
        recon = F.relu(self.d1(state_sample))
        recon = F.relu(self.d2(recon))
        recon = torch.sigmoid(self.d3(recon)) 
        return self.max_action*recon

    def decode_multiple(self, state, num_decode, z=None): 
        # decode atleast {x} samples 
        mean = torch.FloatTensor([0])
        std = torch.FloatTensor([1])

        sampled_actions = [] 
        for num in range(num_decode): 
            if z is None:
                z = self.sample_gumbel(10)
                z = z.unsqueeze(0)

            a = F.relu(self.d1(torch.cat([state, z],1))) 
            a = F.relu(self.d2(a))
            a = torch.sigmoid(self.d3(a)) 
            a = self.max_action*a 
            sampled_actions.append(a) 

        sampled_actions = torch.FloatTensor(sampled_actions)
        return sampled_actions   

class BEAR(nn.Module):

    def __init__(self, num_qs, state_dim, action_dim, num_samples_match, anneal_rate, temp_min, default_temp): 
        super(BEAR, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.num_qs = num_qs
        self.latent_dim = self.action_dim * 2
        self.temp_min = temp_min
        self.temperature = default_temp  

        self.actor = RegularActor(state_dim, action_dim)
        self.actor_target = RegularActor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters()) 

        #initialize the critic networks
        self.critic = EnsembleCritic(num_qs, state_dim, action_dim)
        self.critic_target = EnsembleCritic(num_qs, state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = 4
        self.categorical_dim = 5      
        self.action_size = 1
        self.vae = VAE(self.state_dim, self.action_size, self.latent_dim, self.categorical_dim, anneal_rate, temp_min, default_temp, self.max_action)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())  
        self.num_samples_match = num_samples_match

        #lagrange optimizer
        self.log_lagrange2 = torch.randn((), requires_grad=True)
        self.lagrange2_opt = torch.optim.Adam([self.log_lagrange2,], lr=1e-3)

        self.epoch = 0 
        self.steps_done = 0 
        self.eps_start = .9  
        self.eps_end = .05
        self.eps_decay = 200  

    def compute_kernel(self, x, y):
        x = x.unsqueeze(0) # 
        x_size = x.size(0) 
        y = y.unsqueeze(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)

    def mmd_loss_gaussian(self, x, y, sigma=0.2):
        # gaussian kernel
        xx_kernel = self.compute_kernel(x, x)
        yy_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd_loss = xx_kernel.mean() + yy_kernel.mean() - 2*xy_kernel.mean()
        return mmd_loss
    
    def select_action(self, state):   
        
        q1s = []
        actions = [] 
        with torch.no_grad(): 
            for _ in range(5):  
                action = self.actor(state.unsqueeze(0))
                action = torch.FloatTensor([action])
                actions.append(action)
                q1 = self.critic.q1(state.unsqueeze(0), action) 
                #print(q1) 
                q1s.append(q1.item())   
        q1s = torch.FloatTensor([q1s]).view(1, 5)
        ind = torch.argmax(q1s)
        action = actions[ind] 

        return action 


    def train(self, replay_buffer, batch_size, gamma=.9, tau=.005):
        
        # 1): sample minibatch 
        if len(replay_buffer.storage) < batch_size: 
            return 
        
        mini_batch = replay_buffer.sample(batch_size)  
        state_t = torch.FloatTensor(mini_batch[0][0]) 
        next_state_t = torch.FloatTensor(mini_batch[0][2]) 
        for state, action, next_state, reward, done in mini_batch: 
            state = torch.FloatTensor(state)
            action = torch.FloatTensor([action])
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            done = torch.FloatTensor(1 - done)
        
        # ______TRAINING______
        # Train the Behaviour cloning policy to be able to take more than 1 sample for MMD 
        # recon_action: to produce gradients
        # smapled_action: discretized into 0, 1, 2, 3, 4
        # train the vae 
        recon_action, q_y = self.vae(state, action)  
        #sampled_action = torch.round(recon_action) 
        #take the tensor off of recon later 
        action_ = action.unsqueeze(0)
        vae_mse_loss = nn.MSELoss()
        recon_loss = vae_mse_loss(recon_action, action_)
        kl_loss = self.vae.kl_loss(q_y)
        vae_loss = recon_loss + kl_loss 
        
        self.vae_optimizer.zero_grad() 
        vae_loss.backward() 
        self.vae_optimizer.step() 
        # train the critic (Q functions) Ensemble critic 
        # compute y(s,a) 
        with torch.no_grad():
            # Duplicate state 10 times (10 is a hyperparameter chosen by BCQ)
            next_state_t = next_state_t.unsqueeze(0)
            state_rep = torch.FloatTensor(np.repeat(next_state_t, 10, axis=0))

           #state_rep = next_state_t.repeat(10, 1)
            #state_rep = torch.FloatTensor(np.repeat(ne, 10, axis=0)).to(device)
            
            # Compute value of perturbed actions sampled from the VAE
            target_Q1, target_Q2 = self.critic_target(state_rep, self.actor_target(state_rep)) 

            # Soft Clipped Double Q-learning
            target_Q = 0.75 * torch.min(target_Q1, target_Q2) + 0.25 * torch.max(target_Q1, target_Q2) 

            #target_Q = 0.75 * target_Qs.min(0)[0] + 0.25 * target_Qs.max(0)[0] 
            target_Q = target_Q.view(10, -1).max(0)[0].view(-1, 1)
            target_Q = reward + done * gamma * target_Q

        state = state.unsqueeze(0) 
        #action = action.unsqueeze(1)
        current_Qs = self.critic(state, action, with_var=False)  
        current_Q1 = current_Qs[0]
        current_Q2 = current_Qs[1]
        loss = nn.MSELoss()
        critic_loss = loss(current_Q1, target_Q) + loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step() 
        
        # get the actions 
        sampled_actions = self.vae.decode_multiple(state, num_decode=10)
        sampled_actions_actor = []
        num_decode = 10
        for _ in range(num_decode):
            action= self.actor(state)  
            sampled_actions_actor.append(action)

        sampled_actions_actor = torch.FloatTensor(sampled_actions_actor)
        # compute the mmd loss 
        mmd_loss = self.mmd_loss_gaussian(sampled_actions, sampled_actions_actor)

        num_samples = num_decode 
        
        sampled_actions_actor_i = torch.FloatTensor([sampled_actions_actor[0]]).unsqueeze(0)
        critic_qs, std_q = self.critic.q_all(state, sampled_actions_actor_i, with_var=True)  
        # state.unsqueeze(0) (1, 1, 8)  
        # state.repeat: (10, 1, 8)
        # .view: (10, 8,  
        sampled_actions_actor = sampled_actions_actor.unsqueeze(1) 
        critic_qs = self.critic.q_all(state.unsqueeze(0).repeat(num_samples, 1, 1).view(num_samples*state.size(0), state.size(1)), sampled_actions_actor) 
        #critic_qs = critic_qs.view(self.num_qs, num_samples, sampled_actions_actor.size(), 1)
        critic_qs = critic_qs.mean(1)
        std_q = torch.std(critic_qs, dim=0, keepdim=False, unbiased=False)
        
        _lambda = .4 
        delta_conf = .1  
        #support matching with a warmstart  
        if self.epoch >= 20: 
            actor_loss = (-critic_qs +\
                            _lambda * (np.sqrt((1 - delta_conf)/delta_conf)) * std_q +\
                            self.log_lagrange2.exp() * mmd_loss).mean()
        else:
            actor_loss = (self.log_lagrange2.exp() * mmd_loss).mean()
        
        #print(actor_loss)
        std_loss = _lambda*(np.sqrt((1 - delta_conf)/delta_conf)) * std_q.detach() 
    
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        
        # lagrange optimization 
        thresh = 0.05 
        lagrange_thresh = 10.0 
        lagrange_loss = (-critic_qs +\
                        _lambda * (np.sqrt((1 - delta_conf)/delta_conf)) * (std_q) +\
                        self.log_lagrange2.exp() * (mmd_loss - thresh)).mean()

        self.lagrange2_opt.zero_grad()
        (-lagrange_loss).backward()
        # self.lagrange1_opt.step()
        self.lagrange2_opt.step() 
        self.log_lagrange2.data.clamp_(min=-5.0, max=lagrange_thresh)   
        
        # Update Target Networks 
        tau = .005 
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.epoch += 1 

serieslength = 250 

env = TradingEnvironment(max_stride=4, series_length=serieslength,starting_cash_mean=100, randomize_cash_std=100, starting_shares_mean=100,randomize_shares_std=10)

"""
###_________Testing Actor____________###
regular_actor = RegularActor(8, 5)
# action dim for VAE refers to the size of the action array, not number of actions
for ep in range(10): 
    done = False 
    state = env.reset() 
    while not done: 
        action, log_probs = regular_actor.act(state) 
        next_state, reward, done, msg = env.step(action)
    
        #compute gradients
""" 
"""
#______TESTING VAE______
categorical_dim = 5 
action_size = 1
state_dim = 8 
latent_dim = action_size*2
ANNEAL_RATE = .00003
TEMP_MIN = 0.5  
default_temp = 1.0 
vae = VAE(state_dim, action_size, latent_dim, categorical_dim, ANNEAL_RATE, TEMP_MIN, default_temp) 
state = env.reset() 
state = torch.FloatTensor(state).unsqueeze(0)
action = torch.FloatTensor(1).unsqueeze(1)
#action = torch.FloatTensor([0, 1, 0, 0, 0])
recon, q_y = vae(state, action) 
print("q_y:", q_y)  
print("Recon:", recon)  
kl_loss = vae.kl_loss(q_y) 
mse_loss = vae.mse_loss(recon, torch.FloatTensor([1]))
#state = state.unsqueeze(1) 
"""

"""
_____TESTING ENSEMBLE CRITIC_____
ensemble_critic = EnsembleCritic(1, 8, 1)
state = torch.FloatTensor(env.reset()).unsqueeze(0)
action = torch.FloatTensor(1).unsqueeze(1)
all_qs = ensemble_critic(state, action)
print(all_qs)
"""
ANNEAL_RATE = .00003
TEMP_MIN = 0.005  
default_temp = 1.0
log_interval = 10 
replay = Replay_Buffer(1000)
state = torch.FloatTensor(env.reset()).unsqueeze(0) 
num_qs = 1
state_dim = 8 
num_samples_match = 10 
NUM_EPISODES = 50  
batch_size = 250  
bear = BEAR(num_qs, state_dim, 1, 10, ANNEAL_RATE, TEMP_MIN, default_temp)
running_rewards = [] 

if __name__ == "__main__":
    
    running_reward = 0  
    for i in range(NUM_EPISODES): 
        state = env.reset()
        done = False
        while not done: 
            action = bear.select_action(state)  
            next_state, reward, done, msg = env.step(action) 
            replay.storage.append((state, action, next_state, reward, done)) 
            bear.train(replay, batch_size=batch_size, gamma=.9, tau=.005) 
            state = torch.FloatTensor(next_state) 
        if i % 30 == 1:
            bear.temperature = np.maximum(bear.temperature * np.exp(-.00003 * i), bear.temp_min)
        
        running_reward = running_reward * (1 - 1/log_interval) + reward * (1/log_interval)
        running_rewards.append(running_reward) 
        
        #early stopping 
        if msg["msg"] == "done" and env.portfolio_value() > env.starting_portfolio_value * 1.5 and running_reward > 500:
            print("Early Stopping: " + str(int(reward)))
            break
            
        if i % log_interval == 0:
            print("""Episode {}: started at {:.1f}, finished at {:.1f} because {} @ t={}, \
            last reward {:.1f}, running reward {:.1f}""".format(i, env.starting_portfolio_value, \
                env.portfolio_value(), msg["msg"], env.cur_timestep, reward, running_reward))


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
            action = bear.select_action(torch.FloatTensor(env.state))
            next_state, reward, done, msg = env.step(action)
            
            if msg['msg'] == "done":
                reward_this_go = env.portfolio_value()  
                break
            if done:
                break 
        total_profits += (env.portfolio_value() - env.starting_portfolio_value) / env.starting_portfolio_value
        
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
            
    env = TradingEnvironment(max_stride=4, series_length=serieslength,starting_cash_mean=100, randomize_cash_std=100, starting_shares_mean=100,randomize_shares_std=10)

    env.reset()
    print("starting portfolio value {}".format(env.portfolio_value()))
    for i in range(0,env.series_length + 1):
        action = bear.select_action(torch.FloatTensor(env.state))
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


        
            
            
        
