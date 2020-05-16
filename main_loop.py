import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from Trading_Env import TradingEnvironment
from DQN_Agent import DQN_Network, DQN_Agent 
from Replay_Buffer import Transition, Replay_Buffer
import torch.nn.functional as F 
from data_acq import apl_stock, msf_stock, apl_close, msf_close, apl_open, msf_open
import matplotlib.pyplot as plt
from torch.autograd import Variable
from Actor_RNN import Actor_RNN

def optimize_model(memory,optimizer, gamma):
    # cannot sample transiitons if we don't have enough transitions
    if len(memory.storage) < BATCH_SIZE: 
        return

    #1) sample 
    mini_batch = memory.sample(BATCH_SIZE)

    state_action_vals = [] 
    expected_state_action_vals = [] 
    for state, action, next_state, reward, done in mini_batch:
        if done:
            reward = torch.tensor(reward)
            target = reward 
        else: 
            next_state = torch.tensor((next_state)) 
            next_state_values = dqn.target_net(next_state) 
            target = reward + gamma * torch.max(next_state_values)

        expected_state_action_vals.append(target) 
        q_values = dqn.policy_net(torch.tensor(state))
        state_action_val = q_values[action]
        state_action_vals.append(state_action_val)

    expected_state_action_vals = torch.tensor(expected_state_action_vals).clone().detach().requires_grad_(True)
    state_action_vals = torch.tensor(state_action_vals) 
    expected_state_action_vals.requires_grad = True  
    state_action_vals.requires_grad = True  

    #) compute loss (Huber)
    loss = F.smooth_l1_loss(state_action_vals, expected_state_action_vals)

    #6) Backprop 
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()


def train(num_episodes, target_update, gamma, env, dqn, memory, optimizer):
    
    running_reward = 0 
    log_interval = 10 
    for i in range(num_episodes): 
        state = env.reset()
        done = False 
        score = 0 
        msg = None
        while not done: 
            #action = actor_.act(torch.tensor(state))
            action = dqn.select_action(torch.tensor(state))
            next_state, reward, done, msg = env.step(action) 
            memory.storage.append((state, action, next_state, reward, done))
            score += reward
            optimize_model(memory, optimizer, gamma) 
            #state = next_state 
        
        running_reward = running_reward * (1 - 1/log_interval) + reward * (1/log_interval)

        # update the params of the target net
        if (i % target_update) == 0: 
            #print("Updating params of the target net")
            dqn.target_net.load_state_dict(dqn.policy_net.state_dict())

        #early stopping 
        if msg["msg"] == "done" and env.portfolio_value() > env.starting_portfolio_value * 1.5 and running_reward > 500:
            print("Early Stopping: " + str(int(reward)))
            break
            
        if i % log_interval == 0:
            print("""Episode {}: started at {:.1f}, finished at {:.1f} because {} @ t={}, \
            last reward {:.1f}, running reward {:.1f}""".format(i, env.starting_portfolio_value, \
                env.portfolio_value(), msg["msg"], env.cur_timestep, reward, running_reward))


dqn = DQN_Agent()  
serieslength = 250  
env = TradingEnvironment(max_stride=4, series_length=serieslength,starting_cash_mean=100, randomize_cash_std=100, starting_shares_mean=100,randomize_shares_std=10, inaction_penalty=100.0)
BATCH_SIZE = 250 

if __name__ == "__main__": 
    num_episodes = 50 
    gamma = .97 
    target_update = 10 
    replay_buffer = Replay_Buffer(1000)
    optimizer = optim.RMSprop(dqn.policy_net.parameters())
    train(num_episodes, target_update, gamma, env, dqn, replay_buffer, optimizer) 
    #sample trading run 

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
            action = dqn.select_action(torch.tensor(env.state))
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
        action = dqn.select_action(torch.tensor(env.state))
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