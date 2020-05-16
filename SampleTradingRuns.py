import matplotlib.pyplot as plt 
from data_acq import apl_stock, msf_stock, apl_close, msf_close, apl_open, msf_open
from Trading_Env import TradingEnvironment 
from main_loop import dqn, BATCH_SIZE
from RegularActor import bear, running_rewards, batch_size 
import torch 
from train_actor import policy

batch_size = BATCH_SIZE 

apl_open_orig = apl_stock["Open"].values
apl_close_orig = apl_stock["Close"].values
msf_open_orig = msf_stock["Open"].values
msf_close_orig = msf_stock["Close"].values
apl_open_orig[:108] /= 7
apl_close_orig[:108] /= 7

env = TradingEnvironment(max_stride=4, series_length=250, starting_cash_mean=100, randomize_cash_std=100, starting_shares_mean=100, randomize_shares_std=10)
env.reset()
complete_game = False
while not complete_game:
    bought_apl_at = []
    bought_msf_at = []
    sold_apl_at = []
    sold_msf_at = []
    bought_apl_at_orig = []
    bought_msf_at_orig = []
    sold_apl_at_orig = []
    sold_msf_at_orig = []
    nothing_at = []
    ba_action_times = []
    bm_action_times = []
    sa_action_times = []
    sm_action_times = []
    n_action_times = []
    starting_val = env.starting_portfolio_value
    print("Starting portfolio value: {}".format(starting_val))
    for i in range(0,env.series_length + 1):
        #action, _ = policy(torch.tensor(env.state))
        action = dqn.select_action(torch.tensor(env.state))
        #action = bear.select_action(torch.FloatTensor(env.state))
        #print("Action:", action)
        if action == 0:
            bought_apl_at.append(apl_open[env.cur_timestep])
            bought_apl_at_orig.append(apl_open_orig[env.cur_timestep])
            ba_action_times.append(env.cur_timestep)
        if action == 1:
            sold_apl_at.append(apl_close[env.cur_timestep])
            sold_apl_at_orig.append(apl_close_orig[env.cur_timestep])
            sa_action_times.append(env.cur_timestep)
        if action == 2:
            nothing_at.append(35)
            n_action_times.append(env.cur_timestep)
        if action == 3:
            bought_msf_at.append(msf_open[env.cur_timestep])
            bought_msf_at_orig.append(msf_open_orig[env.cur_timestep])
            bm_action_times.append(env.cur_timestep)
        if action == 4:
            sold_msf_at.append(msf_close[env.cur_timestep])
            sold_msf_at_orig.append(msf_close_orig[env.cur_timestep])
            sm_action_times.append(env.cur_timestep)
        next_state, reward, done, msg = env.step(action)
        #print(msg["msg"])
        if msg["msg"] == 'bankrupted self':
            env.reset()
            break
        if msg["msg"] == 'sold more than have': 
            print("More than have")
            env.reset()
            break
        if msg["msg"] == "done":
            print("{}, have {} aapl and {} msft and {} cash".format(msg["msg"], next_state[0], next_state[1], next_state[2]))
            val = env.portfolio_value()
            
            print("Finished portfolio value {}".format(val))  
            if val >= starting_val: 
                
                portfolio_change = (val - starting_val) 
                print("Portfolio Delta:", portfolio_change.item())  
                return_t = 100*(portfolio_change/starting_val)
                rounded = torch.round(return_t * 10**2) / (10**2)
                portfolio_return = rounded
                portfolio_return = round(portfolio_return.item(), 2)
                print("Portfolio Return: {} %".format(portfolio_return)) 

                complete_game = True
            env.reset() 
            break



hfont = {'fontname':'Gill Sans MT'} 

plt.figure(1, figsize=(14,5))
apl = plt.subplot(121) 
plt.grid(b=True, which='major', color='#666666', linestyle='-') 
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title("Apple", **hfont)
plt.text(-5, 70, "batch size: {}".format(batch_size), fontsize=8, **hfont)
plt.xlabel("TradingSteps(Days)", **hfont) 
plt.ylabel("SharePrice($)", **hfont)
msf = plt.subplot(122)
plt.title("Microsoft", **hfont) 
plt.text(-5, 70, "batch size: {}".format(batch_size), fontsize=8, **hfont)
plt.xlabel("TradingSteps(Days)", **hfont) 
plt.ylabel("SharePrice($)", **hfont) 
plt.grid(b=True, which='major', color='#666666', linestyle='-') 
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
apl.plot(range(0, len(apl_open)), apl_open, label="AAPL raw share price")
msf.plot(range(0, len(msf_open)), msf_open, label="MFST raw share price")
apl.plot(ba_action_times, bought_apl_at, "ro", label="Bought: AAPL")
apl.plot(sa_action_times, sold_apl_at, "go", label="Sold: AAPL")
apl.plot(n_action_times, nothing_at, "yx", label="Inaction")
msf.plot(n_action_times, nothing_at, "yx", label="Inaction")
msf.plot(bm_action_times, bought_msf_at, "ro", label="Bought: MSFT")
msf.plot(sm_action_times, sold_msf_at, "go", label="Sold: MSFT") 
apl.legend(loc="upper left")
msf.legend(loc="upper left") 
plt.show()


plt.figure(1, figsize=(14,5))
apl = plt.subplot(121)
plt.grid(b=True, which='major', color='#666666', linestyle='-') 
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title("Apple", **hfont) 
plt.text(1000, 200, "batch size: {}".format(batch_size), fontsize=8, **hfont)
plt.xlabel("TradingSteps(Days)", **hfont) 
plt.ylabel("SharePrice($)", **hfont)
msf = plt.subplot(122)
plt.title("Microsoft", **hfont) 
plt.text(1000, 200, "batch size: {}".format(batch_size), fontsize=8, **hfont)
plt.xlabel("TradingSteps(Days)", **hfont) 
plt.ylabel("SharePrice($)", **hfont)
plt.grid(b=True, which='major', color='#666666', linestyle='-') 
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
apl.plot(range(0, len(apl_open_orig)), apl_open_orig, label="AAPL raw share price")
msf.plot(range(0, len(msf_open_orig)), msf_open_orig, label="MFST raw share price")
apl.plot(ba_action_times, bought_apl_at_orig, "ro", label="Bought: AAPL")
apl.plot(sa_action_times, sold_apl_at_orig, "go", label="Sold: AAPL")
apl.plot(n_action_times, nothing_at, "yx", label="Inaction")
msf.plot(n_action_times, nothing_at, "yx", label="Inaction")
msf.plot(bm_action_times, bought_msf_at_orig, "ro", label="Bought: MSFT")
msf.plot(sm_action_times, sold_msf_at_orig, "go", label="Sold: MSFT") 
apl.legend(loc="upper left")
msf.legend(loc="upper left") 
plt.show()
