import matplotlib.pyplot as plt
import quandl 
from scipy import signal
import pickle

#setup configuartion info with quandl 
quandl.ApiConfig.api_key = "xZbKkmrySvxot_CYSdqH" 
apl_stock=quandl.get('WIKI/AAPL', start_date="2014-01-01", end_date="2018-08-20")
msf_stock=quandl.get('WIKI/MSFT', start_date="2014-01-01", end_date="2018-08-20")
apl_open = apl_stock["Open"].values
apl_close = apl_stock["Close"].values
msf_open = msf_stock["Open"].values 
msf_close = msf_stock["Close"].values

print(msf_close)

msf_stock.head() 

plt.plot(range(0, len(msf_open)), msf_open) 
plt.plot(range(0, len(apl_open)), apl_open) 

apl_open[:108] /= 7
apl_close[:108] /= 7 

plt.plot(range(0, len(apl_open)), apl_open) 

# detrend the data so that we are not learning trends 
msf_open = signal.detrend(msf_open)
msf_close = signal.detrend(msf_close)

apl_open = signal.detrend(apl_open)
apl_close = signal.detrend(apl_close)
plt.plot(range(0, len(apl_open)), apl_open) 

print(apl_open.min())
print(apl_close.min())
print(msf_open.min())
print(msf_close.min())

# add 35 because the min values are -35 
apl_open += 35
apl_close += 35.
msf_open += 35.
msf_close += 35. 

#store this data
with open("aplmsfopenclose.pkl", "wb+") as f:
    pickle.dump({"ao":apl_open, "ac": apl_close, "mo": msf_open, "mc": msf_close}, f)

# get the data
with open("aplmsfopenclose.pkl", "rb") as f:
    d = pickle.load(f) 

apl_open = d["ao"] 
apl_close = d["ac"]
msf_open = d["mo"] 
msf_close = d["mc"]    

#test 
print(apl_close)


