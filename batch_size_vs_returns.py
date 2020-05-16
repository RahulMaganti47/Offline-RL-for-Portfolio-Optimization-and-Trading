import matplotlib.pyplot as plt

hfont = {'fontname':'Gill Sans MT'} 

batch_size_things = [20, 40, 60, 100, 120, 140, 200, 250]
returns = [14.59, 16.29, 18.88, 17.63, 19.35, 21.13, 23.01, 22.87]
plt.scatter(batch_size_things, returns)
plt.title("Batch Size vs. Returns", **hfont)
plt.xlabel("BatchSize(Transitions)", **hfont)
plt.ylabel("Returns(%)", **hfont)
plt.grid(b=True, which='major', color='#666666', linestyle='-') 
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show() 