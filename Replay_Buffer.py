from collections import namedtuple
import torch 
import random

# transition class a
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done')) 

class Replay_Buffer(): 
    def __init__(self, capacity): 
        self.storage = [] 
        self.capacity = capacity
        self.position = 0

    def __len__(self): 
        return len(self.storage) 

    def push(self, *args): 
        # save a tranisiton
        if len(self.storage) < self.capacity: 
            self.storage.append(None)   
        self.storage[self.position] = Transition(*args)
        # circular buffer
        self.position = (self.position + 1) % self.capacity 

    # get a random sample from the dataset of transition s
    def sample(self, batch_size):
        return random.sample(self.storage, batch_size)
"""
if __name__ == "__main__": 
    replay_buffer = Replay_Buffer(100)
    print("length:", replay_buffer.__len__())
    #replay_buffer.push(1, 1, 1, 1, 0)
    replay_buffer.push(2, 1, 1, 1, 1) 
    print("Storage:", replay_buffer.storage)
    print("Sample dataset", replay_buffer.sample(1)) 
"""

    
    
        