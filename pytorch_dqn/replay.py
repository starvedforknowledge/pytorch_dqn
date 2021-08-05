import torch
import numpy as np
import collections

Experience = collections.namedtuple('Experience', 
                                    field_names=['state',
                                                 'action',
                                                 'reward',
                                                 'done',
                                                 'new_state'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            replace=False
        )
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )
        
        sample = {
            "states":torch.FloatTensor(states),
            "actions":torch.LongTensor(actions).view(-1, 1),
            "rewards":torch.FloatTensor(rewards),
            "dones":torch.FloatTensor(dones),
            "next_states":torch.FloatTensor(next_states)
        }
        return sample