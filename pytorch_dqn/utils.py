from datetime import datetime
import pytz

import torch
import numpy as np

def generate_run_name(env_name):
    date_string = datetime.strftime(
        datetime.now(pytz.timezone('America/Edmonton')),
        '%Y%m%d_T%H%M'
    )
    
    run_name = env_name + '_' + date_string
    return run_name

def test_target_network(
    env,
    target,
    max_steps,
    device,
    num_episodes=100,
    test_epsilon=0.005
):
    episode_rewards = []
    for _ in range(num_episodes):   
        state = env.reset()
        episode_reward = 0
        i = 0
        while True:
            with torch.no_grad():
                x = torch.FloatTensor([state]).to(device)
                Q = target(x)
                
                if np.random.rand(1) <= test_epsilon:
                    action = env.action_space.sample()
                else:
                    _, action = torch.max(Q, -1)
                    action = action.item()
                state, reward, done, info = env.step(action)   
                episode_reward += reward
                
                if i >= max_steps:
                    done = True
                    
                if done: 
                    episode_rewards.append(episode_reward)
                    break

    return np.mean(episode_rewards)   