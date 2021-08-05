import json
import os.path as osp
from pathlib import Path
import collections

import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim

from .schedule import PiecewiseLinearEpsilon
from .replay import Experience, ExperienceReplay
from .utils import test_target_network

# https://github.com/openai/gym/blob/334491803859eaa5a845f5f5def5b14c108fd3a9/gym/envs/__init__.py#L787
# Shows the difference between atari envs.
    
def train(
    run_name,
    env,
    policy,
    target,
    max_steps_per_episode=10000,
    num_episodes=10000000,
    replay_capacity=1000000,
    start_learning=50000,
    batch_size=32,
    learning_rate=0.0000625,
    learning_step_freq=4,
    target_sync_freq=10000,
    testing_freq=10000,
    gamma=0.99
):

    Path(run_name).mkdir(parents=True, exist_ok=False)
    
    replay = ExperienceReplay(replay_capacity)
    epsilon_schedule = PiecewiseLinearEpsilon(frames=[0, 1e6, 24e6],
                                              epsilons=[1.0, 0.1, 0.01])

    env.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)


    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print('Training on {}'.format(device))
        
    policy = policy.to(device)
    target = target.to(device)

    # Huber Loss to prevent large gradients when errors are large.
    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    train_episode_rewards = collections.deque(maxlen=100)
    max_mean_reward = np.NINF
    frame_idx = 0

    ready_to_test = False
    
    # Experience Loop
    for episode in trange(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            # Convert state to tensor adding a batch dimension.
            x = torch.FloatTensor([state]).to(device)
            # Get Q-values from this state.
            with torch.no_grad():
                Q = policy(x)

            # Decide on action with epsilon greedy strategy. 
            # Picks either a random or greedy action with probability
            # defined by the epsilon schedule.
            if np.random.rand() <= epsilon_schedule.get_epsilon(frame_idx):
                action = env.action_space.sample()
            else:
                _, action = torch.max(Q, -1)
                action = action.item()
                
            # Get result of action (new state, reward)
            new_state, reward, done, info = env.step(action)

            # Add experience to experience replay.
            experience = Experience(state=state,
                                    action=action,
                                    reward=reward,
                                    done=done,
                                    new_state=new_state)

            replay.append(experience)
            episode_reward += reward

            # Learning Step
            if len(replay) >= start_learning:
                frame_idx += 1

                if frame_idx%learning_step_freq==0:

                    # Sample experience replay.
                    batch = replay.sample(batch_size)

                    # Preprocess batch states.
                    x = batch['states'].to(device)
                    xprime = batch['next_states'].to(device)

                    a = batch['actions'].to(device)
                    
                    # Get the Q-values associated with the 
                    # actions in "a" using gather.
                    Q = policy(x)
                    y_pred = Q.gather(1, a)
                    
                    # Get the "best-guess" Q-values
                    # from the target network. We
                    # use these to inform the "targets"
                    with torch.no_grad():
                        Qprime = target(xprime)

                    y_true = batch['rewards'].to(device)
                    Qmax, _ = torch.max(Qprime, 1)
                    not_done = 1 - batch['dones'].to(device)
                    
                    # y_true = rewards + gamma x Q (bellman equation).
                    y_true += gamma*not_done*Qmax

                    loss = loss_fn(y_pred, y_true.view(-1, 1).detach())

                    policy.zero_grad()
                    loss.backward()
                    optimizer.step()

                if frame_idx%target_sync_freq == 0:
                    # Synchronize target network weights.
                    target.load_state_dict(policy.state_dict())
                    mean_reward=np.mean(train_episode_rewards)

                    log = {"frames":frame_idx,
                           "mean_reward":mean_reward,
                           "epsilon": epsilon_schedule.get_epsilon(frame_idx),
                           "loss": loss.item()}

                    with open(osp.join(run_name,'train_log.jsonl'),'a') as f:
                        f.write(json.dumps(log)+'\n')

                    print("frames:{}, mean_train_reward:{:.2f}, epsilon:{:.2f}, replay_size:{}".format(
                        log["frames"],
                        log["mean_reward"],
                        log["epsilon"],
                        len(replay)
                    ))

                if frame_idx%testing_freq == 0:
                    ready_to_test = True

                if ready_to_test and done:
                    mean_reward = test_target_network(
                        env,
                        target,
                        max_steps_per_episode,
                        device
                    )
                    torch.save(target.state_dict(),
                               osp.join(run_name,'policy.pt'))

                    if mean_reward > max_mean_reward:
                        max_mean_reward = mean_reward
                        torch.save(
                            target.state_dict(),
                            osp.join(run_name,'best-policy.pt')
                        )

                    log = {"frames":frame_idx,
                           "mean_reward":mean_reward,
                           "epsilon": epsilon_schedule.get_epsilon(frame_idx)}

                    with open(osp.join(run_name,'test_log.jsonl'),'a') as f:
                        f.write(json.dumps(log)+'\n')

                    print("frames:{}, mean_test_reward:{:.2f}, epsilon:{:.2f}".format(
                        log["frames"],
                        log["mean_reward"],
                        log["epsilon"]
                    ))
                    ready_to_test=False

            if done:
                train_episode_rewards.append(episode_reward)
                break
            else:
                state = new_state
