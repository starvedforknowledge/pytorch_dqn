import torch

import gym
from gym.wrappers import FrameStack, AtariPreprocessing

from .model import DensePolicy, ConvPolicy
from .train import train
from .utils import generate_run_name

if __name__ == "__main__":
    num_stack = 4
    
    env_name = 'BreakoutNoFrameskip-v4'
    
    env = gym.make('BreakoutNoFrameskip-v4')
    env = AtariPreprocessing(
        env,
        terminal_on_life_loss=True
    )
    env = FrameStack(env, num_stack=num_stack)
    run_name = generate_run_name(env_name)

    policy = ConvPolicy(
        num_stacked_frames = num_stack,
        action_space_dim = env.action_space.n,
    )

    target = ConvPolicy(
        num_stacked_frames = num_stack,
        action_space_dim = env.action_space.n,
    )
    
    for param in target.parameters():
        param.requires_grad = False
        
    train(run_name, env, policy, target)