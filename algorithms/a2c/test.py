import gym
from .agent import A2cAgent


env = gym.make("CartPole-v0")
A2cAgent(env).train()
