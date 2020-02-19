import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import torch.nn
from .models import ActorCritic
from torch.autograd import Variable
import torch.optim as optim

class A2cAgent():

    theta = None
    theta_v = None

    step_counter = 0

    def __init__(self, env, learning_rate = 3e-4, timesteps_max = 10, trajectories_max = 5, discount_factor = .1):
        #initiate theta and theta_v for policy and value function respectively
        self.env = env
        self.learning_rate = learning_rate
        self.t_max = timesteps_max
        self.T_max = trajectories_max
        self.discount_factor = discount_factor

        self.ac_model = ActorCritic(self.env.observation_space.shape[0], self.env.action_space.n)
        self.ac_optimizer = optim.Adam(self.ac_model.parameters(), lr = self.learning_rate)

    def train(self):
        for trajectory_index in range(0,self.T_max):
            self.step_counter = 0

            d_theta = 0
            d_theta_v = 0
            start_state = self.env.reset() # Should get start state from some initial state distribution
            values, actions, rewards, log_probs, terminated, reward_at_time = self.collect_states_and_rewards(start_state)

            # everything below should probably be in an "update" function
            for j in range(self.step_counter-1,0, -1):
                reward_at_time = reward_at_time*self.discount_factor + rewards[j]

                d_theta = self.get_updated_delta_theta(d_theta, values[j], actions[j], log_probs[j], reward_at_time)
                d_theta_v = self.get_updated_delta_theta_v(d_theta_v, values[j], reward_at_time)

            #TODO: insert logic for updating theta and theta_v with d_theta and d_theta_v




    def get_initial_state(self, isd):
        pass

    def get_value(self, state):
        return self.get_value_function(state).sample()

    def collect_states_and_rewards(self, start_state):
        values = []
        actions = []
        rewards = []
        log_probs = []
        end_value = 0
        done = False

        current_state = start_state
        while (self.step_counter < self.t_max):
            self.step_counter = self.step_counter + 1

            action, policy_dist, value = self.get_models_output(current_state)

            current_state, reward, done, _ = self.env.step(action)

            rewards.append(reward)
            values.append(value)
            actions.append(action)
            log_probs.append(torch.log(policy_dist.squeenze(0)[action]))

            if (done):
                break

        if (not done):
            _, _, end_value = self.get_models_output(current_state)

        return values, actions, rewards, log_probs, done, end_value

    def step(self):
        pass
        #this will probably just be the step function for whatever environment we're operating within

    def get_action(self, policy_dist):
        return np.random.choice(self.num_out, p=policy_dist.detach().numpy().squeeze(0))

    def get_models_output(self,state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value, policy_dist = self.ac_model.forward(state)

        return self.get_action(policy_dist), policy_dist, value

    def get_policy(self,state):
        pass

    def get_value_function(self,state):
        pass

    def get_updated_delta_theta(self, delta_theta, value, action, log_prob, reward):
        pass

    def get_updated_delta_theta_v(self, delta_v_theta, value, reward):
        pass

    #copied straight out of spinning up
    def mlp(self, sizes=3, activation=nn.Tanh, output_activation=nn.Identity):
        # Build a feedforward neural network.
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        return nn.Sequential(*layers)
