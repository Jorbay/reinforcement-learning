import torch
import numpy as np
import torch.nn
from .models import Actor, Critic
from torch.autograd import Variable
import torch.optim as optim


class A2cAgent():

    def __init__(self, env, learning_rate=3e-4, timesteps_max=300, trajectories_max=1500, discount_factor=.1):
        # initiate theta and theta_v for policy and value function respectively
        self.env = env
        self.learning_rate = learning_rate
        self.t_max = timesteps_max
        self.T_max = trajectories_max
        self.discount_factor = discount_factor

        self.actor_model = Actor(self.env.observation_space.shape[0], self.env.action_space.n)
        self.critic_model = Critic(self.env.observation_space.shape[0])

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)

    def train(self):
        for trajectory_index in range(0, self.T_max):
            start_state = self.env.reset()  # Should get start state from some initial state distribution
            trajectory_results = self.iterate_through_single_trajectory(
                start_state)
            qvals = self.get_qvals(trajectory_results)



            #TODO: break down into update method

            values = torch.cat(trajectory_results.values)
            qvals = torch.FloatTensor(qvals)
            log_probs = torch.stack(trajectory_results.log_probs)

            advantage = qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = advantage.pow(2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def iterate_through_single_trajectory(self, start_state):
        values = []
        rewards = []
        log_probs = []
        q_value = 0
        done = False

        current_state = start_state
        for step_counter in range(0,self.t_max):
            action, policy_dist, value = self.get_models_output(current_state)

            current_state, reward, done, _ = self.env.step(action)
            self.env.render()

            rewards.append(reward)
            values.append(value.squeeze(0))
            log_probs.append(torch.log(policy_dist.squeeze(0)[action]))

            if (done):
                break

        if (not done):
            _, _, q_value = self.get_models_output(current_state)

        return A2cAgent.TrajectoryResults(values, rewards, log_probs, q_value, done)

    def get_action(self, policy_dist):
        return np.random.choice(self.env.action_space.n, p=policy_dist.detach().numpy().squeeze(0))

    def get_models_output(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        policy_dist = self.actor_model.forward(state)
        value = self.critic_model.forward(state)

        return self.get_action(policy_dist), policy_dist, value

    def get_qvals(self,trajectory_results):
        trajectory_timesteps = len(trajectory_results.values)
        qvals = torch.zeros(trajectory_timesteps)

        for j in range(trajectory_timesteps - 1, 0, -1):
            q_value = trajectory_results.q_value * self.discount_factor + trajectory_results.rewards[j]
            qvals[j] = q_value
        return qvals

    class TrajectoryResults():
        def __init__(self, values, rewards, log_probs, q_value, done):
            self.values = values
            self.rewards = rewards
            self.log_probs = log_probs
            self.q_value = q_value
            self.done = done