from enum import Enum, IntEnum

import torch
import numpy as np
import torch.nn
from models import Actor, Critic
from plotter import Plotter
from torch.autograd import Variable
import torch.optim as optim
import gym
import statistics

class A2cAgent():

    def __init__(self, env_name, learning_rate, timesteps_max, number_of_batches, discount_factor, entropy_constant,
                 minimum_batch_size):
        self.env = gym.make(env_name)
        self.env.seed(1)

        self.learning_rate = learning_rate
        self.t_max = timesteps_max
        self.number_of_batches = number_of_batches
        self.discount_factor = discount_factor
        self.entropy_constant = entropy_constant
        self.minimum_batch_size = minimum_batch_size

        self.actor_model = Actor(self.env.observation_space.shape[0], self.env.action_space.n)
        self.critic_model = Critic(self.env.observation_space.shape[0])

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)

    def train(self):
        rewards_buffer = []
        average_rewards = []

        for batch_index in range(0, self.number_of_batches):

            values = torch.FloatTensor()
            value_targets = torch.FloatTensor()
            log_probs = torch.FloatTensor()
            total_steps = 0
            self.entropy_term = 0

            rollout = self.get_rollout()

            # get log probabilities
            log_probs = self.get_log_probs(rollout)

            # get value function output

            # get value targets
            value_targets = self.get_value_targets(rollout)
            
            '''
            
            
            

            start_state = self.env.reset()
            trajectory_results = self.iterate_through_single_trajectory(
                start_state)
            current_value_targets = self.get_value_targets(trajectory_results)

            values = torch.cat((torch.cat(trajectory_results.values), values), 0)
            value_targets = torch.cat((torch.FloatTensor(current_value_targets), value_targets), 0)
            log_probs = torch.cat((torch.stack(trajectory_results.log_probs), log_probs), 0)

            number_of_steps = len(trajectory_results.values)

            total_steps = total_steps + number_of_steps
            #all_lengths.append(number_of_steps)
            #all_rewards.append(sum(trajectory_results.rewards))
            #all_advantages.append(sum(current_value_targets - torch.cat(trajectory_results.values)))
            rewards_buffer.append(sum(trajectory_results.rewards))

            if (batch_index % 100 == 0):
                average_rewards.append(statistics.mean(rewards_buffer))
                print("At " + str(batch_index) + "th episode, the last 100 episodes had an average total return of ")
                print(average_rewards[-1])
                rewards_buffer = []


            self.update(value_targets, values, log_probs)
            
            '''

        plotter = Plotter()
        plotter.add_variable(average_rewards, "average reward every 100 trajectories")
        plotter.plot()


    def update(self, value_targets, values, log_probs):

        advantage = value_targets - values
        actor_loss = -(log_probs * advantage.detach()).sum() - self.entropy_term*self.entropy_constant
        critic_loss = advantage.pow(2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def iterate_through_single_trajectory(self, start_state):
        values = []
        rewards = []
        log_probs = []
        final_return = 0
        done = False

        current_state = start_state
        for step_counter in range(0,self.t_max):
            action, policy_dist, value = self.get_models_output(current_state)
            current_state, reward, done, _ = self.env.step(action)

            rewards.append(reward)
            values.append(value.squeeze(0))
            log_probs.append(torch.log(policy_dist.squeeze(0)[action]))

            if (done):
                break

        if (not done):
            _, _, final_return = self.get_models_output(current_state)

        return A2cAgent.TrajectoryResults(values, rewards, log_probs, final_return, done)

    def get_rollout(self):
        #TODO: implement multi-worker version of this
        rollout = []
        current_state = self.env.reset()
        for step_counter in range(0, self.t_max):
            action = self.get_action_from_actor(current_state)
            next_state, reward, done, _ = self.env.step(action)

            rollout.append([current_state, action, reward, next_state, done])

            if (done):
                current_state = self.env.reset()
            else:
                current_state = next_state


    def get_action(self, policy_dist):
        return np.random.choice(self.env.action_space.n, p=policy_dist.detach().numpy().squeeze(0))

    def get_models_output(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        policy_dist = self.actor_model.forward(state)
        value = self.critic_model.forward(state)

        return self.get_action(policy_dist), policy_dist, value

    def get_actor_output(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        policy_dist = self.actor_model.forward(state)
        return policy_dist

    def get_action_from_actor(self, state):
        return self.get_action(self.get_actor_output(state))

    def get_critic_output(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = self.critic_model.forward(state)

        return value


    def get_value_targets(self, rollout):
        number_of_timesteps = len(rollout)
        value_targets = torch.zeros(number_of_timesteps)

        previous_reward = 0
        for j in range(number_of_timesteps - 1, -1, -1):
            current_rollout = rollout[j]
            if (current_rollout[int(self.RolloutIteration.DONE)]):
                previous_reward = self.get_critic_output(current_rollout[int(self.RolloutIteration.STATE)])

            current_reward = current_rollout[int(self.RolloutIteration.REWARD)]
            value_targets[j] = previous_reward * self.discount_factor + current_reward

            previous_reward = current_reward

        return value_targets

    def get_log_probs(self, rollout):
        number_of_timesteps = len(rollout)
        log_probs = []

        for j in range(0, number_of_timesteps):
            policy_distribution = self.get_actor_output(rollout[j][int(self.RolloutIteration.STATE)])
            log_probs[j] = torch.log(policy_distribution.squeeze(0)[int(self.RolloutIteration.ACTION)])

        return log_probs






    class TrajectoryResults():
        def __init__(self, values, rewards, log_probs, final_return, done):
            self.values = values
            self.rewards = rewards
            self.log_probs = log_probs
            self.final_return = final_return
            self.done = done

    class RolloutIteration(IntEnum):
        STATE = 0
        ACTION = 1
        REWARD = 2
        NEXT_STATE = 3
        DONE = 4