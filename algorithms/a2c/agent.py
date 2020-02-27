import torch
import numpy as np
import torch.nn
from models import Actor, Critic
from plotter import Plotter
import torch.optim as optim
import gym
import time

class A2cAgent():

    def __init__(self, env_name, learning_rate, timesteps_max, number_of_batches, discount_factor, entropy_constant,
                 minimum_batch_size):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(env_name)
        self.env.seed(1)

        self.learning_rate = learning_rate
        self.t_max = timesteps_max
        self.number_of_batches = number_of_batches
        self.discount_factor = discount_factor
        self.entropy_constant = entropy_constant
        self.minimum_batch_size = minimum_batch_size
        self.log_interval = 100

        self.actor_model = Actor(self.env.observation_space.shape[0], self.env.action_space.n)
        self.critic_model = Critic(self.env.observation_space.shape[0])

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)

    def train(self):
        rewards_buffer = []
        sampled_total_rewards = []
        sampled_episode_lengths = []
        sampled_advantages = []
        start_time = time.time()

        for batch_index in range(0, self.number_of_batches):

            self.entropy_term = 0

            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.get_rollout()

            log_probs = self.get_log_probabilities_of_actions(batch_states, batch_actions).to(self.device)
            values = self.get_values(batch_states).to(self.device)
            value_targets = self.get_value_targets(batch_next_states, batch_rewards, batch_dones).to(self.device)

            self.update(value_targets, values, log_probs)


            if (batch_index % self.log_interval == 0):
                if (True in batch_dones):
                    first_done = batch_dones.index(True)
                else:
                    first_done = len(batch_dones)

                single_trajectory_rewards = batch_rewards[:first_done]

                total_advantage = torch.mean(value_targets - values).item()

                timesteps_since_beginning = self.t_max*(1+batch_index)
                time_since_beginning = time.time() - start_time
                frames_per_second = timesteps_since_beginning / time_since_beginning


                print("At " + str(batch_index) + "th episode, the last 100 episodes had an average total return of ")
                print(sum(single_trajectory_rewards))
                print("Length of episode was", first_done)
                print("Mean of advantage in episode was", total_advantage)
                print("Timesteps since starting is ", timesteps_since_beginning)
                print("frames per second is ", frames_per_second)


                sampled_total_rewards.append(sum(single_trajectory_rewards))
                sampled_episode_lengths.append(first_done)
                sampled_advantages.append(total_advantage)

                rewards_buffer = []



        plotter = Plotter()
        plotter.add_variable(sampled_total_rewards, "total reward every 100 trajectories")
        plotter.add_variable(sampled_episode_lengths, "length every 100 trajectories")
        plotter.add_variable(sampled_advantages, "average advantage of episode every 100 trajectories")
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


    def get_rollout(self):
        #TODO: implement multi-worker version of this
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []


        current_state = self.env.reset()

        for step_counter in range(0, self.t_max):
            action = self.get_action_from_actor(current_state)
            next_state, reward, done, _ = self.env.step(action)

            states.append(current_state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            if (done):
                current_state = self.env.reset()
            else:
                current_state = next_state

        return states, actions, rewards, next_states, dones


    def get_action(self, policy_dist):
        return np.random.choice(self.env.action_space.n, p=policy_dist.cpu().detach().numpy().squeeze(0))

    def get_actor_output(self, states):
        states = torch.FloatTensor(states)
        policy_dists = self.actor_model.forward(states)
        return policy_dists

    def get_action_from_actor(self, state):
        return self.get_action(self.get_actor_output(np.expand_dims(state, axis=0)))

    def get_critic_output(self, states):
        states = torch.FloatTensor(states)
        value = self.critic_model.forward(states)

        return value

    def get_value_targets(self, next_states, rewards, dones):
        number_of_timesteps = len(next_states)
        value_targets = torch.zeros(number_of_timesteps)

        previous_reward = self.get_critic_output(np.expand_dims(next_states[-1], axis=0))
        for j in range(number_of_timesteps - 1, -1, -1):
            if (dones[j]):
                previous_reward = 0

            current_reward = rewards[j]
            value_targets[j] = previous_reward * self.discount_factor + current_reward

            previous_reward = current_reward

        return value_targets

    def get_log_probabilities_of_actions(self, states, actions):

        policy_distributions = self.get_actor_output(states)

        actions_tensor = torch.LongTensor(list(map(lambda el:[el], actions)))
        action_probabilities = torch.gather(policy_distributions, 1, actions_tensor)
        log_probs = torch.log(action_probabilities)

        return log_probs

    def get_values(self, states):
        return self.get_critic_output(states)
