import torch
import numpy as np
import torch.nn
from models import Actor, Critic
from plotter import Plotter
from torch.autograd import Variable
import torch.optim as optim
import gym

class A2cAgent():

    def __init__(self, env_name, learning_rate, timesteps_max, number_of_batches, discount_factor, entropy_factor,
                 minimum_batch_size):
        self.env = gym.make(env_name)
        self.env.seed(1)

        self.learning_rate = learning_rate
        self.t_max = timesteps_max
        self.number_of_batches = number_of_batches
        self.discount_factor = discount_factor
        self.entropy_factor = entropy_factor
        self.minimum_batch_size = minimum_batch_size

        self.entropy_term = 0

        self.actor_model = Actor(self.env.observation_space.shape[0], self.env.action_space.n)
        self.critic_model = Critic(self.env.observation_space.shape[0])

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate)

    def train(self):
        all_lengths = []
        all_rewards = []
        all_advantages = []

        for batch_index in range(0, self.number_of_batches):

            values = torch.FloatTensor()
            qvals = torch.FloatTensor()
            log_probs = torch.FloatTensor()
            total_steps = 0
            self.entropy_term = 0

            while (total_steps < self.minimum_batch_size):
                start_state = self.env.reset()
                trajectory_results = self.iterate_through_single_trajectory(
                    start_state)
                current_qvals = self.get_qvals(trajectory_results)

                values = torch.cat((torch.cat(trajectory_results.values), values), 0)
                qvals = torch.cat((torch.FloatTensor(current_qvals), qvals), 0)
                log_probs = torch.cat((torch.stack(trajectory_results.log_probs), log_probs), 0)

                number_of_steps = len(trajectory_results.values)

                total_steps = total_steps + number_of_steps
                all_lengths.append(number_of_steps)
                all_rewards.append(sum(trajectory_results.rewards))
                all_advantages.append(sum(current_qvals - torch.cat(trajectory_results.values)))

            self.update(qvals, values, log_probs)

        plotter = Plotter()
        plotter.add_variable(all_lengths, "length of trajectories")
        plotter.add_variable(all_rewards, "total rewards of trajectories")
        plotter.add_variable(all_advantages, "total advantage function values of trajectories")
        plotter.plot()


    def update(self, qvals, values, log_probs):

        advantage = qvals - values
        actor_loss = (-log_probs * advantage.detach()).mean() - self.entropy_term*self.entropy_factor
        critic_loss = advantage.pow(2).mean()

        #print("The current entropy term is ")
        #print(self.entropy_term)

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
        q_value = 0
        done = False

        current_state = start_state
        for step_counter in range(0,self.t_max):
            action, policy_dist, value = self.get_models_output(current_state)
            current_state, reward, done, _ = self.env.step(action)

            policy_dist_detached = policy_dist.detach()
            self.entropy_term = -torch.sum(torch.mean(policy_dist_detached) * torch.log(policy_dist_detached)) \
                                + self.entropy_term

            rewards.append(reward)
            values.append(value.squeeze(0))
            log_probs.append(torch.log(policy_dist.squeeze(0)[action]))

            if (done):
                break

        if (not done):
            _, _, q_value = self.get_models_output(current_state)

        return A2cAgent.TrajectoryResults(values, rewards, log_probs, q_value, done)

    def get_action(self, policy_dist):
        return np.random.choice(self.env.action_space.n, p=policy_dist.numpy().squeeze(0))

    def get_models_output(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        policy_dist = self.actor_model.forward(state)
        value = self.critic_model.forward(state)

        return self.get_action(policy_dist), policy_dist, value

    def get_qvals(self,trajectory_results):
        trajectory_timesteps = len(trajectory_results.values)
        qvals = torch.zeros(trajectory_timesteps)
        q_value = trajectory_results.q_value

        for j in range(trajectory_timesteps - 1, -1, -1):
            q_value = q_value * self.discount_factor + trajectory_results.rewards[j]
            qvals[j] = q_value
        return qvals

    class TrajectoryResults():
        def __init__(self, values, rewards, log_probs, q_value, done):
            self.values = values
            self.rewards = rewards
            self.log_probs = log_probs
            self.q_value = q_value
            self.done = done

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v0')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--timesteps_max', type=int, default=300)
    parser.add_argument('--number_of_batches', type=int, default=200)
    parser.add_argument('--discount_factor', type=float, default=0.1)
    parser.add_argument('--entropy_factor', type=float, default = 0.001)
    parser.add_argument('--minimum_batch_size', type=int, default = 200)

    args = parser.parse_args()

    A2cAgent(args.env_name, args.learning_rate, args.timesteps_max, args.number_of_batches, args.discount_factor,
             args.entropy_factor, args.minimum_batch_size).train()