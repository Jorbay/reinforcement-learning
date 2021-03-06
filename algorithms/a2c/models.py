import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(Actor, self).__init__()

        self.num_actions = num_actions

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return policy_dist

class Critic(nn.Module):
    def __init__(self, num_inputs, hidden_size=256):
        super(Critic, self).__init__()

        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

    def forward(self,state):
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        return value