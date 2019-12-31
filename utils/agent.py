from .network import MLPNetwork
from torch.optim import Adam
from .noise import OUNoise
from .misc import gumbel_softmax, onehot_from_logits, hard_update
import torch


class DDPGAgent(object):
    def __init__(self, number_input_actor, number_output_actor, number_input_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        self.actor = MLPNetwork(number_input_actor, number_output_actor)
        self.target_actor = MLPNetwork(number_input_actor, number_output_actor)
        self.critic = MLPNetwork(number_input_critic, 1)
        self.target_critic = MLPNetwork(number_input_critic, 1)
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.lr = lr

        if not discrete_action:
            self.exploration = OUNoise(number_output_actor)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def step(self, observation, explore=False):
        """

        :param observation:
        :param explore:
        """
        action = self.actor(observation)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += torch.from_numpy(self.exploration.noise())
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        """

        :return:
        """
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'policy_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }