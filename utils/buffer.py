import numpy as np
import torch
from copy import deepcopy


class ReplayBuffer(object):
    def __init__(self, max_size, num_agents):
        self.max_size = int(max_size)
        self.num_agents = num_agents
        self.buffer = []
        self.current_index = 0
        self.to_gpu = False

    def get_size(self):
        """

        :return:
        """
        return len(self.buffer)

    def get_episode_rewards(self, episode_length):
        """

        :return:
        """
        temp_index = deepcopy(self.current_index)
        temp_index -= 1
        episode_reward = np.zeros(self.num_agents)
        for i in range(episode_length):
            observations, actions, rewards, next_observations, dones = self.buffer[temp_index]
            episode_reward += np.array(rewards)
            temp_index -= 1
            if temp_index == 0:
                temp_index = len(self.buffer) - 1
        return episode_reward / episode_length

    def push_data(self, observations, actions, rewards, next_observations, dones):
        """

        :param observations:
        :param actions:
        :param rewards:
        :param next_observations:
        :param dones:
        """
        data = (observations, actions, rewards, next_observations, dones)
        if self.current_index >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.current_index] = data
        self.current_index = (self.current_index + 1) % self.max_size

    def sample(self, batch_size, to_gpu=False):
        """

        :param batch_size:
        :param to_gpu:
        :return:
        """
        self.to_gpu = to_gpu
        if batch_size > len(self.buffer):
            index_of_data = self.make_index(batch_size)
        else:
            index_of_data = range(0, len(self.buffer))

        return self.generate_sample(index_of_data)

    def make_index(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        return [np.random.randint(0, len(self.buffer)) for _ in range(batch_size)]

    # def to_tensor(self, x):
    #     """
    #
    #     :param x:
    #     :return:
    #     """
    #     if self.to_gpu:
    #         tensor_x = torch.tensor(x, requires_grad=False).cuda()
    #     else:
    #         tensor_x = torch.tensor(x, requires_grad=False)
    #     return tensor_x

    def generate_sample(self, index_of_data):
        """

        :param index_of_data:
        :return:
        """
        observations_batch, actions_batch, rewards_batch, next_observations_batch, dones_batch = [], [], [], [], []
        for i in index_of_data:
            data = self.buffer[i]
            observations, actions, rewards, next_observations, dones = data
            observations_batch.append(observations)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_observations_batch.append(next_observations)
            dones_batch.append(dones)

        return observations_batch, actions_batch, rewards_batch, next_observations_batch, dones_batch
