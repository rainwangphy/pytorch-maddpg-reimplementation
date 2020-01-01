"""
@author
"""

from gym.spaces import Box
from utils.agent import DDPGAgent
from utils.misc import onehot_from_logits, gumbel_softmax, soft_update
import torch

MSELoss = torch.nn.MSELoss()


class MADDPG(object):
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
        self.num_agent = len(alg_types)
        self.alg_types = alg_types
        self.gamma = gamma
        self.tau = tau
        self.lr = lr

        self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]

        self.agent_init_params = agent_init_params
        self.discrete_action = discrete_action

        self.hidden_dim = hidden_dim

        self.num_iteration = 0

    def step(self, observations, explore=False):
        """

        :param observations: observations of agent
        :param explore: whether the agents explore or not
        """
        return [a.step(observation, explore=explore) for a, observation in
                zip(self.agents, observations)]

    @classmethod
    def init_from_env(cls, env, agent_types, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """

        :param agent_types:
        :param env:
        :param agent_alg:
        :param adversary_alg:
        :param gamma:
        :param tau:
        :param lr:
        :param hidden_dim:
        :return:
        """
        alg_types = [adversary_alg if agent_type == 'adversary' else agent_alg for
                     agent_type in agent_types]

        agent_init_parameters = []

        for action_space, observation_space, algtype in zip(env.action_space,
                                                             env.observation_space,
                                                             alg_types):
            number_input_actor = observation_space.shape[0]

            if isinstance(action_space, Box):
                discrete_action = False
                number_output_actor = action_space.shape[0]
            else:
                discrete_action = True
                number_output_actor = action_space.n

            if algtype == "MADDPG":
                number_input_critic = 0
                for obs_space in env.observation_space:
                    number_input_critic += obs_space.shape[0]
                for act_space in env.action_space:
                    if isinstance(act_space, Box):
                        number_input_critic += act_space.shape[0]
                    else:
                        number_input_critic += act_space.n
            else:
                number_input_critic = observation_space.shape[0] + number_output_actor

            agent_init_parameters.append(
                {
                    'number_input_actor': number_input_actor,
                    'number_output_actor': number_output_actor,
                    'number_input_critic': number_input_critic
                }
            )

        init_dict = {
            'gamma': gamma, 'tau': tau, 'lr': lr,
            'hidden_dim': hidden_dim,
            'alg_types': alg_types,
            'agent_init_params': agent_init_parameters,
            'discrete_action': discrete_action
        }

        instance = cls(**init_dict)
        instance.init_dict = init_dict

        return instance

    def actors(self):
        """

        :return:
        """
        return [self.agents[i].actor for i in range(self.num_agent)]

    def target_actors(self):
        """

        :return:
        """
        return [self.agents[i].target_actor for i in range(self.num_agent)]

    def update(self, sample, agent_i):
        """
        TODO: to make the sample into valid inputs
        :param sample:
        :param agent_i:
        :return:
        """
        obs, acs, rews, next_obs, dones = sample
        current_agent = self.agents[agent_i]

        current_agent.critic_optimizer.zero_grad()
        if self.alg_types == 'MADDPG':
            all_target_actors = self.target_actors()
            if self.discrete_action:
                all_target_actions = [onehot_from_logits(pi(n_obs)) for pi, n_obs in
                                      zip(all_target_actors, next_obs)]
            else:
                all_target_actions = [pi(n_obs) for pi, n_obs in zip(all_target_actors, next_obs)]

            target_critic_input = torch.cat((*next_obs, *all_target_actions), dim=1)  # TODO need to be refined
        else:
            if self.discrete_action:
                target_critic_input = torch.cat((next_obs[agent_i], onehot_from_logits(
                    current_agent.target_actor(next_obs[agent_i]))), dim=1)
            else:
                target_critic_input = torch.cat((next_obs[agent_i],
                                                 current_agent.target_actor(next_obs[agent_i])), dim=1)
        target_critic_value = current_agent.target_critic(target_critic_input)
        target_value = rews[agent_i].view(-1, 1) + self.gamma * target_critic_value * (1-dones[agent_i]).view(-1, 1)
        if self.alg_types[agent_i] == 'MADDPG':
            critic_input = torch.cat((*obs, *acs), dim=1)  # TODO need to be refined
        else:  # DDPG
            critic_input = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        actual_value = current_agent.critic(critic_input)

        critic_loss = MSELoss(actual_value, target_value.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(current_agent.critic.parameters(), 0.5)
        current_agent.critic_optimizer.step()

        current_agent.actor_optimizer.zero_grad()
        if self.discrete_action:
            current_action_out = current_agent.actor[obs[agent_i]]
            current_action_input_critic = gumbel_softmax(current_action_out, hard=True)
        else:
            current_action_out = current_agent.actor[obs[agent_i]]
            current_action_input_critic = current_action_out

        if self.alg_types[agent_i] == 'MADDPG':
            all_actor_action = []
            all_target_actors = self.target_actors()
            for i, pi, ob in zip(range(self.num_agent), all_target_actors, obs):
                if i == agent_i:
                    all_actor_action.append(current_action_input_critic)
                else:
                    if self.discrete_action:
                        all_actor_action.append(onehot_from_logits(all_target_actors[i](ob)))
                    else:
                        all_actor_action.append(all_target_actors[i](ob))
            critic_input = torch.cat([obs, all_actor_action], dim=1)
        else:
            critic_input = torch.cat([obs[agent_i], current_action_input_critic], dim=1)

        actor_loss = -current_agent.critic(critic_input).mean()
        actor_loss += (current_action_out**2).mean() * 1e-3
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm(current_agent.actor.parameters(), 0.5)
        current_agent.actor_optimizer.step()

    def update_all_agent(self):
        """
        soft update all agent
        """
        for a in self.agents:
            soft_update(a.target_actor, a.actor, self.tau)
            soft_update(a.target_critic, a.critic, self.tau)
        self.num_iteration += 1


