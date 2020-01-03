import argparse

import numpy as np
import torch

from algorithm.maddpg import MADDPG
from utils.buffer import ReplayBuffer
# from pathlib import Path
from utils.make_env import make_env


def run(config):
    """

    :param config:
    """
    # model_dir = Path('./models') / config.env_id / config.model_name
    env = make_env(config.env_id)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if all([hasattr(a, 'adversary') for a in env.agents]):
        agent_types = ['adversary' if a.adversary else 'agent' for a in env.agents]
    else:
        agent_types = ['agent' for _ in env.agents]

    maddpg = MADDPG.init_from_env(env, agent_types, agent_alg=config.agent_alg, adversary_alg=config.adversary_alg,
                                  tau=config.tau, lr=config.lr, hidden_dim=config.hidden_dim)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.num_agent)

    for ep_i in range(config.n_episodes):
        print("Episodes %i of %i" % (ep_i + 1, config.n_episodes))
        observations = env.reset()

        for et_i in range(config.episode_length):
            torch_observations = [torch.from_numpy(observations[i]).float() for i in range(maddpg.num_agent)]
            torch_agent_actions = maddpg.step(torch_observations)
            agent_actions = [action.data.numpy() for action in torch_agent_actions]
            next_observations, rewards, dones, infos = env.step(agent_actions)

            replay_buffer.push_data(observations, agent_actions, rewards, next_observations, dones)

            observations = next_observations

            if replay_buffer.get_size() >= config.batch_size:
                for a_i in range(maddpg.num_agent):
                    sample = replay_buffer.sample(config.batch_size)
                    maddpg.update(sample, agent_i=a_i)
                maddpg.update_all_agent()
        print("Episode rewards ")
        print(replay_buffer.get_episode_rewards(config.episode_length))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="simple_tag", help="Name of environment")
    parser.add_argument("--model_name", default="simple_tag",
                        help="Directory to store the training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=48, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()

    run(config)
