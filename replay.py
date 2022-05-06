import argparse
import json
import os
import pathlib
from pathlib import Path

import gym
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from TD3_BC import TD3_BC
from wrappers import make_env


def prefill_memory(obses, capacity, obs_shape):
    """Reserves memory for replay buffer"""
    c, h, w = obs_shape
    for _ in range(capacity):
        frame = np.ones((3, h, w), dtype=np.uint8)
        obses.append(frame)
    return obses


class ReplayBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, prefill=True):
        self.capacity = capacity
        self.batch_size = batch_size

        self._obses = []
        if prefill:
            self._obses = prefill_memory(self._obses, capacity, obs_shape)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        obses = (obs, next_obs)
        if self.idx >= len(self._obses):
            self._obses.append(obses)
        else:
            self._obses[self.idx] = (obses)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


class Experience:
    def __init__(self, img_path: str, action: float, reward: float):
        self.img_path = img_path
        self.action = action
        self.reward = reward


def run_replay(env: gym.Env, agent: TD3_BC, replay_dir: str, num_episodes: int, episode_length: int):
    replay_buffer = ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=episode_length,
        batch_size=args.batch_size
    )
    data = []
    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        done = False
        episode_step = 0
        episode_path = Path(os.path.join(replay_dir, f'episode_{episode}'))
        episode_path.mkdir(parents=True, exist_ok=True)
        image_paths = []
        while not done:
            action = agent.select_action(obs)
            # frame = env.render(mode='rgb_array', height=img_size, width=img_size)
            frame = env.render(mode='rgb_array')
            next_obs, reward, done, _ = env.step(action)
            done_bool = 0 if episode_step + 1 == episode_length else float(done)
            replay_buffer.add(obs, action, reward, next_obs, done_bool)
            obs = next_obs

            image_path = f'{episode_path}/{episode_step}.png'
            image = Image.fromarray(frame)
            image.save(image_path)
            image_paths.append(f'episode_{episode}/{episode_step}.png')
            episode_step += 1

        episode_data = list(map(lambda x: Experience(*x).__dict__,
                                zip(image_paths, replay_buffer.actions.squeeze().tolist(),
                                    replay_buffer.rewards.squeeze().tolist())))
        data.append(episode_data)

    with open(f'{replay_dir}/data.json', 'w') as file:
        json.dump(data, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name", default="reacher")  # DeepMind Control Suite domain name
    parser.add_argument("--task_name", default="easy")  # DeepMind Control Suite task name
    parser.add_argument('--replay_episodes', default=500, type=int)  # Number of episodes to record
    parser.add_argument('--episode_length', default=1000, type=int)  # Length of the evaluated episode
    # parser.add_argument("--image_size", default=84, type=int)  # Number of pixels of the rendered frame
    args = parser.parse_args()
    # Initialize environments
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        episode_length=args.episode_length,
        # image_size=args.image_size,
    )
    print('Observations:', env.observation_space.shape)

    # Create root directory
    root_dir = pathlib.Path(__file__).parent.resolve()
    print('Root directory', root_dir)

    env_name = f'{args.domain_name}_{args.task_name}'
    model_path = f'{root_dir}/models/{env_name}.pt'
    print('Using model', model_path)
    agent = torch.load(model_path)

    print(f'\nEvaluating {env_name} for {args.eval_episodes} episodes')
    replay_dir = f'{root_dir}/replay'
    run_replay(env, agent, replay_dir, args.replay_episodes, args.episode_length)
