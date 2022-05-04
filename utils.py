import os
import time

import cv2
import numpy as np
import torch
import logging

import augmentations


class ReplayBuffer(object):

    def __init__(self, dataset, data_path, batch_size, image_size):
        N = len(dataset[0])
        self.batch_size = batch_size
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.not_dones = []
        for episode in dataset:
            obs = []
            next_obs = []
            actions = []
            rewards = []
            not_dones = np.ones(N - 1)
            not_dones[N - 2] = 0.0
            for i in range(N - 1):
                obs.append(load_image(os.path.join(data_path, episode[i]['img_path']), image_size)/255.*2 -1)
                next_obs.append(load_image(os.path.join(data_path, episode[i + 1]['img_path']), image_size)/255.*2 -1)
                actions.append(episode[i]['action'])
                rewards.append(episode[i]['reward'])
            self.states.extend(obs)
            self.next_states.extend(next_obs)
            self.actions.extend(actions)
            self.rewards.extend(rewards)
            self.not_dones.extend(not_dones)

        self.states = np.array(self.states).transpose((0, 3, 1, 2))
        self.next_states = np.array(self.next_states).transpose((0, 3, 1, 2))
        self.actions = np.array(self.actions)
        self.rewards = np.expand_dims(self.rewards, axis=1)
        self.not_dones = np.expand_dims(self.not_dones, axis=1)
        self.size = len(self.states)

    def sample(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)

        obs = torch.FloatTensor(self.states[idxs]).cuda().float()
        next_obs = torch.FloatTensor(self.next_states[idxs]).cuda().float()
        actions = torch.FloatTensor(self.actions[idxs]).cuda()
        rewards = torch.FloatTensor(self.rewards[idxs]).cuda()
        not_dones = torch.FloatTensor(self.not_dones[idxs]).cuda()

        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones

    def normalize_states(self, eps=1e-3):
        # mean = self.states.mean(0, keepdims=True)
        # std = self.states.std(0, keepdims=True) + eps
        mean = self.states.mean()
        std = self.states.std() + eps
        self.states = (self.states - mean) / std
        self.next_states = (self.next_states - mean) / std
        return mean, std


def load_image(path: str, image_size: int):
    image = cv2.imread(path)
    image = cv2.resize(image, dsize=(image_size, image_size))
    return image


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def retry(times, exceptions):
    """
    Retry Decorator https://stackoverflow.com/a/64030200/1645784
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param exceptions: Lists of exceptions that trigger a retry attempt
    :type exceptions: Tuple of Exceptions
    """

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    print(f'Exception thrown when attempting to run {func}, attempt {attempt} out of {times}')
                    time.sleep(min(2**attempt, 10))
                    attempt += 1

            return func(*args, **kwargs)

        return newfn

    return decorator
