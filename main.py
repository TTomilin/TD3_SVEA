import argparse
import json
import os
import pathlib
import time
from datetime import datetime

import numpy as np
import torch
from tensorboardX import SummaryWriter

import TD3_BC
import utils
from video import VideoRecorder
from wandb_utils import init_wandb
from wrappers import make_env
from utils import setup_logger
import logging

def evaluate(env, agent, video, num_episodes, step):
    episode_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(not episode % 5))
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            video.record(env)
            episode_reward += reward

        video.save(f'{step}_{episode}.mp4')
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--domain_name", default="walker")              # DeepMind Control Suite domain name
    parser.add_argument("--task_name", default="walk")                  # DeepMind Control Suite task name
    parser.add_argument("--name_suffix", default="")                    # Experiment suffix for better distinction
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--train_steps', default=5e5, type=int)         # Number of total training iterations
    parser.add_argument('--save_freq', default=10000, type=int)         # How often to save the model
    parser.add_argument("--image_size", default=84, type=int)           # Number of pixels of the downscaled image
    parser.add_argument('--replay_episodes', default=100, type=int)     # Training dataset size
    # Model storage
    parser.add_argument("--save_model", default=True, action='store_true')      # Save model and optimizer parameters
    parser.add_argument("--load_model", default=False, action='store_true')     # Load existing model
    # Eval
    parser.add_argument('--eval_freq', default=1000, type=int)              # How often to evaluate
    parser.add_argument('--eval_episodes', default=30, type=int)            # How many episodes to evaluate
    parser.add_argument('--episode_length', default=1000, type=int)         # Length of the evaluation episode
    parser.add_argument('--save_video', default=True, action='store_true')  # Save video
    # TD3
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    # TD3 + BC
    parser.add_argument("--alpha", default=2.5)
    parser.add_argument("--normalize", default=True, type=bool)
    # Weights and Biases experiment monitoring
    parser.add_argument('--with_wandb', default=True, action='store_true', help='Enables Weights and Biases integration')
    parser.add_argument('--wandb_user', default=None, type=str, help='WandB username (entity). Must be specified from command line! Also see https://docs.wandb.ai/quickstart#1.-set-up-wandb')
    parser.add_argument('--wandb_project', default='TD3_SVEA', type=str, help='WandB "Project"')
    parser.add_argument('--wandb_group', default=None, type=str, help='WandB "Group" (to group your experiments). By default this is the name of the env.')
    parser.add_argument('--wandb_job_type', default='default', type=str, help='WandB job type')
    parser.add_argument('--wandb_tags', default=[], type=str, nargs='*', help='Tags can help with finding experiments in WandB web console')
    parser.add_argument('--wandb_key', default=None, type=str, help='API key for authorizing WandB')


    parser.add_argument("--lr", default=3e-4, type=float)

    args = parser.parse_args()

    env_name = f'{args.domain_name}_{args.task_name}_{args.replay_episodes}{args.name_suffix}'
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        image_size=args.image_size,
    )

    # Create root directory
    root_dir = pathlib.Path(__file__).parent.resolve()
    print('Root directory', root_dir)
    current_time = datetime.now().strftime('%b%d_%H-%M')



    work_dir = f'{root_dir}/experiments/{env_name}/{args.seed}/{current_time}'
    print('Experiment directory', env_name)

    summary_dir = utils.ensure_dir_exists(f"{work_dir}/summary")
    model_dir = utils.ensure_dir_exists(f"{work_dir}/models")
    video_dir = utils.ensure_dir_exists(f"{work_dir}/video")

    video = VideoRecorder(video_dir if args.save_video else None)
    writer = SummaryWriter(summary_dir, flush_secs=20)

    log = {}
    setup_logger('{}_log'.format(env_name),
                 r'{0}/logger'.format(work_dir))
    log['{}_log'.format(env_name)] = logging.getLogger(
        '{}_log'.format(env_name))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(env_name)].info('{0}: {1}'.format(k, d_args[k]))


    print("---------------------------------------")
    print(f"Env: {env_name}, Seed: {args.seed}")
    print("---------------------------------------")

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "obs_shape": obs_shape,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        # TD3 + BC
        "alpha": args.alpha,

        # add by Reed
        "lr": args.lr
    }

    # Initialize or load policy
    if args.load_model:
        model_names = next(os.walk(model_dir), (None, None, []))[2]
        latest_model = sorted(model_names)[-1]
        agent = torch.load(os.path.join(model_dir, latest_model))
    else:
        agent = TD3_BC.TD3_BC(**kwargs)

    # Initialize WandB
    # init_wandb(args)

    replay_dir = f'{root_dir}/replay/{env_name}'
    print('Replay directory', replay_dir)
    assert os.path.exists(replay_dir)
    with open(f'{replay_dir}/data.json', 'r') as data_file:
        dataset = json.load(data_file)
    replay_buffer = utils.ReplayBuffer(dataset, replay_dir, args.batch_size, args.image_size)

    mean, std = replay_buffer.normalize_states() if args.normalize else 0, 1

    experiment_start = time.time()
    for step in range(int(args.train_steps) + 1):
        # Evaluate agent periodically
        if not step % args.eval_freq:
            # print("---------------------------------------")
            # print(f'Evaluating step {step} for {args.eval_episodes} episodes')
            avg_reward = evaluate(env, agent, video, args.eval_episodes, step)
            writer.add_scalar(f'eval/reward', avg_reward, step)
            # print(f"Evaluation reward: {avg_reward:.3f}")
            log['{}_log'.format(env_name)].info(
                        "Time {0}, Evaluating step {1} for {2} episodes, ave reward {3:.3f}".
                        format(
                            time.strftime("%Hh %Mm %Ss",
                                        time.gmtime(time.time() - experiment_start)), step, args.eval_episodes, avg_reward))

            # print("---------------------------------------")

        # Save agent periodically
        if not step % args.save_freq:
            # agent.save(f'model_dir/{step}')
            torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

        # Run training update
        train_start = time.time()
        agent.train(replay_buffer, writer, step, args.batch_size)
        writer.add_scalar('train/duration', time.time() - train_start, step)

    print('Completed training for', env_name)
