# A Minimalist Approach to Offline Reinforcement Learning With Data Augmentation

TD3+BC+SVEA is a simple approach to offline RL where: 
- A weighted behavior cloning loss is added to the policy update
- The states are normalized
- Q-Learning is stabilized with ConvNets and Vision Transformers under Data Augmentation

## Setup
All dependencies can then be installed with the following commands:
```
conda env create -f setup/conda.yml
conda activate svea
sh setup/install_envs.sh
```

## Running instructions
Example of recording replay with a trained master agent
```bash
$ python TD3_SVEA/main.py \
    --domain_name reacher \
    --task_name easy \
    --episode_length 1000 \
    --seed 1
```

Example of training an agent using offline RL
```bash
$ python TD3_SVEA/main.py \
    --domain_name reacher \
    --task_name easy \
    --replay_episodes 1000 \
    --save_video \
    --seed 1
```