import argparse
import cv2
import os
# workaround to unpickle olf model files
import sys
import pickle

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='carenv',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/ppo/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--config',
    type=str,
    default='',
    help='training config name for comparisons')
parser.add_argument(
    '--save-flag',
    type=str,
    default='',
    help='special string for saving file to see how different env configs affects the trained agents.')
args = parser.parse_args()

args.det = not args.non_det
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

num_processes = 16
if 'train' in args.save_flag:
    env_seed = 100
    num_steps = 1000
else:
    env_seed = 1000
    num_steps = 200
env = make_vec_envs(args.env_name, env_seed, num_processes, None, None, device=device, allow_early_resets=False, pixels=False)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + '_'+args.config+'_s'+str(args.seed) + ".pt"),
                                  map_location=lambda storage, loc: storage)
actor_critic.to(device)

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size).to(device)
masks = torch.zeros(num_processes)

obs = env.reset()

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

t = 0
trajectories = {}
for i in range(num_processes):
    trajectories[i] = []

storage = []
while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, info = env.step(action)

    for i in range(num_processes):
        trajectories[i].append(obs[i].tolist())

    for i, done_ in enumerate(done):
        if done_:
            storage.append(trajectories[i][:-1])
            trajectories[i] = []

    if len(storage) > num_steps:
        break

    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

storage = np.array(storage)
np.save("probing/dataset/" + args.env_name+'_'+args.config+'_s'+str(args.seed)+'_trajectories_'+args.save_flag+'.npy', storage)
