import copy
import glob
import os
import pickle
import time
from collections import deque
import json

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate, evaluate_carenv


def main():
    args = get_args()

    traindir_name = args.base_encoder + "_model"
    if args.recurrent_policy:
        traindir_name += "_recurrent"
    traindir_name += "/LR{0}NSteps{1}".format(args.lr, args.num_steps)
    if "TL" in args.env_name:
        traindir_name += "_TL"

    args.config += args.base_encoder
    if args.recurrent_policy:
        args.config += "_recurrent"
    print("CONFIG:", args.config)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, args.log_dir,
                         device, False, pixels=args.pixels)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy, 'predict_intention': args.predict_intention},
                          base_encoder=args.base_encoder)
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    all_rewards = []

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    print("Total number of updates:", num_updates)

    final_states_data = []
    best_success = 0

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(agent.optimizer, j, num_updates,
                                         agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            for info in infos:
                if 'bad_transition' in info.keys():
                    print("Bad Transition Encountered...")
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]
            ).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: "
                "mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                "dist_entropy {}, value_loss {}, action_loss {} \n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss), "\n")

            all_rewards.append(np.mean(episode_rewards))

        if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0:
            if "CarEnv" not in args.env_name:
                ob_rms = utils.get_vec_normalize(envs).ob_rms
                evaluate(actor_critic, ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)
            else:
                eval_fstates = evaluate_carenv(actor_critic, args.env_name, args.seed, args.num_processes, device)
                final_states_data.append(dict(eval_fstates))
                save_path = os.path.join(args.save_dir, args.algo, traindir_name)
                fname = os.path.join(save_path, args.env_name + '_' + args.config + '_s' + str(args.seed) + '_fstates.pickle')
                with open(fname, 'wb') as handle:
                    pickle.dump(final_states_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo, traindir_name)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            if len(final_states_data) > 0 and 'success' in final_states_data[-1].keys():
                if final_states_data[-1]['success'] > best_success:
                    best_success = final_states_data[-1]['success']
                    torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'ob_rms', None)],
                               os.path.join(save_path, args.env_name + '_' + args.config + '_s'+str(args.seed)+
                                            "_best.pt"))

            torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'ob_rms', None)],
                       os.path.join(save_path, args.env_name + '_' + args.config + '_s' + str(args.seed) + ".pt"))

            np.save(os.path.join(save_path, args.env_name + '_' + args.config + '_s' + str(args.seed) + '.npy'),
                    all_rewards)


if __name__ == "__main__":
    main()
