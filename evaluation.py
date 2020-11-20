import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from collections import Counter


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))


def evaluate_carenv(actor_critic, env_name, eval_seed, num_processes, device, num_eval_episodes=100):
    env = make_vec_envs(env_name, eval_seed, num_processes, None, None, device=device,
                        allow_early_resets=False, pixels=False)

    recurrent_hidden_states = torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size).to(device)
    masks = torch.zeros(num_processes, 1).to(device)
    obs = env.reset()
    t = 0
    final_states = []
    final_positions = []
    while True:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(obs, recurrent_hidden_states, masks)

        # Obser reward and next obs
        obs, reward, done, info = env.step(action)
        t += 1
        for i, done_ in enumerate(done):
            if done_:
                if len(final_positions) % 10 == 0:
                    print(len(final_positions))
                for key in info[i].keys():
                    if 'final_position' not in key:
                        final_states.append(key)
                    else:
                        final_positions.append(info[i][key])

        if len(final_positions) > num_eval_episodes:
            break

        # masks.fill_(0.0 if done else 1.0)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)

    env.close()
    print("Evaluation Performance:", Counter(final_states[:num_eval_episodes]))
    return Counter(final_states[:num_eval_episodes])
