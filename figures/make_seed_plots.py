import numpy as np
import matplotlib.pyplot as plt


labels = {
    'normal': {'fnames': "CarEnv-Chaos-v0_normal_s", "color": 'r'},
    'noIntent': {'fnames': "CarEnv-Chaos-v0_no_intention_s", "color": 'b'},
    '8_ped': {'fnames': "CarEnv-Chaos-v0_8_ped_s", "color": 'g'},
    'Rped200': {'fnames': "None", "color": 'y'},
    'noVelR': {'fnames': "CarEnv-Chaos-v0_no_vel_s", "color": 'c'}
}

for key, vals in labels.items():
    if key not in ["normal", "noIntent", "8_ped", "noVelR"]:
        continue
    data = np.array([np.load("data/"+vals['fnames']+"0.npy")[:4879],
                     np.load("data/"+vals['fnames']+"1.npy")[:4879],
                     np.load("data/"+vals['fnames']+"2.npy")[:4879]])

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.5, facecolor=vals['color'], label=key)
    # plt.plot(mean, color=vals['color'], label=key)

plt.legend()
plt.xlabel('Training Step', fontsize=18)
plt.ylabel('Episodic Reward', fontsize=16)
plt.savefig("CarEnv-Chaos_ppo_experiments.png")
