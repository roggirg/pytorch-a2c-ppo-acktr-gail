import numpy as np
import matplotlib.pyplot as plt
import glob
import json


# configs = ['simple', 'recurrentsimple', 'deep', 'special']
configs = ['TL_kp200_special', 'TL_kp250_special', 'TL_kp300_special']


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

labels = {}
labels["NoIntent"] = "CarEnv-TenOpponent-States-SpeedControl-v0_"
labels['Intent'] = "CarEnv-TenOpponentWithIntention-States-SpeedControl-v0_"

# for config in configs:
    # print(glob.glob("data/"+labels["NoIntent"]+config+"*.npy"))
#     print(glob.glob("data/"+labels["PredIntent"]+config+"*.npy"))
#     print(glob.glob("data/" + labels["Intent"]+config+"*.npy"))

for key, val in labels.items():
    for config in configs:
        fnames = glob.glob("data/" + val + config + "*.npy")
        data = []
        for fname in fnames:
            data.append(np.load(fname)[:7300])

        data = np.array(data)
        mean = running_mean(np.mean(data, axis=0), 30)
        std = running_mean(np.std(data, axis=0), 30)
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)
        plt.plot(range(len(mean)) , mean, label=key+"_"+config)

plt.legend(loc='lower right')
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Episodic Reward', fontsize=12)
plt.tight_layout()
plt.savefig("CarEnv-TenOpponent_states_SpeedControl_ppo_experiments.png")


# plt.figure()
# colors ={"Intent": (0.9, 0.0, 0.0), "NoIntent": (0.0, 0.9, 0.0)}
# patches = []
# import matplotlib.patches as mpatches
#
# for key, vals in labels.items():
#     patches.append(mpatches.Patch(color=colors[key], label=key))
#     for i in range(4):
#         data = np.array([np.load("data/"+vals+str(i)+".npy")[:9750]])
#         mean = running_mean(np.mean(data, axis=0), 30)
#         std = running_mean(np.std(data, axis=0), 30)
#         plt.plot(range(len(mean)), mean, color=colors[key])  # , label=key+"_s"+str(i))
#
#
# plt.legend(handles=patches)
# plt.xlabel('Training Step', fontsize=12)
# plt.ylabel('Episodic Reward', fontsize=12)
# plt.savefig("CarEnv-TenOpponent_states_seeds_ppo_experiments.png")


# More stats
for key, label_val in labels.items():
    for config in configs:
        fnames = glob.glob("data/" + label_val + config + "*.json")
        data = []
        for fname in fnames:
            with open(fname) as f:
                current_data = json.load(f)
                data.append(current_data)

        values = {}
        for k in data[0][0].keys():
            if 'final_position' not in k:
                values[k] = np.zeros(len(data))

        for seed, seed_data in enumerate(data):
            total_count = 0
            for j in reversed(range(len(seed_data))):
                for k, val in seed_data[j].items():
                    if 'final_position' not in k:
                        values[k][seed] += val

        mean_values = {}
        for k, val in values.items():
            mean_values[k] = np.mean(val)

        plt.figure()
        plt.bar(mean_values.keys(), mean_values.values(), 1, alpha=0.7)
        plt.savefig("CarEnv-TenOpponent_states_SpeedControl_ppo_final_states_"+key+"_"+config+".png")
