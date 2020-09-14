import numpy as np
import matplotlib.pyplot as plt
import glob
import json


configs = ['kp50_special']


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

labels = {}
labels["NoIntent"] = "CarEnv-TenOpponent-States-SpeedControl-v0_"
labels['Intent'] = "CarEnv-TenOpponentWithIntention-States-SpeedControl-v0_"

for config in configs:
    print(glob.glob("data/"+labels["NoIntent"]+config+"*.npy"))
    # print(glob.glob("data/"+labels["PredIntent"]+config+"*.npy"))
    print(glob.glob("data/" + labels["Intent"]+config+"*.npy"))

for key, val in labels.items():
    for config in configs:
        fnames = glob.glob("data/" + val + config + "*.npy")
        data = []
        for fname in fnames:
            data.append(np.load(fname)[:13392])

        data = np.array(data)
        mean = running_mean(np.mean(data, axis=0), 30)
        std = running_mean(np.std(data, axis=0), 30)
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)
        plt.plot(range(len(mean)) , mean, label=key+"_"+config)

plt.legend(loc='lower right')
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Episodic Reward', fontsize=12)
plt.tight_layout()
plt.savefig("CarEnv-TenOpponent_states_SpeedControl_ppo_experiments_3M.png")
