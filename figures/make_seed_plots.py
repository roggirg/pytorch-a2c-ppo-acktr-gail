import numpy as np
import matplotlib.pyplot as plt


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

labels = {}
labels["NoIntent"] = "CarEnv-TenOpponent-States-v0__s"
labels['Intent'] = "CarEnv-TenOpponentWithIntention-States-v0__s"

for key, val in labels.items():
    data = [np.load("data/"+val+"0.npy")[:9750],
            np.load("data/"+val+"1.npy")[:9750],
            np.load("data/"+val+"2.npy")[:9750],
            np.load("data/"+val+"3.npy")[:9750]]

    data = np.array(data)
    mean = running_mean(np.mean(data, axis=0), 30)
    std = running_mean(np.std(data, axis=0), 30)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.1)
    plt.plot(range(len(mean)), mean, label=key)

plt.legend()
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Episodic Reward', fontsize=12)
plt.tight_layout()
plt.savefig("CarEnv-TenOpponent_states_ppo_experiments.png")

plt.figure()
colors ={"Intent": (0.9, 0.0, 0.0), "NoIntent": (0.0, 0.9, 0.0)}
patches = []
import matplotlib.patches as mpatches

for key, vals in labels.items():
    patches.append(mpatches.Patch(color=colors[key], label=key))
    for i in range(4):
        data = np.array([np.load("data/"+vals+str(i)+".npy")[:9750]])
        mean = running_mean(np.mean(data, axis=0), 30)
        std = running_mean(np.std(data, axis=0), 30)
        plt.plot(range(len(mean)), mean, color=colors[key])  # , label=key+"_s"+str(i))


plt.legend(handles=patches)
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Episodic Reward', fontsize=12)
plt.savefig("CarEnv-TenOpponent_states_seeds_ppo_experiments.png")
