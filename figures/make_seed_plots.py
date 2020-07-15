import numpy as np
import matplotlib.pyplot as plt


# configs = ["kf1.0_kl1.0_kt1.0_kp1.0", "kf1.0_kl1.0_kt0.1_kp0.1", "kf1.0_kl1.0_kt0.1_kp1.0", "kf1.0_kl1.0_kt1.0_kp0.1",
#            "kf1.0_kl1.0_kt10.0_kp1.0", "kf1.0_kl10.0_kt0.1_kp1.0", "kf1.0_kl0.1_kt0.1_kp10.0",
#            "kf1.0_kl10.0_kt0.1_kp10.0", "kf10.0_kl1.0_kt1.0_kp1.0", "kf10.0_kl1.0_kt0.1_kp10.0"]

configs = ["kf1.0_kl1.0_kt1.0_kp1.0", "kf1.0_kl1.0_kt0.1_kp1.0", "kf10.0_kl1.0_kt0.1_kp10.0",
           "kf10.0_kl1.0_kt1.0_kp0.1", "kf10.0_kl0.1_kt0.1_kp1.0"]


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# fig, ax = plt.subplots(nrows=2, ncols=3, sharex='col', gridspec_kw={'hspace': 0.3, 'wspace': 0.2})
# idx = 0
# for row in ax:
#     for col in row:
#         if idx > len(configs)-1:
#             continue
#         config = configs[idx]
#         idx += 1
#         for task in ['CarEnv-TwoOpponent-States-', 'CarEnv-TwoOpponentWithIntention-States-']:
#             print(task+config+"-v0")
#             data = [np.load("data/"+task+config+"-v0__s0.npy")[:4879],
#                     np.load("data/"+task+config+"-v0__s1.npy")[:4879],
#                     np.load("data/"+task+config+"-v0__s2.npy")[:4879]]
#
#             if "WithInt" in task:
#                 label = 'WithIntent'
#                 color = (0.9, 0.6, 0, 0.5)
#             else:
#                 label = 'NoIntent'
#                 color = (0, 0, 0, 0.5)
#
#             for i, dat in enumerate(data):
#                 if len(dat) < 4000:
#                     print("deleting..."+task+config)
#                     del data[i]
#             data = np.array(data)
#             mean = running_mean(np.mean(data, axis=0), 50)
#             std = running_mean(np.std(data, axis=0), 50)
#             col.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3, facecolor=color)
#             col.plot(range(len(mean)), mean, label=label, color=color)
#             col.set_title(config, size=8)
#
# ax[-1][-2].legend(loc=3, bbox_to_anchor=(1.0, 0.0, 0.5, 0.5), fancybox=False, shadow=False)
#
# # fig.delaxes(ax[-1][-2])
# fig.delaxes(ax[-1][-1])

labels = {}
labels["NoIntent"] = "CarEnv-TwoOpponent-States-v0__s"
labels['Intent'] = "CarEnv-TwoOpponentWithIntention-States-v0__s"

for key, val in labels.items():
    data = [np.load("data/"+val+"0.npy")[:4879],
            np.load("data/"+val+"1.npy")[:4879],
            np.load("data/"+val+"2.npy")[:4879],
            np.load("data/"+val+"3.npy")[:4879],
            np.load("data/"+val+"4.npy")[:4879],
            np.load("data/"+val+"5.npy")[:4879]]

    data = np.array(data)
    mean = running_mean(np.mean(data, axis=0), 30)
    std = running_mean(np.std(data, axis=0), 30)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.1)
    plt.plot(range(len(mean)), mean, label=key)

plt.legend()
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Episodic Reward', fontsize=12)
plt.tight_layout()
plt.savefig("CarEnv-TwoOpponent_states_ppo_experiments.png")

plt.figure()
colors ={"Intent": (0.9, 0.0, 0.0), "NoIntent": (0.0, 0.9, 0.0)}
patches = []
import matplotlib.patches as mpatches

for key, vals in labels.items():
    patches.append(mpatches.Patch(color=colors[key], label=key))
    for i in range(6):
        data = np.array([np.load("data/"+vals+str(i)+".npy")[:4879]])
        mean = running_mean(np.mean(data, axis=0), 30)
        std = running_mean(np.std(data, axis=0), 30)
        plt.plot(range(len(mean)), mean, color=colors[key])  # , label=key+"_s"+str(i))


plt.legend(handles=patches)
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Episodic Reward', fontsize=12)
plt.savefig("CarEnv-TwoOpponent_states_seeds_ppo_experiments.png")
