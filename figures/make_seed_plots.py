import numpy as np
import matplotlib.pyplot as plt


# labels = {
#     'NoIntent': {'fnames': "CarEnv-TwoOpponent-v0_state_s", "color": 'r'},
#     'Intent': {'fnames': "CarEnv-TwoOpponentWithIntention-v0_state_s", "color": 'b'},
# }

labels = {
    'NoIntent': {'fnames': "CarEnv-TwoOpponent-v0_bin_imgs_s", "color": 'r'},
    'Intent': {'fnames': "CarEnv-TwoOpponentWithIntention-v0_bin_imgs_s", "color": 'b'},
    'Old_NoIntent': {'fnames': "CarEnv-TwoOpponent-v0_bin_imgs_oldarch_s", "color": 'g'},
}


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


for key, vals in labels.items():
    data = np.array([np.load("data/"+vals['fnames']+"1.npy")[:4879],
                     np.load("data/"+vals['fnames']+"2.npy")[:4879]])

    mean = running_mean(np.mean(data, axis=0), 50)
    std = running_mean(np.std(data, axis=0), 50)
    # plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.5, facecolor=vals['color'], label=key)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.1, facecolor=vals['color'])
    plt.plot(range(len(mean)), mean, vals['color'], label=key)
    # plt.plot(mean, color=vals['color'], label=key)

plt.legend()
plt.xlabel('Training Step', fontsize=18)
plt.ylabel('Episodic Reward', fontsize=16)
plt.savefig("CarEnv-TwoOpponent_binIMG_ppo_experiments.png")
