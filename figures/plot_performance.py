import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter


configs = [['kc0.1-', 'special'], ['kc0.5-', 'special'], ['kc1.0-', 'special'], ['kc1.5-', 'special'],
           ['kc2.0-', 'special'], ['kc2.5-', 'special'], ['kc5.0-', 'special'], ['kc10.0-', 'special']]
config_folder = ''

use_tl = 'TL'
global_config = "tenOpps_intention_kcX_" + '_' + use_tl
nrows = 3
ncols = 3

labels = {}
# labels["NoIntent"] = "CarEnv-TenOpponent-States-SpeedControl-"+use_tl+"-v0_"
labels['Intent'] = "CarEnv-TenOpponentWithIntention-States-SpeedControl-"+use_tl+"-"

all_mean_values = {}
all_std_values = {}
for key, label_val in labels.items():
    for config in configs:
        config_results = {"success": 0, "sw_crash": 0, "car_crash": 0, "out_of_time": 0, "burned_light": 0}
        fnames = glob.glob("test_results/" + config_folder + label_val + config[0] + "v0_"+config[1]+"_s*_final_states.npy")
        if len(fnames) == 0:
            continue
        data = []
        for fname in fnames:
            current_data = np.load(fname)
            data.append(Counter(current_data[:100]))
        print(fnames)
        print(config, data)

        for config_k in config_results.keys():
            for seed_data in data:
                if config_k in seed_data.keys():
                    config_results[config_k] += seed_data[config_k] / len(data)
                else:
                    config_results[config_k] += 0

        all_mean_values[key + '_' + config[0] + '_' + config[1]] = config_results

font = {'size': 10}
plt.rc('font', **font)

keys = list(all_mean_values.keys())
fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(8,8))
idx = 0
for row in range(nrows):
    for col in range(ncols):
        if idx < len(all_mean_values):
            if nrows > 1:
                ax[row, col].bar(all_mean_values[keys[idx]].keys(), all_mean_values[keys[idx]].values(), 0.5)
                ax[row, col].set_title(use_tl+keys[idx])
                ax[row, col].set_xticklabels(list(all_mean_values[keys[idx]].keys()), rotation=20)
                ax[row, col].axis(ymin=0, ymax=100)
                ax[row, col].yaxis.grid(True)
            else:
                ax[col].bar(all_mean_values[keys[idx]].keys(), all_mean_values[keys[idx]].values(), 0.5)
                ax[col].set_title(use_tl+keys[idx])
                ax[col].set_xticklabels(list(all_mean_values[keys[idx]].keys()), rotation=20)
                ax[col].axis(ymin=0, ymax=100)
                ax[col].yaxis.grid(True)
            idx += 1

plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.2, wspace=0.1)
plt.tight_layout()
plt.savefig(global_config+"_final_states.png")


######################################## final state locations #####################################
colors = {'success': 'g', 'sw_crash': 'b', 'car_crash': 'r', 'out_of_time': 'k', 'burned_light': 'y'}
fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
idx = 0
plt_row = 0
plt_col = 0
for label_key, label_val in labels.items():
    for config in configs:
        config_results = {"success": [], "sw_crash": [], "car_crash": [], "out_of_time": [], "burned_light": []}
        fnames_locs = sorted(glob.glob("test_results/" + config_folder + label_val + config[0] + "v0_"+config[1]+"_s*_final_positions.npy"))
        fnames_fstates = sorted(glob.glob("test_results/" + config_folder + label_val + config[0] + "v0_"+config[1]+"_s*_final_states.npy"))
        if len(fnames_locs) == 0:
            continue
        data = []
        for i, fname_locs in enumerate(fnames_locs):
            current_locs = np.load(fname_locs)[:100].tolist()
            current_fstates = np.load(fnames_fstates[i])[:100].tolist()
            for row in range(len(current_locs)):
                config_results[current_fstates[row]].append(current_locs[row])

        for config_key, config_val in config_results.items():
            if len(config_val) == 0:
                continue
            data = np.array(config_val)
            if nrows > 1:
                ax[plt_row, plt_col].scatter(data[:, 0], data[:, 1], color=colors[config_key], s=1)
            else:
                ax[plt_col].scatter(data[:, 0], data[:, 1], color=colors[config_key], s=1)

        if nrows > 1:
            ax[plt_row, plt_col].axis(xmin=-0.1, xmax=1.1, ymin=0, ymax=1)
            ax[plt_row, plt_col].set_title(use_tl + label_key + '_' + config[0]+'_'+config[1], size=8)
        else:
            ax[plt_col].axis(xmin=-0.1, xmax=1.1, ymin=0, ymax=1)
            ax[plt_col].set_title(use_tl + label_key + '_' + config[0]+'_'+config[1], size=8)

        idx += 1
        plt_col += 1
        if idx % ncols == 0:
            plt_row += 1
            plt_col = 0

all_patches = []
for key, val in colors.items():
    all_patches.append(mpatches.Patch(color=val, label=key))
plt.legend(handles=all_patches, prop={'size': 10}, bbox_to_anchor=(0.75, 1))
plt.tight_layout()
plt.savefig(global_config+"_final_locs.png")
