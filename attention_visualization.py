import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


label = "noIntention"
traj_index = 78
image_folder = "trajectory_dataset/attn_figs"

filelist = glob.glob(os.path.join(image_folder, "*.png"))
for f in filelist:
    os.remove(f)


def intention_to_color(intentions):
    colors = ['r', 'orange', 'b', 'g']
    intention_colors = []
    for int in intentions:
        intention_colors.append(colors[int])
    return intention_colors


fname = glob.glob("trajectory_dataset/*.npy")[0]
trajectories = np.load(fname, allow_pickle=True)

agent_dim = 5  # pos(2), vel(2), heading(1)
env_dim = 8  # four corners of the environment(4), light status(4)
opponent_dim = 5  # pos(2), vel(2), heading(1)
intention_dim = 4  # one-hot vector for four possible intentions(4)
num_opponents = 10
hidden_size = 32


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if "no" not in label:
    actor_critic, ob_rms = torch.load(
        "trained_models/ppo/CarEnv-TenOpponentWithIntention-States-SpeedControl-TL-v0_kp250_special_s0.pt",
        map_location=lambda storage, loc: storage
    )
    intention_multiplier = 1.0
else:
    actor_critic, ob_rms = torch.load(
        "trained_models/ppo/CarEnv-TenOpponent-States-SpeedControl-TL-v0_kp150_special_s0.pt",
        map_location=lambda storage, loc: storage
    )
    intention_multiplier = 0.0
actor_critic.to(device)

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)
masks = torch.zeros(1, 1).to(device)

for i in range(len(trajectories[traj_index])):
    datap = torch.from_numpy(np.array(trajectories[traj_index][i])).to(device).unsqueeze(0).float()
    agent_position = np.array(trajectories[traj_index][i])[:2]

    agent_enc = actor_critic.base.agent_model(datap[:, :agent_dim])
    env_enc = actor_critic.base.env_model(datap[:, agent_dim:agent_dim+env_dim])

    opponent_state = datap[:, agent_dim+env_dim:].reshape((-1, opponent_dim+intention_dim+1))
    opp_positions = opponent_state[:, :2].cpu().numpy()
    opp_velocities = opponent_state[:, 2:4].cpu().numpy()
    opponentdyn_enc = actor_critic.base.opponent_model(opponent_state[:, :opponent_dim])*opponent_state[:, -1].unsqueeze(1)
    intention = opponent_state[:, opponent_dim:opponent_dim + intention_dim]
    intention_label = np.argmax(intention.cpu().numpy(), axis=-1)
    opponent_state = torch.cat((opponentdyn_enc, intention*intention_multiplier), dim=-1).view((1, num_opponents, -1))

    # computing intention weights
    expanded_agentenv_state = torch.cat((agent_enc, env_enc), dim=-1).repeat(1, num_opponents).view(1, num_opponents, -1)
    agentenvopp_states = torch.cat((expanded_agentenv_state, opponent_state), dim=-1).reshape(-1, 3*hidden_size + intention_dim)
    attention_weights = F.softmax(actor_critic.base.attention_model(agentenvopp_states).view(1, -1)).unsqueeze(2)
    attn_ws = attention_weights.squeeze().cpu().detach().numpy()

    # getting the attended opponent encoding.
    attended_opponentdyn_enc = (opponentdyn_enc.view((1, num_opponents, -1)) * attention_weights).sum(1)

    x = torch.cat((agent_enc, env_enc, attended_opponentdyn_enc), dim=-1)

    fig, ax = plt.subplots()
    intention_colors = intention_to_color(intention_label)
    for idx in range(num_opponents):
        if not (abs(opp_positions[idx, 0]) < 0.01 and abs(opp_positions[idx, 1]) < 0.01):
            # plt.text(opp_positions[idx, 0] + agent_position[0], opp_positions[idx, 1] + agent_position[1], attn_ws[idx], fontsize=8)

            # circle = plt.Circle((opp_positions[idx, 0] + agent_position[0], opp_positions[idx, 1] + agent_position[1]), radius=attn_ws[idx]/10, color='w')
            # ax.add_artist(circle)

            vel_norm = opp_velocities[idx] / (np.linalg.norm(opp_velocities[idx])+1e-05)
            plt.arrow(opp_positions[idx, 0] + agent_position[0], opp_positions[idx, 1] + agent_position[1],
                      0.02*np.linalg.norm(opp_velocities[idx])*vel_norm[0], 0.02*np.linalg.norm(opp_velocities[idx])*vel_norm[1],
                      length_includes_head=True, head_width=0.015, head_length=0.008)

            plt.scatter(opp_positions[idx, 0] + agent_position[0], opp_positions[idx, 1] + agent_position[1], s=attn_ws[idx]*1000, color='w')
            plt.scatter(opp_positions[idx, 0] + agent_position[0], opp_positions[idx, 1] + agent_position[1], c=intention_colors[idx])

    plt.scatter(agent_position[0], agent_position[1], color='y')
    plt.axis(xmin=-0.1, xmax=1.1, ymin=0, ymax=1)
    ax.set_facecolor('gray')
    plt.title(label + " Model - Timestep " + str(i))
    plt.savefig("trajectory_dataset/attn_figs/step_"+f"{i:05d}.png")


video_name = 'trajectory_dataset/attn_trajectory_'+label+'Model_traj'+str(traj_index)+'.mp4'

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()