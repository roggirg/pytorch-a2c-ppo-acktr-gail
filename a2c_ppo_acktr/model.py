import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, base_encoder='simple'):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                if 'mlp' in base_encoder:
                    base = MLPBase
                elif 'deep' in base_encoder:
                    base = DeepMLPBase
                elif 'attn' in base_encoder:
                    base = AttnMLP
                elif 'special' in base_encoder:
                    base = SpecialMLP
                elif 'MHSA' in base_encoder:
                    base = MHeadAttnModel
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size, model_type='normal'):
        super(NNBase, self).__init__()

        self._recurrent = recurrent
        self.model_type = model_type
        self._output_size = hidden_size
        self.hidden_size = hidden_size
        self.recurrent_input_size = recurrent_input_size

        if recurrent:
            if 'special' in model_type or 'attn' in model_type:
                self._recurrent_hidden_state_size = recurrent_input_size
            else:
                self._recurrent_hidden_state_size = hidden_size

            if 'attn' in model_type:
                self.gru = nn.GRU(3*hidden_size, hidden_size)
            else:
                self.gru = nn.GRU(recurrent_input_size, self._recurrent_hidden_state_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._recurrent_hidden_state_size
        return 1

    @property
    def output_size(self):
        return self._output_size

    def _forward_gru(self, x, hxs, masks):
        if 'attn' in self.model_type and hxs.size(0) != 4:
            hxs = (hxs * masks).view((-1, self.hidden_size))
        if x.size(0) == hxs.size(0):
            if 'attn' not in self.model_type:
                hxs = (hxs * masks).unsqueeze(0)
            else:
                hxs = hxs.unsqueeze(0)
            x = x.unsqueeze(0)
            x, hxs = self.gru(x, hxs)
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
            if 'attn' in self.model_type:
                hxs = hxs.view((-1, self.recurrent_hidden_state_size))
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)
            if 'attn' in self.model_type:
                num_opps = hxs.size(1) // self.hidden_size
                T = T // num_opps
                x = x.view(T, N*num_opps, -1)
            else:
                # unflatten
                x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                if 'attn' in self.model_type:
                    if hxs.size(1) != N:
                        hxs = hxs.view(1, N, -1)
                    hxs = (hxs * masks[start_idx].view(1, -1, 1)).view((1, -1, self.hidden_size))
                else:
                    hxs = hxs * masks[start_idx].view(1, -1, 1)

                rnn_scores, hxs = self.gru(x[start_idx:end_idx], hxs)

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            hxs = hxs.squeeze(0)
            if 'attn' in self.model_type:
                hxs = hxs.view((-1, self.recurrent_hidden_state_size))
                x = x.view(T * x.size(1), -1)
            else:
                x = x.view(T * N, -1)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 1, stride=1)), nn.ReLU(),  # out: 128x128x32
            init_(nn.Conv2d(32, 32, 4, stride=2)), nn.ReLU(),  # out: 63x63x32
            init_(nn.Conv2d(32, 32, 5, stride=2)), nn.ReLU(),  # out: 30x30x32
            init_(nn.Conv2d(32, 32, 4, stride=2)), nn.ReLU(),  # out: 14x14x32
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),  # out: 6x6x64
            Flatten())

        self.linear = nn.Sequential(init_(nn.Linear(6*6*64, hidden_size)), nn.ReLU())

        # self.main = nn.Sequential(
        #     init_(nn.Conv2d(num_inputs, 32, 1, stride=1)), nn.ReLU(),
        #     init_(nn.Conv2d(32, 32, 3, stride=2)), nn.ReLU(),
        #     init_(nn.Conv2d(32, 64, 5, stride=2)), nn.ReLU(),
        #     init_(nn.Conv2d(64, 32, 5, stride=2)), nn.ReLU(), Flatten(),
        #     init_(nn.Linear(32 * 8 * 8, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)
        x = self.linear(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class SpecialCNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(SpecialCNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        self.frame_stack = 3
        num_channels = 16

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.agentpath_model = self.main = nn.Sequential(
            init_(nn.Conv2d(1 + self.frame_stack, num_channels, 1, stride=1)), nn.ReLU(),
        )

        self.agentopponents_model = self.main = nn.Sequential(
            init_(nn.Conv2d(2*self.frame_stack, num_channels, 1, stride=1)), nn.ReLU(),
        )

        self.agentenv_model = self.main = nn.Sequential(
            init_(nn.Conv2d(2 + self.frame_stack, num_channels, 1, stride=1)), nn.ReLU(),
        )

        # in: 48x128x128
        self.main = nn.Sequential(
            init_(nn.Conv2d(num_channels*3, 64, 4, stride=2)), nn.ReLU(),  # out: 64x63x63
            init_(nn.Conv2d(64, 32, 5, stride=2)), nn.ReLU(),  # out: 32x30x30
            init_(nn.Conv2d(32, 32, 4, stride=2)), nn.ReLU(),  # out: 32x14x14
            init_(nn.Conv2d(32, 32, 4, stride=2)), nn.ReLU(),  # out: 32x6x6
            Flatten(),
            init_(nn.Linear(6 * 6 * 32, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        total_params = 0
        for model in [self.agentenv_model,
                      self.agentopponents_model,
                      self.agentpath_model,
                      self.main]:
            total_params += sum(p.numel() for p in model.parameters())
            print("TOTAL NUM of PARAMS:", total_params)

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        inputs = inputs / 255.0

        agentpath_feats = self.agentpath_model(
            torch.stack((inputs[:, 1], inputs[:, 2], inputs[:, 7], inputs[:, 12]), dim=1)
        )

        agentopponent_feats = self.agentopponents_model(
            torch.stack((inputs[:, 2], inputs[:, 7], inputs[:, 12],
                         inputs[:, 3], inputs[:, 8], inputs[:, 13]), dim=1)
        )

        agentenv_feats = self.agentenv_model(
            torch.stack((inputs[:, 0], inputs[:, 2], inputs[:, 7], inputs[:, 12], inputs[:, 14]), dim=1)
        )

        x = self.main(torch.cat((agentpath_feats, agentopponent_feats, agentenv_feats), dim=1))

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, predict_intention=False):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class DeepMLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(DeepMLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class AttnMLP(NNBase):
    '''
    MLP class specialized for car environment with opponents.
    '''
    def __init__(self, num_inputs, recurrent=False, hidden_size=32, predict_intention=False):
        self.predict_intention = predict_intention
        self.hidden_size = hidden_size
        self.agent_dim = 5  # pos(2), vel(2), heading(1)
        self.env_dim = 8  # four corners of the environment(4), light status(4)
        self.opponent_dim = 5  # pos(2), vel(2), heading(1)
        self.intention_dim = 4  # one-hot vector for four possible intentions(4)
        self.num_opponents = int((num_inputs-self.agent_dim-self.env_dim) / (self.opponent_dim+self.intention_dim+1))
        act_crit_dim = 3*hidden_size
        super(AttnMLP, self).__init__(recurrent, hidden_size*self.num_opponents, hidden_size, model_type='attn')

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.agent_model = nn.Sequential(
            init_(nn.Linear(self.agent_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()
        )

        self.env_model = nn.Sequential(
            init_(nn.Linear(self.env_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        if self.num_opponents > 0:
            self.opponent_model = nn.Sequential(
                init_(nn.Linear(self.opponent_dim, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()
            )

            self.attention_model = nn.Sequential(
                init_(nn.Linear(hidden_size+self.intention_dim, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, 1))
            )

            if not self.is_recurrent:
                self.agentopponent_encoder = nn.Sequential(
                    init_(nn.Linear(3*hidden_size, hidden_size)), nn.Tanh(),
                    init_(nn.Linear(hidden_size, hidden_size))
                )

            if self.predict_intention:
                # A model that predicts intention based on opponent state
                self.intention_prediction = nn.Sequential(
                    init_(nn.Linear(self.agent_dim+self.opponent_dim, hidden_size)), nn.Tanh(),
                    init_(nn.Linear(hidden_size, 4)), nn.Softmax()
                )

        self.actor = nn.Sequential(
            init_(nn.Linear(act_crit_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(act_crit_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        batch_size = inputs.size(0)
        agent_state = self.agent_model(x[:, :self.agent_dim])
        env_state = self.env_model(x[:, self.agent_dim:self.agent_dim+self.env_dim])
        expanded_agentenv_state = torch.cat((agent_state, env_state), dim=-1).repeat(1, self.num_opponents).view(batch_size*self.num_opponents, -1)

        if self.num_opponents > 0:
            # encoding opponent state
            opponent_state = x[:, self.agent_dim+self.env_dim:].reshape((-1, self.opponent_dim+self.intention_dim+1))
            intention = opponent_state[:, self.opponent_dim:self.opponent_dim + self.intention_dim]
            active_opp = opponent_state[:, -1].unsqueeze(1)
            opponentdyn_enc = self.opponent_model(opponent_state[:, :self.opponent_dim])
            if self.is_recurrent:
                opponentdyn_enc, rnn_hxs = self._forward_gru(
                    torch.cat((expanded_agentenv_state, opponentdyn_enc), dim=-1) * active_opp, rnn_hxs, masks)
            else:
                opponentdyn_enc = torch.cat((expanded_agentenv_state, opponentdyn_enc), dim=-1) * active_opp
                opponentdyn_enc = self.agentopponent_encoder(opponentdyn_enc)

            # Computing attention
            attention_weights = F.softmax(
                (self.attention_model(torch.cat((opponentdyn_enc, intention), dim=-1))*active_opp).view(batch_size, -1)
            ).unsqueeze(2)
            attended_opponentdyn_enc = (opponentdyn_enc.view((batch_size, self.num_opponents, -1)) * attention_weights).sum(1)

            x = torch.cat((agent_state, env_state, attended_opponentdyn_enc), dim=-1)
        else:
            x = torch.cat((agent_state, env_state), dim=-1)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MHeadAttnModel(NNBase):
    '''
    MLP class specialized for car environment with opponents.
    '''
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, predict_intention=False):
        self.predict_intention = predict_intention
        self.hidden_size = hidden_size
        self.agent_dim = 5  # pos(2), vel(2), heading(1)
        self.env_dim = 8  # four corners of the environment(4), light status(4)
        self.opponent_dim = 5  # pos(2), vel(2), heading(1)
        self.intention_dim = 4  # one-hot vector for four possible intentions(4)
        self.num_opponents = int((num_inputs-self.agent_dim-self.env_dim) / (self.opponent_dim+self.intention_dim+1))
        act_crit_dim = hidden_size + self.intention_dim
        super(MHeadAttnModel, self).__init__(recurrent, hidden_size*self.num_opponents, hidden_size, model_type='mhead')

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.agent_model = nn.Sequential(
            init_(nn.Linear(self.agent_dim+self.env_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()
        )

        if self.num_opponents > 0:
            self.opponent_model = nn.Sequential(
                init_(nn.Linear(self.opponent_dim+self.env_dim, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()
            )

            self.num_heads = 4
            self.attention_model1 = nn.MultiheadAttention(hidden_size, self.num_heads)
            self.attention_model2 = nn.MultiheadAttention(hidden_size+self.intention_dim, self.num_heads)

        self.actor = nn.Sequential(
            init_(nn.Linear(act_crit_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(act_crit_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        batch_size = inputs.size(0)
        agent_enc = self.agent_model(x[:, :self.agent_dim+self.env_dim])
        env_state = x[:, self.agent_dim:self.agent_dim+self.env_dim].repeat(1, self.num_opponents).view(batch_size*self.num_opponents, -1)

        if self.num_opponents > 0:
            # encoding opponent state
            opponent_state = x[:, self.agent_dim+self.env_dim:].reshape((-1, self.opponent_dim+self.intention_dim+1))
            intention = opponent_state[:, self.opponent_dim:self.opponent_dim + self.intention_dim].reshape(batch_size, -1)
            active_opp = opponent_state[:, -1].unsqueeze(1).view(batch_size, self.num_opponents, -1)
            active_agentopp = torch.cat((torch.ones(batch_size, 1).to(inputs.device), active_opp.squeeze(-1)), dim=-1)
            key_padding_mask = (1.0 - active_agentopp).type(torch.BoolTensor).to(inputs.device)
            opponentdyn_enc = (
                self.opponent_model(torch.cat((opponent_state[:, :self.opponent_dim], env_state), dim=-1))
            ).view((batch_size, self.num_opponents, -1))

            # First self attention layer -  Social attention
            agentopps_tensor = torch.cat((agent_enc, opponentdyn_enc.view(batch_size, self.num_opponents*self.hidden_size)), dim=-1).\
                view(batch_size, self.num_opponents+1, self.hidden_size).transpose(0, 1)
            socialattn_output, socialattn_weights = self.attention_model1(agentopps_tensor, agentopps_tensor,
                                                                          agentopps_tensor,
                                                                          key_padding_mask=key_padding_mask)
            socialattn_output = socialattn_output + agentopps_tensor  # residual connection

            # Second self attention layer - Policy attention
            ego_intention = torch.zeros(batch_size, self.intention_dim).to(inputs.device)
            agent_enc = torch.cat((agent_enc, ego_intention), dim=-1)
            all_intentions = torch.cat((ego_intention, intention), dim=-1).view(batch_size, self.num_opponents+1, -1).transpose(0, 1)
            socialattn_intention = torch.cat((socialattn_output, all_intentions), dim=-1)
            agent_attn_output, agent_attn_weights = self.attention_model2(agent_enc.unsqueeze(0), socialattn_intention,
                                                                          socialattn_intention,
                                                                          key_padding_mask=key_padding_mask)

            x = agent_attn_output.squeeze(0)
        else:
            x = agent_enc

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
