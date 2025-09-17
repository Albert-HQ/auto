import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bilevel_maddpg.model import Actor, Critic, Cost

# follower agent for constrained stackelberg maddpg
class Follower:
    def __init__(self, args, agent_id): 
        self.args = args
        self.agent_id = agent_id
        self.l_multiplier = args.lagrangian_multiplier
        self.cost_threshold = args.cost_threshold
        self.train_step = 0
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() and getattr(args, 'use_cuda', True) else 'cpu')

        # create the network
        self.actor_network = Actor(args, agent_id, 9).to(self.device)
        self.critic_network = Critic(args).to(self.device)
        # 双Q网络（TD3）
        self.critic_network2 = Critic(args).to(self.device)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id, 9).to(self.device)
        self.critic_target_network = Critic(args).to(self.device)
        self.critic_target_network2 = Critic(args).to(self.device)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.critic_target_network2.load_state_dict(self.critic_network2.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        self.critic2_optim = torch.optim.Adam(self.critic_network2.parameters(), lr=self.args.lr_critic)

        # create the cost
        self.cost_network = Cost(args).to(self.device)
        self.cost_target_network = Cost(args).to(self.device)
        self.cost_target_network.load_state_dict(self.cost_network.state_dict())
        self.cost_optim = torch.optim.Adam(self.cost_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # load model
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl', map_location=self.device))
        if os.path.exists(self.model_path + '/critic_params.pkl'):
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl', map_location=self.device))
        if os.path.exists(self.model_path + '/critic2_params.pkl'):
            self.critic_network2.load_state_dict(torch.load(self.model_path + '/critic2_params.pkl', map_location=self.device))
        if os.path.exists(self.model_path + '/cost_params.pkl'):
            self.cost_network.load_state_dict(torch.load(self.model_path + '/cost_params.pkl', map_location=self.device))

    # soft update
    def _soft_update_target_network(self, tau=None):
        tau = self.args.tau if tau is None else tau
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.critic_target_network2.parameters(), self.critic_network2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.cost_target_network.parameters(), self.cost_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # update the network
    def train(self, transitions, leader_agent=None):
        pass
    
    # select action
    def select_action(self, o, leader_action, noise_rate=0.0, epsilon=0.0):
        self.actor_network.eval()
        with torch.no_grad():
            # 拼接领导者动作到观察，形成9维输入
            if isinstance(leader_action, np.ndarray):
                la = leader_action
            else:
                la = np.array([leader_action], dtype=np.float32)
            o_aug = np.concatenate([np.asarray(o, dtype=np.float32), la.reshape(-1)], axis=0)
            o_t = torch.tensor(o_aug, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = self.actor_network(o_t)
            a = a.squeeze(0).cpu().numpy()
        self.actor_network.train()
        if noise_rate and noise_rate > 0:
            a = a + noise_rate * np.random.randn(*a.shape)
            a = np.clip(a, -self.args.high_action, self.args.high_action)
        return a

    def save_model(self, train_step):
        torch.save(self.actor_network.state_dict(), self.model_path + '/actor_params.pkl')
        torch.save(self.critic_network.state_dict(), self.model_path + '/critic_params.pkl')
        torch.save(self.critic_network2.state_dict(), self.model_path + '/critic2_params.pkl')
        torch.save(self.cost_network.state_dict(), self.model_path + '/cost_params.pkl')

