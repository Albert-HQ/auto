import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bilevel_maddpg.model import Actor, Critic

# follower agent for unconstrained stackelberg maddpg
class Follower_Bilevel:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() and getattr(args, 'use_cuda', True) else 'cpu')

        # create the network
        self.actor_network = Actor(args, agent_id, 9).to(self.device)
        self.critic_network = Critic(args).to(self.device)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id, 9).to(self.device)
        self.critic_target_network = Critic(args).to(self.device)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

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
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            
        if os.path.exists(self.model_path + '/critic_params.pkl'):
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl', map_location=self.device))

    # soft update
    def _soft_update_target_network(self, tau=None):
        tau = self.args.tau if tau is None else tau
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # update the network (placeholder, real update handled in runner)
    def train(self, transitions, leader_agent=None):
        pass
    
    # select action (condition on leader_action)
    def select_action(self, o, leader_action, noise_rate=0.0, epsilon=0.0):
        self.actor_network.eval()
        with torch.no_grad():
            # concat leader action into observation if policy is conditioned on it externally
            o = torch.tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = self.actor_network(o)
            a = a.squeeze(0).cpu().numpy()
        self.actor_network.train()
        if noise_rate and noise_rate > 0:
            a = a + noise_rate * np.random.randn(*a.shape)
            a = np.clip(a, -self.args.high_action, self.args.high_action)
        return a

    def save_model(self, train_step):
        torch.save(self.actor_network.state_dict(), self.model_path + '/actor_params.pkl')
        torch.save(self.critic_network.state_dict(), self.model_path + '/critic_params.pkl')


