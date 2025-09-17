import os
import numpy as np
import torch

from bilevel_maddpg.model import Actor, Critic
from bilevel_maddpg.utils import agent_obs_dim, safe_load_state_dict

# leader agent for unconstrained stackelberg maddpg
class Leader_Bilevel:
    def __init__(self, args, agent_id): 
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() and getattr(args, 'use_cuda', True) else 'cpu')

        actor_input_dim = agent_obs_dim(args, agent_id)
        # create the network
        self.actor_network = Actor(args, agent_id, actor_input_dim).to(self.device)
        self.critic_network = Critic(args).to(self.device)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id, actor_input_dim).to(self.device)
        self.critic_target_network = Critic(args).to(self.device)

        self._sync_target_networks()

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        os.makedirs(self.args.save_dir, exist_ok=True)

        # path to save the model
        self.model_path = self.args.save_dir + '/' + 'agent_%d' % agent_id
        os.makedirs(self.model_path, exist_ok=True)

        # load model
        actor_loaded = safe_load_state_dict(
            self.actor_network,
            self.model_path + '/actor_params.pkl',
            self.device,
            f'leader-bilevel-{agent_id} actor'
        )

        critic_loaded = safe_load_state_dict(
            self.critic_network,
            self.model_path + '/critic_params.pkl',
            self.device,
            f'leader-bilevel-{agent_id} critic'
        )

        if actor_loaded or critic_loaded:
            self._sync_target_networks()

    def _sync_target_networks(self):
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

    # soft update
    def _soft_update_target_network(self, tau=None):
        tau = self.args.tau if tau is None else tau
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # update the network (placeholder, real update handled in runner)
    def train(self, transitions, follower_agent=None):
        pass
    
    # select action
    def select_action(self, o, noise_rate=0.0, epsilon=0.0):
        self.actor_network.eval()
        with torch.no_grad():
            o = torch.tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = self.actor_network(o)
            a = a.squeeze(0).cpu().numpy()
        self.actor_network.train()
        # add exploration noise if provided
        if noise_rate and noise_rate > 0:
            a = a + noise_rate * np.random.randn(*a.shape)
            a = np.clip(a, -self.args.high_action, self.args.high_action)
        return a

    def save_model(self, train_step):
        torch.save(self.actor_network.state_dict(), self.model_path + '/actor_params.pkl')
        torch.save(self.critic_network.state_dict(), self.model_path + '/critic_params.pkl')
