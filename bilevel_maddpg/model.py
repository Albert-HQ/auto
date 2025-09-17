import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id, input_dim):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        if isinstance(args.obs_shape, (list, tuple)):
            obs_dim = int(sum(args.obs_shape))
        else:
            obs_dim = int(args.obs_shape)
        act_dim = sum(args.action_shape) if isinstance(args.action_shape, (list, tuple)) else int(args.action_shape)
        self.fc1 = nn.Linear(obs_dim + act_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action, dim=1):
        # state: [B, obs_dim] 或 [obs_dim]
        # action: List[Tensor] 或 Tensor（若为List则拼接）
        if isinstance(action, (list, tuple)):
            action = [a.float() for a in action]
            action = torch.cat(action, dim=dim)
        x = torch.cat([state.float(), action.float()], dim=dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value


class Cost(nn.Module):
    def __init__(self, args):
        super(Cost, self).__init__()
        self.max_action = args.high_action
        if isinstance(args.obs_shape, (list, tuple)):
            obs_dim = int(sum(args.obs_shape))
        else:
            obs_dim = int(args.obs_shape)
        act_dim = sum(args.action_shape) if isinstance(args.action_shape, (list, tuple)) else int(args.action_shape)
        self.fc1 = nn.Linear(obs_dim + act_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action, dim=1):
        if isinstance(action, (list, tuple)):
            action = [a.float() for a in action]
            action = torch.cat(action, dim=dim)
        x = torch.cat([state.float(), action.float()], dim=dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value


class Critic_Discrete(nn.Module):
    """
    Centralized discrete Q critic: input is global state concat and joint action one-hot concat.
    Outputs scalar Q(s, a1, a2).
    """
    def __init__(self, args, hidden: int = 128):
        super(Critic_Discrete, self).__init__()
        obs_dim = sum(args.obs_shape) if isinstance(args.obs_shape, (list, tuple)) else int(args.obs_shape)
        act_dim = sum(getattr(args, 'action_dim', [])) if hasattr(args, 'action_dim') else 0
        if act_dim == 0:
            # fallback to action_shape if provided
            act_dim = sum(args.action_shape) if isinstance(args.action_shape, (list, tuple)) else int(args.action_shape)
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q_out = nn.Linear(hidden, 1)

    def forward(self, state, action_onehot):
        # state: [B, obs_dim]; action_onehot: [B, act_dim]
        x = torch.cat([state.float(), action_onehot.float()], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_out(x)


class Cost_Discrete(nn.Module):
    """
    Centralized discrete cost critic: same input as Critic_Discrete.
    """
    def __init__(self, args, hidden: int = 128):
        super(Cost_Discrete, self).__init__()
        obs_dim = sum(args.obs_shape) if isinstance(args.obs_shape, (list, tuple)) else int(args.obs_shape)
        act_dim = sum(getattr(args, 'action_dim', [])) if hasattr(args, 'action_dim') else 0
        if act_dim == 0:
            act_dim = sum(args.action_shape) if isinstance(args.action_shape, (list, tuple)) else int(args.action_shape)
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q_out = nn.Linear(hidden, 1)

    def forward(self, state, action_onehot):
        x = torch.cat([state.float(), action_onehot.float()], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_out(x)
