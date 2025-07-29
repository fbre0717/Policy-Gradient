import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from running_mean_std_my import RunningMeanStd

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std)

    
    def forward(self, obs_tensor:torch.Tensor):
        '''take state and output action distribution '''
        x = F.relu(self.fc1(obs_tensor))
        x = F.relu(self.fc2(x))
        mu: torch.Tensor = self.mu_head(x)
        std = self.log_std.exp()
        return mu, std


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs_tensor: torch.Tensor):
        x = F.relu(self.fc1(obs_tensor))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, lr, log_std):
        super().__init__()
        self.lr = lr
        self.log_std = log_std

        self.actor = Actor(obs_dim, action_dim, hidden_dim, self.log_std)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(obs_dim, hidden_dim)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.obs_mean_std = RunningMeanStd((obs_dim,))
        self.value_mean_std = RunningMeanStd((1,))


    def forward_eval(self, obs_tensor:torch.Tensor):
        # Eval
        norm_obs_tensor = self.norm_obs(obs_tensor)
        mu, sigma = self.actor(norm_obs_tensor)
        value = self.critic(norm_obs_tensor)
        distr = torch.distributions.Normal(mu, sigma, validate_args=False)

        selected_action = distr.sample()
        if selected_action.dim() == 1:
            selected_action = selected_action.unsqueeze(-1)
        
        neglogp = -distr.log_prob(selected_action).sum(dim=-1)
        return torch.squeeze(neglogp), self.denorm_value(value), selected_action, mu, sigma


    def forward(self, obs_tensor:torch.Tensor, prev_actions:torch.Tensor):
        # Train
        norm_obs_tensor = self.norm_obs(obs_tensor)
        mu, sigma = self.actor(norm_obs_tensor)
        value = self.critic(norm_obs_tensor)
        distr = torch.distributions.Normal(mu, sigma, validate_args=False)

        neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
        return torch.squeeze(neglogp), value, mu, sigma

    def norm_obs(self, observation:torch.Tensor):
        with torch.no_grad():
            return self.obs_mean_std(observation)

    def denorm_value(self, value:torch.Tensor):
        with torch.no_grad():
            return self.value_mean_std(value, denorm=True)

    
    ''' add anything (normalization, etc)'''