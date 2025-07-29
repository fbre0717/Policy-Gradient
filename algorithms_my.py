import os
import time
from datetime import datetime

import gymnasium
import numpy as np
import torch
import wandb

from model_my import ActorCritic
from experience_buffer_my import ExperienceBuffer

def flatten(tensor: torch.Tensor):
    """
    Input:
        - (H, N, D) -> (H*N, D)
        - (H, N)    -> (H*N,)
    """
    if tensor is None:
        return None
    if tensor.ndim == 3:
        H, N, D = tensor.shape
        return tensor.reshape(H * N, D)
    elif tensor.ndim == 2:
        H, N = tensor.shape
        return tensor.reshape(H * N)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

class MiniBatchDataset:
    def __init__(self, obs, actions, advantages, returns, neglogp_old, minibatch_size):
        self.obs = obs
        self.actions = actions
        self.advantages = advantages
        self.returns = returns
        self.neglogp_old = neglogp_old
        self.minibatch_size = minibatch_size

        self.indices = torch.arange(self.obs.shape[0])
        self.apply_permutation()

    def __len__(self):
        return len(self.indices) // self.minibatch_size

    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = start + self.minibatch_size
        mb_idx = self.indices[start:end]
        return {
            "obs": self.obs[mb_idx],
            "actions": self.actions[mb_idx],
            "advantages": self.advantages[mb_idx],
            "returns": self.returns[mb_idx],
            "neglogp_old": self.neglogp_old[mb_idx],
        }

    def apply_permutation(self):
        self.indices = torch.randperm(self.obs.shape[0])


class PolicyGradeintAgent:
    def __init__(
            self, 
            envs:gymnasium.vector.VectorEnv | gymnasium.Env,
            num_envs:int,
            num_epochs:int, 
            horizon_length:int, 
            batch_size:int, 
            save_name="",
            algorithm='PPO',
            num_mini_epochs=5,
            num_mini_batch=1,
            iswandb=True,
            issave=True
            ):
        self.envs = envs
        self.env_name = self.envs.spec.id
        self.num_envs = num_envs
        self.horizon_length = horizon_length
        self.num_epochs = num_epochs

        self.batch_size = batch_size
        self.init_algorithm(algorithm, num_mini_epochs, num_mini_batch)


        self.gamma = 0.99
        self.lam = 0.95
        self.lr = 3e-4
        self.log_std = 0
    
        self.init_dims_from_env()
        self.hidden_dim = 128
        self.model = ActorCritic(self.state_dim, self.action_dim, self.hidden_dim, self.lr, self.log_std)

        self.experience_buffer = ExperienceBuffer(
            self.state_dim, 
            self.action_dim, 
            self.num_envs, 
            self.horizon_length
            )

        self.loadpoint = 0

        self.iswandb = iswandb
        self.issave = issave

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_name = save_name

        self.reset_times = []
        self.sample_times = []
        self.update_times = []


        self.obs_npy = np.zeros((self.num_envs, self.state_dim))
        self.dones_npy = np.zeros(self.num_envs, dtype=bool)


    def init_algorithm(self, algorithm, num_mini_epochs, num_mini_batch):
        self.algorithm = algorithm
        if self.algorithm == 'A2C':
            self.num_mini_epochs = 1
            self.num_mini_batch = 1
        elif self.algorithm == 'PPO':
            self.num_mini_epochs = num_mini_epochs
            self.num_mini_batch = num_mini_batch
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")            
        self.mini_batch_size = self.batch_size // self.num_mini_batch


    def init_dims_from_env(self):
        if isinstance(self.envs, gymnasium.vector.VectorEnv):
            self.state_dim = self.envs.single_observation_space.shape[0]
            self.action_dim = self.envs.single_action_space.shape[0]
        elif isinstance(self.envs, gymnasium.Env):
            self.state_dim = self.envs.observation_space.shape[0]
            self.action_dim = self.envs.action_space.shape[0]
        else:
            raise ValueError(f"Unsupported environment type: {type(self.envs)}")


    def init_wandb(self):
        if self.iswandb:
            wandb.init(
                project="my-rl-project",
                name=f"{self.algorithm}-{self.env_name}_{self.num_envs}_{self.timestamp}_{self.save_name}",
                config={
                    "env_name": self.env_name,
                    "algorithm": self.algorithm,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "lambda": self.lam,
                    "batch_size": self.batch_size,
                    "horizon_length": self.horizon_length,
                    "num_envs": self.num_envs,
                    "epochs": self.num_epochs,
                }
            )
    

    def record_wandb(self, epoch, epoch_rewards):
        if self.iswandb:
            wandb.log({
                    "reward": epoch_rewards,
                    "epoch": epoch,
                })


    def eval(self):
        # Reset Env
        self.obs_npy, _ = self.envs.reset(seed=123)
        self.dones_npy = np.zeros(self.num_envs, dtype=bool)

        while True:
            self.set_eval()
            with torch.no_grad():
                epoch_rewards = torch.zeros(self.num_envs)

                for t in range(self.horizon_length):
                    obs_tensor = torch.tensor(self.obs_npy[np.newaxis, :], dtype=torch.float32)
                    neglogpacs_tensor, value_tensor, action_tensor, mu, sigma = self.model.forward_eval(obs_tensor)
                    self.obs_npy, rewards, terminations, truncations, _ = self.envs.step(action_tensor.squeeze(0).numpy())
                    reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0)
                    epoch_rewards += reward_tensor

            print("epoch rewards :", epoch_rewards.mean().item())


    def save(self, epoch:int):
        if self.issave:
            dir_path = f"run/{self.env_name}_{self.num_envs}_{self.timestamp}_{self.save_name}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            path = dir_path + "/" + str(epoch) + ".pth"
            torch.save(self.model.state_dict(), path)


    def load(self, loadpath:str):
        state_dict = torch.load(loadpath, weights_only=True)

        for key in ["obs_mean_std.running_mean", "obs_mean_std.running_var"]:
            if key in state_dict and state_dict[key].ndim == 2:  # [num_envs, obs_dim] 형태일 때만
                state_dict[key] = state_dict[key].mean(dim=0)    # [obs_dim] 으로 변경

        if "obs_mean_std.count" in state_dict and state_dict["obs_mean_std.count"].ndim != 0:
            state_dict["obs_mean_std.count"] = torch.tensor(1.0)

        self.model.load_state_dict(state_dict)
        print("Load Complete :", loadpath)


    def set_eval(self):
        self.model.eval()


    def set_train(self):
        self.model.train()


    def train(self):
        # Reset Env
        self.init_wandb()
        self.obs_npy, _ = self.envs.reset(seed=123)
        self.dones_npy = np.zeros(self.num_envs, dtype=bool)

        for epoch in range(self.num_epochs):
            epoch_rewards = self.train_epoch()
            print("epoch", epoch, ":", epoch_rewards)
            self.record_wandb(epoch, epoch_rewards)

            if epoch % 500 == 0:
                self.save(self.loadpoint + epoch)

        self.save(self.loadpoint + self.num_epochs - 1)


    def train_epoch(self):
        self.experience_buffer._init_from_env_info()

        self.set_eval()
        # Sample
        with torch.no_grad():
            epoch_rewards = self.play_steps()

        # Prepare Dataset
        dataset = MiniBatchDataset(
            flatten(self.experience_buffer.tensor_dict['obses']),              # HXN, O
            flatten(self.experience_buffer.tensor_dict['actions']),            # HXN, A
            flatten(self.experience_buffer.tensor_dict['advs']),               # HxN, 1
            flatten(self.experience_buffer.tensor_dict['returns']),            # HXN, 1
            flatten(self.experience_buffer.tensor_dict['neglogpacs']),         # HXN
            self.mini_batch_size
        )

        # Train
        self.set_train()
        for mini_epoch in range(self.num_mini_epochs):  # Mini Epoch
            dataset.apply_permutation()
            for i in range(len(dataset)):               # Mini Batch
                batch = dataset[i]
                self.calc_gradients(
                    batch["obs"],
                    batch["actions"],
                    batch["advantages"],
                    batch["returns"],
                    batch["neglogp_old"],
                )

            if mini_epoch == 0:
                self.model.obs_mean_std.eval()
                self.model.value_mean_std.eval()

        return epoch_rewards


    def play_steps(self):
        '''Sample and Update '''
        epoch_rewards = torch.zeros(self.num_envs)

        for step_index in range(self.horizon_length):
            # Update Buffer now obs, dones
            obs_tensor = torch.tensor(self.obs_npy, dtype=torch.float32)
            done_tensor = torch.tensor(self.dones_npy, dtype=torch.float32)

            # Network Forward(Eval)
            neglogpacs_tensor, value_tensor, action_tensor, mu, sigma = self.model.forward_eval(obs_tensor)

            # Env Step
            self.obs_npy, rewards, terminations, truncations, _ = self.envs.step(action_tensor.squeeze(0).numpy())
            self.dones_npy = (terminations | truncations)

            reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

            # Epoch Rewards
            epoch_rewards += reward_tensor.squeeze(-1)

            # Update Buffer
            self.experience_buffer.update_step_datas(step_index, obs_tensor, done_tensor, neglogpacs_tensor, value_tensor, action_tensor, reward_tensor)


        last_obs_tensor = torch.tensor(self.obs_npy, dtype=torch.float32)
        last_done_tensor = torch.tensor(self.dones_npy, dtype=torch.float32)
        _, last_value_tensor, _, _, _ = self.model.forward_eval(last_obs_tensor)

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(last_done_tensor, last_value_tensor, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        self.experience_buffer.update_horizon_data('returns', mb_returns)
        self.experience_buffer.update_horizon_data('advs', mb_advs)

        return epoch_rewards.mean().item()


    def calc_gradients(self, obs_tensor, prev_actions, advantages, returns, neglogp_old):
        neglogp_new, values_new, mu, sigma = self.model.forward(obs_tensor, prev_actions)
        if self.algorithm == 'A2C':
            a_loss, c_loss = self.calc_losses_a2c(neglogp_new, values_new, advantages, returns) # A2C
            ratio = 1
        elif self.algorithm == 'PPO':
            a_loss, c_loss, ratio = self.calc_losses_ppo(neglogp_new, neglogp_old, values_new, advantages, returns)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")            
        self.update_actor_critic(a_loss, c_loss)
        return a_loss, c_loss, ratio


    def update_actor_critic(self, a_loss, c_loss):
        self.model.actor_optim.zero_grad()
        a_loss.backward()
        self.model.actor_optim.step()
        self.model.critic_optim.zero_grad()
        c_loss.backward()
        self.model.critic_optim.step()


    def calc_losses_a2c(self, neglogp, values, advantages, returns):
        # nelogp : HXN
        # advantages : HXN, 1
        # values : HXN, 1
        # returns : HXN, 1
        a_loss = (neglogp * advantages.squeeze(1)).mean()
        c_loss = ((returns - values)**2).mean()
        return a_loss, c_loss


    def calc_losses_ppo(self, neglogp_new, neglogp_old, values, advantages, returns, curr_e_clip=0.2):
        ratio = torch.exp(neglogp_old - neglogp_new)
        surr1 = advantages.squeeze(1) * ratio
        surr2 = advantages.squeeze(1) * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2).mean()
        c_loss = ((returns - values) ** 2).mean()
        return a_loss, c_loss, ratio


    def discount_values(self, last_dones, last_values, dones, values, rewards):
        '''Calculate GAE(Generalized Advantage Estimation)'''
        lastgaelam = 0
        mb_advs = torch.zeros_like(rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                notdones = 1.0 - last_dones
                next_values = last_values
            else:
                notdones = 1.0 - dones[t+1]
                next_values = values[t+1]
            notdones = notdones.unsqueeze(1)

            delta = rewards[t] + self.gamma * next_values * notdones - values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * notdones * lastgaelam
        return mb_advs
