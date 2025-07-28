import os
import time
from datetime import datetime

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from model_my import ActorCritic
from experience_buffer_my import ExperienceBuffer, ReplayBuffer

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


class PolicyGradeintAgent:
    def __init__(self, envs:gymnasium.vector.VectorEnv | gymnasium.Env,
                 num_envs:int,
                 num_epochs:int, 
                 horizon_length:int, 
                 batch_size:int, 
                 save_name="",
                 algorithm='ppo',
                 num_mini_epochs=4,
                 iswandb=True
                 ):
        self.envs = envs
        self.env_name = self.envs.spec.id
        self.num_envs = num_envs

        self.algorithm = algorithm              # a2c | ppo
        self.num_mini_epochs = num_mini_epochs  # 재사용 

        self.gamma = 0.99
        self.lam = 0.95
        self.lr = 0.0002
        self.log_std = 0
    
        if isinstance(self.envs, gymnasium.vector.VectorEnv):
            self.state_dim = self.envs.single_observation_space.shape[0]
            self.action_dim = self.envs.single_action_space.shape[0]
        elif isinstance(self.envs, gymnasium.Env):
            self.state_dim = self.envs.observation_space.shape[0]
            self.action_dim = self.envs.action_space.shape[0]
        else:
            raise ValueError(f"Unsupported environment type: {type(self.envs)}")
        self.hidden_dim = 128
        self.model = ActorCritic(self.state_dim, self.action_dim, self.hidden_dim, self.lr, self.log_std)

        self.num_epochs = num_epochs
        self.horizon_length = horizon_length
        self.batch_size = batch_size

        self.replay_buffer = ReplayBuffer()
        self.experience_buffer = ExperienceBuffer(self.state_dim, 
                                                  self.action_dim, 
                                                  self.num_envs, 
                                                  self.horizon_length
                                                  )

        self.loadpoint = 0

        self.iswandb = iswandb

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_name = save_name

        self.reset_times = []
        self.sample_times = []
        self.update_times = []


        self.obs_npy = np.zeros((self.num_envs, self.state_dim))
        self.dones_npy = np.zeros(self.num_envs, dtype=bool)


    def init_wandb(self):
        if self.iswandb:
            wandb.init(
                project="my-rl-project",
                name=f"PG-{self.env_name}_{self.num_envs}_{self.timestamp}_{self.save_name}",
                config={
                    "env_name": self.env_name,
                    "algorithm": "PG",
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "lambda": self.lam,
                    "batch_size": self.batch_size,
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



    def eval_old(self):
        obs_npy, _ = self.envs.reset(seed=123)
        done = False
        episode_length = 0
        while(True):
            obs = torch.tensor(obs_npy, dtype=torch.float32).unsqueeze(0)  # (1, state_dim)
            action = self.model.actor.get_eval_action(obs)

            next_obs_npy, reward, terminated, truncated, info = self.envs.step(action)
            done = terminated or truncated
            episode_length += 1

            obs_npy = next_obs_npy

            if done:
                print("episode length :",episode_length)
                episode_length = 0
                obs_npy, _ = self.envs.reset()


    def save(self, epoch:int):
        dir_path = f"run/{self.env_name}_{self.num_envs}_{self.timestamp}_{self.save_name}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        path = dir_path + "/" + str(epoch) + ".pth"
        torch.save(self.model.state_dict(), path)


    def load(self, loadpath:str):
        state_dict = torch.load(loadpath, weights_only=True)
        # print(state_dict['obs_mean_std.running_var'].shape)

        for key in ["obs_mean_std.running_mean", "obs_mean_std.running_var"]:
            if key in state_dict and state_dict[key].ndim == 2:  # [num_envs, obs_dim] 형태일 때만
                state_dict[key] = state_dict[key].mean(dim=0)    # [obs_dim] 으로 변경

        if "obs_mean_std.count" in state_dict and state_dict["obs_mean_std.count"].ndim != 0:
            state_dict["obs_mean_std.count"] = torch.tensor(1.0)

        # print(state_dict['obs_mean_std.running_var'].shape)
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
            # SAVE
            if epoch % 500 == 0:
                self.save(self.loadpoint + epoch)

        self.save(self.loadpoint + self.num_epochs - 1)

        # print(f"envs reset() : {np.mean(self.reset_times):.6f}sec")
        # print(f"epoch sample : {np.mean(self.sample_times):.6f}sec")
        # print(f"epoch update : {np.mean(self.update_times):.6f}sec")


    def train_epoch(self):
        # Buffer Init
        # start_time = time.time()
        self.experience_buffer._init_from_env_info()
        # reset_time = time.time() - start_time
        # self.reset_times.append(reset_time)

        # Play Steps
        # start_time = time.time()
        self.set_eval()
        with torch.no_grad():
            epoch_rewards = self.play_steps()
        # sample_time = time.time() - start_time
        # self.sample_times.append(sample_time)

        # Update Model
        # start_time = time.time()
        self.set_train()
        obs_tensor = self.experience_buffer.tensor_dict['obses']
        fl_obs_tensor = flatten(obs_tensor)
        prev_actions = self.experience_buffer.tensor_dict['actions']
        fl_prev_actions = flatten(prev_actions)
        
        if self.algorithm == 'a2c':
            self.calc_gradients(fl_obs_tensor, fl_prev_actions)
        elif self.algorithm == 'ppo':
            for n in range(self.num_mini_epochs):
                self.calc_gradients(fl_obs_tensor, fl_prev_actions)

                if n == 0:
                    self.model.obs_mean_std.eval()
                    self.model.value_mean_std.eval()


        # if isinstance(self.envs, gymnasium.vector.VectorEnv):
        #     self.update_from_experience_buffer()
        # else:
        #     self.update_from_replay_buffer()
        # update_time = time.time() - start_time
        # self.update_times.append(update_time)

        return epoch_rewards


    # num_envs = 1024
    def play_steps(self):
        epoch_rewards = torch.zeros(self.num_envs)

        for t in range(self.horizon_length):
            # Update now obs, dones
            obs_tensor = torch.tensor(self.obs_npy, dtype=torch.float32)
            done_tensor = torch.tensor(self.dones_npy, dtype=torch.float32)
            self.experience_buffer.update_data('obses', t, obs_tensor)
            self.experience_buffer.update_data('dones', t, done_tensor)
            
            # Network
            neglogpacs_tensor, value_tensor, action_tensor, mu, sigma = self.model.forward_eval(obs_tensor)

            self.experience_buffer.update_data('actions', t, action_tensor)
            self.experience_buffer.update_data('values', t, value_tensor)
            self.experience_buffer.update_data('neglogpacs', t, neglogpacs_tensor.squeeze(-1))

            # Step
            self.obs_npy, rewards, terminations, truncations, _ = self.envs.step(action_tensor.squeeze(0).numpy())
            self.dones_npy = (terminations | truncations)

            reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            self.experience_buffer.update_data('rewards', t, reward_tensor)

            # Epoch Rewards
            epoch_rewards += reward_tensor.squeeze(-1)


        last_obs_tensor = torch.tensor(self.obs_npy, dtype=torch.float32)
        _, last_value_tensor, _, _, _ = self.model.forward_eval(last_obs_tensor)
        last_done_tensor = torch.tensor(self.dones_npy, dtype=torch.float32)

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(last_done_tensor, last_value_tensor, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        self.experience_buffer.upadte_data_val('returns', mb_returns)
        self.experience_buffer.upadte_data_val('advs', mb_advs)

        return epoch_rewards.mean().item()


    def calc_gradients(self, obs_tensor:torch.Tensor, prev_actions:torch.Tensor):
        '''
        obs_tensor : (HXN, obs_dim)
        prev_actions : (HXN, act_dim)
        '''
        neglogp_new, values_new, mu, sigma = self.model.forward(obs_tensor, prev_actions)
        # neglogp_new : HXN
        # values_new : HXN, 1
        advantages = self.experience_buffer.tensor_dict['advs']         # H, N, 1
        returns = self.experience_buffer.tensor_dict['returns']         # H, N, 1
        neglogp_old = self.experience_buffer.tensor_dict['neglogpacs']  # H, N
        fl_advantages = flatten(advantages)             # HxN, 1
        fl_return = flatten(returns)                    # HXN, 1
        fl_neglogp_old = flatten(neglogp_old)           # HXN


        if self.algorithm == 'a2c':
            a_loss, c_loss = self.calc_losses_a2c(neglogp_new, values_new, fl_advantages, fl_return) # A2C
        elif self.algorithm == 'ppo':
            a_loss, c_loss = self.calc_losses_ppo(neglogp_new, fl_neglogp_old, values_new, fl_advantages, fl_return) # PPO
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        self.update_actor_critic(a_loss, c_loss)


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
        return a_loss, c_loss


    def discount_values(self, last_dones, last_values, dones, values, rewards):
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



    ############### Legacy

    def old_train_epoch(self):
        # Buffer Init
        # start_time = time.time()
        self.experience_buffer._init_from_env_info()
        # reset_time = time.time() - start_time
        # self.reset_times.append(reset_time)

        # Play Epoch
        # start_time = time.time()
        if isinstance(self.envs, gymnasium.vector.VectorEnv):
            epoch_rewards = self.play_steps()
        else:
            epoch_rewards = self.play_one_epoch_env(self.obs_npy)
        # sample_time = time.time() - start_time
        # self.sample_times.append(sample_time)

        # Update Model
        # start_time = time.time()
        if isinstance(self.envs, gymnasium.vector.VectorEnv):
            self.update_from_experience_buffer()
        else:
            self.update_from_replay_buffer()
        # update_time = time.time() - start_time
        # self.update_times.append(update_time)

        return epoch_rewards


    def update_from_experience_buffer(self):
        # ----- Actor Loss -----
        mb_advs_detached = self.experience_buffer.tensor_dict['advs'].detach()
        mb_neglogpacs = self.experience_buffer.tensor_dict['neglogpacs']
        a_loss = (mb_neglogpacs * mb_advs_detached.squeeze(-1)).mean()

        self.model.actor_optim.zero_grad()
        a_loss.backward()
        self.model.actor_optim.step()

        # ----- Critic Loss -----
        mb_returns = self.experience_buffer.tensor_dict['returns'].detach()
        mb_values = self.experience_buffer.tensor_dict['values']
        c_loss = F.mse_loss(mb_values, mb_returns)

        self.model.critic_optim.zero_grad()
        c_loss.backward()
        self.model.critic_optim.step()


    def update_from_replay_buffer(self, batch_size:int):
        obs, next_obs, neglogpacs, reward, done = self.replay_buffer.sample(batch_size)

        value = self.model.critic(obs)                          # (B,1)
        notdone = (1.0 - done).unsqueeze(1)

        with torch.no_grad():
            next_value = self.model.critic(next_obs)            # (B, 1)
            target = reward + self.gamma * next_value * notdone  # (B, 1)

        critic_loss = F.mse_loss(value, target)                 # ()

        advantage = (target - value).detach()                   # (B, 1)
        actor_loss = (neglogpacs * advantage).mean()             # ()

        # Backpropagation
        self.model.actor_optim.zero_grad()
        actor_loss.backward()
        self.model.actor_optim.step()

        self.model.critic_optim.zero_grad()
        critic_loss.backward()
        self.model.critic_optim.step()

    # num_envs = 1
    def play_one_epoch_env(self):
        ''' Get state action, reward transition tuples '''
        epoch_reward = 0
        obs_tensor = torch.tensor(self.obs_npy[np.newaxis, :], dtype=torch.float32)

        for _ in range(self.horizon_length):
            obs, reward, done = self.play_one_step_env(obs_tensor)
            epoch_reward += reward
            if done:
                obs, _ = self.envs.reset()
                obs = torch.tensor(obs[np.newaxis, :], dtype=torch.float32)
        return epoch_reward

    # num_envs = 1
    def play_one_step_env(self, obs:torch.Tensor):
            action, log_prob = self.model.actor.get_action(obs)
            next_obs, reward, terminated, truncated, _ = self.envs.step(action.squeeze(0).numpy())
            done = terminated or truncated

            next_obs = torch.tensor(next_obs[np.newaxis, :], dtype=torch.float32)
            log_prob = log_prob.squeeze(0)
            reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
            done = torch.tensor(done, dtype=torch.bool).unsqueeze(0).float()

            self.replay_buffer.append((obs, next_obs, log_prob, reward, done))
            obs = next_obs
            return obs, reward, done

    ''' add anything (e.g. normalization, etc)'''