import torch
from collections import deque
import random


class ExperienceBuffer:
    def __init__(self, obs_dim:int, action_dim:int, num_envs:int, horizon_length=16, value_size=1, device="cpu"):
        self.horizon_length = horizon_length
        self.num_envs = num_envs
        self.obs_base_shape = (self.horizon_length, self.num_envs)

        self.obs_shape = (obs_dim,)
        self.action_shape = (action_dim,)
        self.val_shape = (value_size,)

        self.device = device

        self.tensor_dict = {}
        self._init_from_env_info()

    def _init_from_env_info(self):
        # observations: (horizon_length, num_envs, obs_dim)
        self.tensor_dict['obses'] = torch.zeros(
            self.obs_base_shape + self.obs_shape,
            dtype=torch.float32,
            device=self.device
        )

        # rewards: (horizon_length, num_envs, value_size)
        self.tensor_dict['rewards'] = torch.zeros(
            self.obs_base_shape + self.val_shape,
            dtype=torch.float32,
            device=self.device
        )

        # V function: (horizon_length, num_envs, value_size)
        self.tensor_dict['values'] = torch.zeros(
            self.obs_base_shape + self.val_shape,
            dtype=torch.float32,
            device=self.device
        )

        # Q function: (horizon_length, num_envs, value_size)
        self.tensor_dict['returns'] = torch.zeros(
            self.obs_base_shape + self.val_shape,
            dtype=torch.float32,
            device=self.device
        )

        # A function: (horizon_length, num_envs, value_size)
        self.tensor_dict['advs'] = torch.zeros(
            self.obs_base_shape + self.val_shape,
            dtype=torch.float32,
            device=self.device
        )

        # neglogpacs: (horizon_length, num_envs)
        self.tensor_dict['neglogpacs'] = torch.zeros(
            self.obs_base_shape,  # 스칼라
            dtype=torch.float32,
            device=self.device
        )
        
        # dones: (horizon_length, num_envs)
        self.tensor_dict['dones'] = torch.zeros(
            self.obs_base_shape,  # 스칼라
            dtype=torch.float32,
            device=self.device
        )



        # actions: (horizon_length, num_envs, action_dim)
        self.tensor_dict['actions'] = torch.zeros(
            self.obs_base_shape + self.action_shape,
            dtype=torch.float32,
            device=self.device
        )

        # mus: (horizon_length, num_envs, action_dim)
        self.tensor_dict['mus'] = torch.zeros(
            self.obs_base_shape + self.action_shape,
            dtype=torch.float32,
            device=self.device
        )

        # sigmas: (horizon_length, num_envs, action_dim)
        self.tensor_dict['sigmas'] = torch.zeros(
            self.obs_base_shape + self.action_shape,
            dtype=torch.float32,
            device=self.device
        )


    def update_data(self, name, index, val):
        if isinstance(val, dict):
            for k, v in val.items():
                self.tensor_dict[name][k][index, :] = v
        else:
            self.tensor_dict[name][index, :] = val

    def upadte_data_val(self, name, val):
        self.tensor_dict[name] = val
 


 
class ReplayBuffer(deque):
    """
    Tuple:(
            obs:        torch.Tensor, shape (1, obs_dim)
            next_obs:   torch.Tensor, shape (1, obs_dim)
            log_prob:   torch.Tensor, shape (1,)
            reward:     torch.Tensor, shape (1,)
            done:       torch.Tensor, shape (1,)
        )
    """
    def __init__(self, capacity=1000000):
        super().__init__(maxlen=capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ReplayBuffer에서 무작위로 batch_size만큼의 transition을 샘플링하여 반환합니다.

        Parameters
        ----------
        batch_size : int
            샘플링할 transition의 개수.

        Returns
        -------
        obs : torch.Tensor
            shape (batch_size, obs_dim)
        next_obs : torch.Tensor
            shape (batch_size, obs_dim)
        log_prob : torch.Tensor
            shape (batch_size, 1)
        reward : torch.Tensor
            shape (batch_size, 1)
        done : torch.Tensor
            shape (batch_size, 1)
        """
        if batch_size > len(self):
            raise ValueError(f"Requested batch_size={batch_size} but buffer has only {len(self)} items.")

        batch = random.sample(self, batch_size)
        obs, next_obs, log_prob, reward, done = zip(*batch)

        return (
            torch.cat(obs, dim=0),
            torch.cat(next_obs, dim=0),
            torch.stack(log_prob),
            torch.stack(reward),
            torch.stack(done),
        )
