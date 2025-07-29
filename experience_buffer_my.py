import torch

class ExperienceBuffer:
    """
    Experience buffer for storing rollout data across horizon and environments.

    tensor_dict structure:
    
        - obses:      torch.Size([H, N, O])
            Observations (state vectors)
        - rewards:    torch.Size([H, N, 1])
            Scalar rewards per step
        - values:     torch.Size([H, N, 1])
            Value function estimates
        - returns:    torch.Size([H, N, 1])
            Discounted returns (used for critic target)
        - advs:       torch.Size([H, N, 1])
            Generalized advantage estimates (GAE)
        - neglogpacs: torch.Size([H, N])
            Negative log probability of selected actions
        - dones:      torch.Size([H, N])
            Done flags (1 if episode terminated/truncated)
        - actions:    torch.Size([H, N, A])
            Actions sampled from policy
        - mus:        torch.Size([H, N, A])
            Mean of action distribution
        - sigmas:     torch.Size([H, N, A])
            Std deviation of action distribution

    Notation:
        H = horizon_length (rollout length)
        N = number of parallel environments
        O = observation dimension
        A = action dimension
    """    
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


    def update_step_datas(
            self, 
            step_index:int, 
            obs_tensor:torch.Tensor, 
            done_tensor:torch.Tensor, 
            neglogpacs_tensor:torch.Tensor, 
            value_tensor:torch.Tensor, 
            action_tensor:torch.Tensor, 
            reward_tensor:torch.Tensor
            ):
        """
        Update Step Data

        Args:
            step_index (int):
            obs_tensor (torch.Size([N, O]):
            done_tensor (torch.Size([N]):
            action_tensor (torch.Size([N, A]):
            value_tensor (torch.Size([N, 1]):
            neglogpacs_tensor (torch.Size([N]):
            reward_tensor (torch.Size([N, 1]):
        """
        self.update_step_data('obses', step_index, obs_tensor)
        self.update_step_data('dones', step_index, done_tensor)
        self.update_step_data('actions', step_index, action_tensor)
        self.update_step_data('values', step_index, value_tensor)
        self.update_step_data('neglogpacs', step_index, neglogpacs_tensor.squeeze(-1))
        self.update_step_data('rewards', step_index, reward_tensor)


    def update_step_data(self, name:str, index:int, val:torch.Tensor):
        """
        Update environment batch data (step-level).

        Args:
            name (str):
            index (int):
            val (torch.Size([N, D]) or torch.Size([N])):
                - If dict, updates each sub-key individually.
        """
        if isinstance(val, dict):
            for k, v in val.items():
                self.tensor_dict[name][k][index, :] = v
        else:
            self.tensor_dict[name][index, :] = val


    def update_horizon_data(self, name:str, val:torch.Tensor):
        """
        Update horizon data (trajectory-level).

        Args:
            name (str): Key name in tensor_dict.
            val (Tensor):
                torch.Size([H, N, 1])
                - H = horizon length (timesteps)
                - N = number of environments
                - 1 = scalar feature (e.g., reward, done flag)
                - Typically used to store full rollout data across horizon length.
        """
        self.tensor_dict[name] = val