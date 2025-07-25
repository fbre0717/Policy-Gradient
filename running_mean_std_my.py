import torch
import torch.nn as nn
import numpy as np
import gymnasium

class BaseModelNetwork(nn.Module):
    def __init__(self, obs_shape, normalize_value, normalize_input, value_size):
        nn.Module.__init__(self)
        self.obs_shape = obs_shape
        self.normalize_value = normalize_value
        self.normalize_input = normalize_input
        self.value_size = value_size

        if normalize_value:
            self.value_mean_std = torch.jit.script(RunningMeanStd((self.value_size,)))
        if normalize_input:
            self.running_mean_std = torch.jit.script(RunningMeanStd(obs_shape))

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation) if self.normalize_input else observation

    def denorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, denorm=True) if self.normalize_value else value



class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        in_size = insize
        self.epsilon = epsilon
        
        self.norm_only = norm_only
        self.per_channel = per_channel

        self.axis = [0]


        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))


    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count:int):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count
    

    def forward(self, input, denorm:bool=False):
        if input.dim() == 1:
            input = input.unsqueeze(0)
                    
        if self.training:
            mean = input.mean(self.axis)
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(
                self.running_mean,
                self.running_var,
                self.count,
                mean,
                var,
                input.size(0)
            )

        current_mean = self.running_mean
        current_var = self.running_var

        # get output
        if denorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y