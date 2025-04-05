import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, envs, actor_layer_size):
        super(Actor, self).__init__()

        self.action_bias = (
            envs.action_space.high + envs.action_space.low
        ) / 2.0
        self.action_bias = torch.Tensor(self.action_bias).view(-1)

        self.action_scale = (
            envs.action_space.high - envs.action_space.low
        ) / 2.0
        self.action_scale = torch.Tensor(self.action_scale).view(-1)

        self.mu = self.action_bias
        self.cov = torch.diag(self.action_scale**2)
        self.mvn = torch.distributions.MultivariateNormal(self.mu, self.cov)

    def forward(self, x):
        out = self.mvn.sample().view(1, -1)
        return out
