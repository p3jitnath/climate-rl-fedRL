import numpy as np
import torch
import torch.nn as nn
from filterpy.kalman import EnsembleKalmanFilter as EnKF


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

        self.state_dim = np.array(envs.single_observation_space.shape).prod()

        self.enkf = EnKF(
            x=np.array(self.action_bias),  # initial guess for [A, B]
            P=np.diag(self.action_scale**2),  # initial covariance
            dim_z=self.state_dim,
            dt=1.0,
            N=100,
            hx=lambda x: x,  # dummy
            fx=lambda x, dt: x,  # dummy
        )

        self.enkf.R = (
            np.eye(self.state_dim) * 1e-5
        )  # adjust based on observation uncertainty
        self.enkf.Q = (
            np.eye(self.action_bias.shape[0]) * 0.5
        )  # allow [A, B] to drift slightly

    def forward(self, x):
        return torch.Tensor(self.enkf.x).view(1, -1)

    def update(self, hx, z):
        self.enkf.hx = lambda x: hx
        self.enkf.predict()
        self.enkf.update(z)
