"""Small explicit test backend, never presented as a π0.5 checkpoint."""
import torch
from torch import nn


class ToyFlow(nn.Module):
    def __init__(self, horizon=10, action_dim=32, state_dim=8, steps=3):
        super().__init__()
        self.horizon, self.action_dim, self.deployment_steps = horizon, action_dim, steps
        self.frozen_prefix = nn.Linear(state_dim, 4, bias=False).requires_grad_(False)
        self.expert = nn.Linear(action_dim + 5, action_dim)

    def prepare_condition(self, observation, train=False):
        with torch.no_grad():
            return torch.tanh(self.frozen_prefix(observation['state']))

    def velocity(self, condition, action, time):
        h = action.shape[1]
        prefix = condition[:, None].expand(-1, h, -1)
        t = time[:, None, None].expand(-1, h, 1).to(action)
        return torch.tanh(self.expert(torch.cat([action, prefix, t], -1)))

    def sample_flow_time(self, batch_size, device):
        return torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0)).sample((batch_size,)).to(device) * .999 + .001

    @torch.no_grad()
    def deployment_sample(self, observation, noise):
        condition = self.prepare_condition(observation, train=False)
        action, t, dt = noise.clone(), 1.0, -1.0 / self.deployment_steps
        while t >= -dt / 2:
            time = torch.full((action.shape[0],), t, device=action.device, dtype=torch.float32)
            action = action + dt * self.velocity(condition, action, time)
            t += dt
        return action
