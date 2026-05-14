import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    
    def __init__(self, dim_in, dim_out, hidden, layers, act=nn.ELU):
        super().__init__()
        net = []
        d = dim_in
        for _ in range(layers):
            net += [nn.Linear(d, hidden), nn.LayerNorm(hidden), act()]
            d = hidden
        net.append(nn.Linear(d, dim_out))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class Pixelencoder(nn.Module):
    
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class PixelBCActor(nn.Module):
 
    
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(
        self, image_size, embed_dim, action_dim, hidden, layers, goal_dim=2
    ):
        super().__init__()
        self.encoder = Pixelencoder(embed_dim)
        self.head = MLP(embed_dim + goal_dim, 2 * action_dim, hidden, layers)
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.register_buffer('goal_scale', torch.tensor(float(image_size)))

    @property
    def trunk(self):
        raise NotImplementedError

    def encode(self, pixels):
        return self.encoder(pixels)

    def _norm_goal(self, goal):
        return (goal / self.goal_scale) * 2.0 - 1.0

    def forward(self, pixels, goal):
        z = self.encoder(pixels)
        if goal.ndim == 3:
            goal = goal.squeeze(1)

        g = self._norm_goal(goal)
        h = torch.cat([z, g], dim=-1)
        mu, log_std = self.head(h).chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        u = normal.rsample()
        a = torch.tanh(u)

        log_p = (normal.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)

        return a, log_p, -log_p, mu


class GaussianActor(nn.Module):

    
    LOG_STD_MIN, LOG_STD_MAX = -2.0, 2.0

    def __init__(
        self, embed_dim, action_dim, hidden, layers, goal_dim=2, goal_scale=224
    ):
        super().__init__()
        self.trunk = MLP(embed_dim + goal_dim, 2 * action_dim, hidden, layers)
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.register_buffer('goal_scale', torch.tensor(float(goal_scale)))

    def _norm_goal(self, goal):
        return (goal / self.goal_scale) * 2.0 - 1.0

    def forward(self, s, goal):
        if goal.ndim == s.ndim + 1:
            goal = goal.squeeze(1)
        goal = self._norm_goal(goal)

        h = torch.cat([s, goal], dim=-1)

        mu, log_std = self.trunk(h).chunk(2, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        u = normal.rsample()
        a = torch.tanh(u)
        log_p = (normal.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
        entropy = -log_p
        return a, log_p, entropy, mu


class Critic(nn.Module):
    
    def __init__(self, embed_dim, hidden, layers):
        super().__init__()
        self.v = MLP(embed_dim, 1, hidden, layers)
        self.v_target = MLP(embed_dim, 1, hidden, layers)
        self.v_target.load_state_dict(self.v.state_dict())
        self.v_target.requires_grad_(False)

    def forward(self, s):
        return self.v(s).squeeze(-1), self.v_target(s).squeeze(-1)

    def sync_target(self, tau=0.005):
        """Soft update target network."""
        for p, pt in zip(self.v.parameters(), self.v_target.parameters()):
            pt.data.lerp_(p.data, tau)
