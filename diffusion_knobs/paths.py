import math

import torch
import torch.nn.functional as F

T = 1000

# noise schedules (discrete time, t = 0 .. T)

def linear_alpha_bar(beta_start = 1e-4, beta_end = 0.02):
    betas = torch.linspace(beta_start, beta_end, T, dtype = torch.float64)
    return F.pad(torch.cumprod(1. - betas, dim = 0), (1, 0), value = 1.)

def cosine_alpha_bar(s = 0.008):
    t = torch.linspace(0, 1, T + 1, dtype = torch.float64)
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    betas = (1. - f[1:] / f[:-1]).clamp(max = 0.999)
    return F.pad(torch.cumprod(1. - betas, dim = 0), (1, 0), value = 1.)

def sigmoid_alpha_bar(start = -3, end = 3, tau = 1):
    # Jabri et al. 2023 (RIN); `sigmoid_beta_schedule` in denoising-diffusion-pytorch
    t = torch.linspace(0, 1, T + 1, dtype = torch.float64)
    v_start, v_end = torch.tensor(start / tau).sigmoid(), torch.tensor(end / tau).sigmoid()
    f = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    betas = (1. - f[1:] / f[:-1]).clamp(0., 0.999)
    return F.pad(torch.cumprod(1. - betas, dim = 0), (1, 0), value = 1.)

# a path says how much image (a) and how much noise (s) are in x_t = a * x0 + s * eps

class Diffusion:
    """variance preserving: a^2 + s^2 = 1, integer t, network sees t / T"""

    def __init__(self, alpha_bar):
        self.a = alpha_bar.sqrt()
        self.s = (1. - alpha_bar).clamp(min = 0.).sqrt()
        self.phi = torch.atan2(self.s, self.a)

    def coef(self, t):
        return self.a[t].item(), self.s[t].item()

    def time_input(self, t):
        return t / T

    def random_times(self, batch, device):
        t = torch.randint(1, T + 1, (batch,))
        return (t / T).to(device), self.a[t].float().to(device), self.s[t].float().to(device)

    def timesteps(self, steps):
        return [round(T * (1 - i / steps)) for i in range(steps + 1)]

    def euler_step(self, x, x0, eps, t, t_next):
        a, s = self.coef(t)
        dphi = (self.phi[t_next] - self.phi[t]).item()
        return x + dphi * (a * eps - s * x0)          # d x / d phi = v

class Straight:
    """flow matching / rectified flow: x_t = (1 - t) x0 + t eps, t in [0, 1]"""

    def coef(self, t):
        return 1. - t, t

    def time_input(self, t):
        return t

    def random_times(self, batch, device):
        t = torch.rand(batch, device = device)
        return t, 1. - t, t

    def timesteps(self, steps):
        return [1. - i / steps for i in range(steps + 1)]

    def euler_step(self, x, x0, eps, t, t_next):
        return x + (t_next - t) * (eps - x0)          # d x / d t = eps - x0

PATHS = dict(
    linear = lambda: Diffusion(linear_alpha_bar()),
    cosine = lambda: Diffusion(cosine_alpha_bar()),
    sigmoid = lambda: Diffusion(sigmoid_alpha_bar()),
    straight = Straight,
)

def get_path(name):
    return PATHS[name]()
