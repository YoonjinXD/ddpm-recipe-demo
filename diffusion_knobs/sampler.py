import math

import torch

from diffusion_knobs.targets import to_x0_eps

def exists(v):
    return v is not None

@torch.no_grad()
def sample(
    model,
    path,
    target,
    x,                       # where sampling starts: gaussian noise, or anything for flow matching
    steps,
    method = 'ddim',         # 'ddpm' (fresh noise every step), 'ddim' (deterministic), 'euler' (follow the velocity)
    clip = False,            # clamp the clean-image guess to [-1, 1], as most image codebases do
    generator = None,
    callback = None          # callback(i, t, x, x0, eps) - used to record the frames for the website
):
    times = path.timesteps(steps)

    for i, (t, t_next) in enumerate(zip(times[:-1], times[1:])):
        a, s = path.coef(t)
        a_next, s_next = path.coef(t_next)

        out = model(x, torch.full((len(x),), path.time_input(t), device = x.device))
        x0, eps = to_x0_eps(target, out, x, a, s)

        if clip:
            x0 = x0.clamp(-1., 1.)
            eps = (x - a * x0) / s

        if exists(callback):
            callback(i, t, x, x0, eps)

        if method == 'euler':
            x = path.euler_step(x, x0, eps, t, t_next)

        elif method == 'ddpm' and s_next > 0:
            c = s_next / s * math.sqrt(max(s ** 2 - (a / a_next * s_next) ** 2, 0.))
            noise = torch.randn(x.shape, generator = generator).to(x.device)
            x = a_next * x0 + math.sqrt(s_next ** 2 - c ** 2) * eps + c * noise

        else:
            x = a_next * x0 + s_next * eps

    if exists(callback):
        callback(steps, times[-1], x, x, torch.zeros_like(x))

    return x
