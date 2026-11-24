import copy
import math

import torch
import torch.nn.functional as F

from diffusion_knobs.targets import make_target

def exists(v):
    return v is not None

def train(
    model,
    path,
    target,
    images,
    sources = None,          # images to start from instead of gaussian noise (flow matching only)
    source_noise = 0.,
    steps = 20_000,
    batch_size = 128,
    lr = 3e-4,
    warmup = 500,
    ema_decay = 0.9995,
    log_every = 1000,
    log = print
):
    # algorithm 1 of the DDPM paper, with the prediction target as a knob

    device = images.device
    ema = copy.deepcopy(model).eval().requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr = lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1., (i + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * i / steps)))

    history, running = [], 0.

    for step in range(steps):
        x0 = images[torch.randint(len(images), (batch_size,), device = device)]

        if exists(sources):
            eps = sources[torch.randint(len(sources), (batch_size,), device = device)]
            eps = eps + source_noise * torch.randn_like(eps)
        else:
            eps = torch.randn_like(x0)

        t, a, s = path.random_times(batch_size, device)
        a, s = a.view(-1, 1, 1, 1), s.view(-1, 1, 1, 1)

        xt = a * x0 + s * eps
        loss = F.mse_loss(model(xt, t), make_target(target, x0, eps, a, s))

        opt.zero_grad(set_to_none = True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        sched.step()

        decay = min(ema_decay, (step + 1) / (step + 10))
        with torch.no_grad():
            for p_ema, p in zip(ema.parameters(), model.parameters()):
                p_ema.lerp_(p, 1. - decay)

        running = running + loss.detach()

        if (step + 1) % log_every == 0 or step == steps - 1:
            avg = (running / (step % log_every + 1)).item()
            history.append((step + 1, avg))
            log(f'step {step + 1:>6}   loss {avg:.4f}')
            running = 0.

    return ema, history
