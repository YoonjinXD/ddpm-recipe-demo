"""run every setting shown on the website and write docs/data/"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusion_knobs import CONFIGS, Unet, get_path, make_target, mnist, sample, to_x0_eps
from diffusion_knobs.metrics import Classifier, Scorer

DIFFUSION = [f'{path}-{target}' for path in ('linear', 'cosine', 'sigmoid') for target in ('eps', 'x0', 'v')]
SAMPLERS = dict(ddpm1000 = ('ddpm', 1000), ddim50 = ('ddim', 50), ddim10 = ('ddim', 10), ddim5 = ('ddim', 5))
FLOW_STEPS = (1, 2, 5, 10, 50)

DEMO_DIGITS = (7, 3, 5, 0)
TRAIN_T = [1, *range(100, 1001, 100)]
FLOW_TRAIN_T = [0.001, *(i / 10 for i in range(1, 11))]
LOSS_T = [1, 2, 5, 10, 20, 35, 50, 75, 100, *range(150, 1001, 50)]
NUM_TRAJECTORIES = 4
SOURCE_NOISE = CONFIGS['fm-1to7']['source_noise']

def frame_indices(steps, max_frames = 11):
    if steps < max_frames:
        return set(range(steps + 1))
    return {round(i * steps / (max_frames - 1)) for i in range(max_frames)}

def clean(v):
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, float):
        return float(f'{v:.6g}') if math.isfinite(v) else None
    return v

class Exporter:
    def __init__(self, out, checkpoints, device, num_eval):
        self.out = Path(out)
        self.checkpoints = Path(checkpoints)
        self.device = device
        self.num_eval = num_eval

        g = torch.Generator().manual_seed(11)
        test, labels = mnist('test')
        first = lambda d, n = 1: (labels == d).nonzero()[:n, 0]

        self.demo = dict(
            x0 = test[torch.cat([first(d) for d in DEMO_DIGITS])],
            eps = torch.randn(4, 1, 28, 28, generator = g),
            sevens = test[first(7, 4)],
            ones = test[first(1, 4)] + SOURCE_NOISE * torch.randn(4, 1, 28, 28, generator = g),
        )

        g = torch.Generator().manual_seed(1234)
        self.start_noise = torch.randn(num_eval, 1, 28, 28, generator = g)
        self.start_ones = test[labels == 1][:num_eval] + SOURCE_NOISE * torch.randn(num_eval, 1, 28, 28, generator = g)

        g = torch.Generator().manual_seed(21)
        self.eval_x0 = test[:1000]
        self.eval_eps = torch.randn(1000, 1, 28, 28, generator = g)
        self.eval_sevens = test[labels == 7][:1000]
        self.eval_ones = test[labels == 1][:1000] + SOURCE_NOISE * torch.randn(1000, 1, 28, 28, generator = g)

        classifier = Classifier().to(device)
        classifier.load_state_dict(torch.load(self.checkpoints / 'classifier.pt', map_location = device))
        self.score_all = Scorer(classifier.eval(), test.to(device))
        self.score_sevens = Scorer(classifier.eval(), test[labels == 7].to(device))

        train_images, train_labels = mnist('train')
        self.mean_digit = train_images.mean(0)
        self.mean_seven = train_images[train_labels == 7].mean(0)

    # files

    def f16(self, name, tensor):
        path = self.out / name
        path.parent.mkdir(parents = True, exist_ok = True)
        tensor.float().cpu().numpy().astype(np.float16).tofile(path)
        return name

    def grid(self, name, images, n = 8, pad = 2):
        images = images[:n * n, 0].float().cpu().numpy()
        size = n * 28 + (n + 1) * pad
        canvas = np.full((size, size, 3), (26, 26, 25), np.uint8)

        for k, img in enumerate(images):
            r, c = divmod(k, n)
            gray = ((np.nan_to_num(img, nan = 0., posinf = 1., neginf = -1.).clip(-1, 1) + 1) * 127.5).astype(np.uint8)
            tile = np.repeat(gray[..., None], 3, axis = -1)
            tile[~np.isfinite(img)] = (170, 50, 50)
            y, x = pad + r * (28 + pad), pad + c * (28 + pad)
            canvas[y:y + 28, x:x + 28] = tile

        path = self.out / name
        path.parent.mkdir(parents = True, exist_ok = True)
        Image.fromarray(canvas).save(path, optimize = True)
        return name

    # models

    def load(self, name):
        ckpt = torch.load(self.checkpoints / f'{name}.pt', map_location = self.device)
        model = Unet(dim = ckpt['dim']).to(self.device)
        model.load_state_dict(ckpt['model'])
        return model.eval(), ckpt['history']

    @torch.no_grad()
    def forward(self, model, path, target, x0, eps, t):
        a, s = path.coef(t)
        x0, eps = x0.to(self.device), eps.to(self.device)
        xt = a * x0 + s * eps
        out = model(xt, torch.full((len(xt),), path.time_input(t), device = self.device))
        x0_hat, eps_hat = to_x0_eps(target, out, xt, a, s)
        return out, make_target(target, x0, eps, a, s), x0_hat, eps_hat

    def training_view(self, name, model, path, target, x0, eps, times):
        frames, losses = [], []
        for t in times:
            out, tgt, x0_hat, eps_hat = self.forward(model, path, target, x0, eps, t)
            frames.append(torch.stack((eps_hat, x0_hat), dim = 1))
            losses.append(((out - tgt) ** 2).mean(dim = (1, 2, 3)))
        frames = torch.stack(frames, dim = 1).reshape(4, len(times), 2, 784)
        return self.f16(f'train/{name}.f16', frames), torch.stack(losses, dim = 1).tolist()

    def error_by_time(self, model, path, target, x0, eps, times):
        rows = []
        for t in times:
            out, tgt, x0_hat, eps_hat = self.forward(model, path, target, x0, eps, t)
            mse = lambda a, b: ((a - b) ** 2).mean().item()
            rows.append([t, mse(out, tgt), mse(x0_hat, x0.to(self.device)), mse(eps_hat, eps.to(self.device))])
        return rows

    def run(self, stem, model, path, target, start, steps, method, clip = False, digit = None):
        keep, frames = frame_indices(steps), []

        def record(i, t, x, x0, eps):
            if i in keep:
                frames.append((i, t, torch.stack((x, eps, x0), dim = 1)[:NUM_TRAJECTORIES]))

        samples = sample(model, path, target, start.to(self.device), steps, method = method, clip = clip,
                         generator = torch.Generator().manual_seed(99), callback = record)

        trajectories = torch.stack([f[2] for f in frames], dim = 1).reshape(NUM_TRAJECTORIES, len(frames), 3, 784)
        scorer = self.score_sevens if digit == 7 else self.score_all

        return dict(
            file = self.f16(f'traj/{stem}.f16', trajectories),
            grid = self.grid(f'grid/{stem}.png', samples),
            frames_t = [f[1] for f in frames],
            frame_steps = [f[0] for f in frames],
            n = steps,
            **scorer(samples, digit = digit)
        )

def main():
    parser = argparse.ArgumentParser(description = 'export the website data')
    parser.add_argument('--out', default = 'docs/data')
    parser.add_argument('--checkpoints', default = 'checkpoints')
    parser.add_argument('--num-eval', type = int, default = 1000)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    ex = Exporter(args.out, args.checkpoints, device, args.num_eval)
    demo = ex.demo

    common = dict(
        T = 1000,
        demo = dict(digits = DEMO_DIGITS),
        train_t = TRAIN_T,
        fm_train_t = FLOW_TRAIN_T,
        loss_t = LOSS_T,
        schedules = {name: dict(a = get_path(name).a.tolist(), s = get_path(name).s.tolist()) for name in ('linear', 'cosine', 'sigmoid')},
        baseline = dict(
            mean_digit_mse = ((ex.eval_x0 - ex.mean_digit) ** 2).mean().item(),
            mean_seven_mse = ((ex.eval_sevens - ex.mean_seven) ** 2).mean().item(),
        ),
        n_eval = args.num_eval,
        n_traj = NUM_TRAJECTORIES,
        src_noise = SOURCE_NOISE,
    )

    ex.f16('base.f16', torch.stack((demo['x0'], demo['eps'], demo['sevens'], demo['ones'])).reshape(4, 4, 784))

    # part 1 - the DDPM family

    part1 = dict(models = {}, runs = {})

    for name in DIFFUSION:
        model, history = ex.load(name)
        config = CONFIGS[name]
        path, target = get_path(config['path']), config['target']

        train_file, train_loss = ex.training_view(name, model, path, target, demo['x0'], demo['eps'], TRAIN_T)
        part1['models'][name] = dict(
            train_file = train_file,
            train_loss = train_loss,
            loss_by_t = ex.error_by_time(model, path, target, ex.eval_x0, ex.eval_eps, LOSS_T),
            hist = history,
        )

        for key, (method, steps) in SAMPLERS.items():
            for clip in (False, True):
                stem = f'{name}_{key}_{"clip" if clip else "raw"}'
                run = ex.run(stem, model, path, target, ex.start_noise, steps, method, clip = clip)
                part1['runs'][f'{name}|{key}|{"clip" if clip else "raw"}'] = run
                print(f'{stem:<28} fid {run["fid"]}   digits {run["digit_rate"]:.3f}   nan {run["nan_frac"]:.2f}')

    # part 2 - flow matching

    part2 = dict(models = {}, runs = {})
    straight = get_path('straight')

    for name, x0, src, eval_x0, eval_src in (
        ('fm-gauss', demo['x0'], demo['eps'], ex.eval_x0, ex.eval_eps),
        ('fm-1to7', demo['sevens'], demo['ones'], ex.eval_sevens, ex.eval_ones)
    ):
        model, history = ex.load(name)
        train_file, train_loss = ex.training_view(name, model, straight, 'u', x0, src, FLOW_TRAIN_T)
        part2['models'][name] = dict(
            train_file = train_file,
            train_loss = train_loss,
            loss_by_t = ex.error_by_time(model, straight, 'u', eval_x0, eval_src, [t / 1000 for t in LOSS_T]),
            hist = history,
        )

    flows = dict(
        straight = (ex.load('fm-gauss')[0], straight, 'u'),
        curved = (ex.load('cosine-v')[0], get_path('cosine'), 'v'),
        ones = (ex.load('fm-1to7')[0], straight, 'u'),
    )

    for start, path_name in (('gauss', 'straight'), ('gauss', 'curved'), ('ones', 'straight')):
        model, path, target = flows['ones' if start == 'ones' else path_name]
        start_x = ex.start_ones if start == 'ones' else ex.start_noise

        for method in ('euler', 'stoch'):
            if start == 'ones' and method == 'stoch':
                continue

            for steps in FLOW_STEPS:
                stem = f'p2_{start}_{path_name}_{method}_{steps}'
                run = ex.run(stem, model, path, target, start_x, steps, 'euler' if method == 'euler' else 'ddpm',
                             digit = 7 if start == 'ones' else None)
                part2['runs'][f'{start}|{path_name}|{method}|{steps}'] = run
                print(f'{stem:<28} fid {run["fid"]}   digits {run["digit_rate"]:.3f}')

    manifest = clean(dict(common = common, p1 = part1, p2 = part2))
    (Path(args.out) / 'manifest.json').write_text(json.dumps(manifest, separators = (',', ':')))
    print('wrote', Path(args.out) / 'manifest.json')

if __name__ == '__main__':
    main()
