# DDPM Recipe Demo

An interactive look at the three recipes inside a diffusion model, using MNIST digits drawn as 3D bars:

1. **Target**: what the network predicts (ε, x₀ or v)
2. **Schedule**: how much noise at each step (linear, cosine, sigmoid)
3. **Sampler**: how to generate a sample from those predictions (DDPM, DDIM)

Part 2 shows how flow matching fits into the same three recipes.

**Website:** https://yoonjinxd.github.io/ddpm-recipe-demo/ (or serve `docs/` locally, see below).

## Code

```
diffusion_knobs/
    paths.py       noise schedules, and the straight path of flow matching
    targets.py     what the network predicts, and how every target turns back into (x0, eps)
    unet.py        a small UNet for 28 x 28 images
    trainer.py     the training loop (algorithm 1 of DDPM)
    sampler.py     one sampler for DDPM, DDIM and Euler
    configs.py     the 11 models shown on the website
    metrics.py     a small MNIST classifier that scores samples
    data.py        MNIST, scaled to [-1, 1]
train.py           train models
export.py          run every setting and write docs/data/ for the website
docs/              the website (served by GitHub Pages)
```

The training step in `trainer.py` is a handful of lines; switching from ε- to v-prediction only changes the target it is compared against (`targets.py`).

## Reproduce

```bash
pip install -r requirements.txt
python train.py all                         # 11 models + the classifier (a GPU helps)
python export.py                            # writes docs/data/
python -m http.server 8000 --directory docs # then open http://localhost:8000
```

`python train.py linear-eps cosine-v` trains just those two. Names follow `<schedule>-<target>`, plus `fm-gauss` and `fm-1to7` for flow matching.

## References

Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models* (2020) · Song, Meng & Ermon, *Denoising Diffusion Implicit Models* (2021) ·
Nichol & Dhariwal, *Improved DDPM* (2021) · Salimans & Ho, *Progressive Distillation* (2022) · Lin et al., *Common Diffusion Noise Schedules and Sample Steps Are Flawed* (2024) ·
Lipman et al., *Flow Matching for Generative Modeling* (2023) · Liu, Gong & Liu, *Rectified Flow* (2023). The full list is at the bottom of the website.
