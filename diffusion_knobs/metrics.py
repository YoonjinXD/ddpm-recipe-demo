import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def exists(v):
    return v is not None

class Classifier(nn.Module):
    """a small MNIST CNN, used only to score samples"""

    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25), nn.Flatten(),
            nn.Linear(9216, 128), nn.ReLU()
        )
        self.to_logits = nn.Sequential(nn.Dropout(0.5), nn.Linear(128, 10))

    def features(self, x):
        return self.convs(x)

    def forward(self, x):
        return self.to_logits(self.features(x))

def train_classifier(images, labels, epochs = 3, batch_size = 128, lr = 1e-3, log = print):
    model = Classifier().to(images.device)
    opt = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):
        for idx in torch.randperm(len(images), device = images.device).split(batch_size):
            loss = F.cross_entropy(model(images[idx]), labels[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        log(f'epoch {epoch + 1}   loss {loss.item():.4f}')

    return model.eval()

def frechet_distance(feats1, feats2):
    mu1, mu2 = feats1.mean(0), feats2.mean(0)
    cov1, cov2 = np.cov(feats1, rowvar = False), np.cov(feats2, rowvar = False)

    w, v = np.linalg.eigh(cov1)
    sqrt_cov1 = (v * np.sqrt(w.clip(min = 0))) @ v.T
    eig = np.linalg.eigvalsh(sqrt_cov1 @ cov2 @ sqrt_cov1)

    return float(((mu1 - mu2) ** 2).sum() + np.trace(cov1) + np.trace(cov2) - 2 * np.sqrt(eig.clip(min = 0)).sum())

class Scorer:
    """FID-style distance in the classifier's feature space, and how many samples it confidently calls a digit"""

    def __init__(self, classifier, reference_images, confidence = 0.9):
        self.classifier = classifier
        self.confidence = confidence
        self.reference, _ = self.embed(reference_images)

    @torch.no_grad()
    def embed(self, images, batch_size = 1000):
        feats, probs = [], []
        for chunk in images.split(batch_size):
            f = self.classifier.features(chunk)
            feats.append(f.cpu().double())
            probs.append(self.classifier.to_logits(f).softmax(dim = -1).cpu())
        return torch.cat(feats).numpy(), torch.cat(probs).numpy()

    def __call__(self, samples, digit = None):
        finite = torch.isfinite(samples).flatten(1).all(dim = 1).cpu().numpy()
        samples = samples.nan_to_num(0., 1., -1.).clamp(-1., 1.)

        feats, probs = self.embed(samples)
        confident = finite & (probs.max(1) >= self.confidence)
        labels = probs.argmax(1)

        out = dict(
            fid = frechet_distance(feats, self.reference) if finite.all() else None,
            digit_rate = float(confident.mean()),
            nan_frac = float(1 - finite.mean()),
            classes = np.bincount(labels[confident], minlength = 10).tolist(),
        )

        if exists(digit):
            out['want_rate'] = float((confident & (labels == digit)).mean())

        return out
