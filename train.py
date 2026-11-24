import argparse
from pathlib import Path

import torch

from diffusion_knobs import CONFIGS, Unet, get_path, mnist, train
from diffusion_knobs.metrics import train_classifier

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def main():
    parser = argparse.ArgumentParser(description = 'train the models behind the website')
    parser.add_argument('names', nargs = '+', help = f'"all", "classifier", or any of: {", ".join(CONFIGS)}')
    parser.add_argument('--steps', type = int, default = 20_000)
    parser.add_argument('--dim', type = int, default = 64)
    parser.add_argument('--checkpoints', default = 'checkpoints')
    args = parser.parse_args()

    device = get_device()
    folder = Path(args.checkpoints)
    folder.mkdir(parents = True, exist_ok = True)

    names = [*CONFIGS, 'classifier'] if args.names == ['all'] else args.names

    for name in names:
        torch.manual_seed(0)

        if name == 'classifier':
            images, labels = mnist('train')
            classifier = train_classifier(images.to(device), labels.to(device))
            torch.save(classifier.state_dict(), folder / 'classifier.pt')
            continue

        config = CONFIGS[name]
        images, _ = mnist('train', digit = config.get('data_digit'))
        sources = mnist('train', digit = config['source_digit'])[0].to(device) if 'source_digit' in config else None

        print(f'\n{name}: {len(images)} images, predicting {config["target"]} on the {config["path"]} path')

        ema, history = train(
            Unet(dim = args.dim).to(device),
            get_path(config['path']),
            config['target'],
            images.to(device),
            sources = sources,
            source_noise = config.get('source_noise', 0.),
            steps = args.steps
        )

        torch.save(dict(model = ema.state_dict(), dim = args.dim, steps = args.steps, history = history), folder / f'{name}.pt')

if __name__ == '__main__':
    main()
