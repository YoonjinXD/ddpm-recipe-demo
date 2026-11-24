# every model on the website is one of these: same network, same data, same budget

DIFFUSION = {
    f'{path}-{target}': dict(path = path, target = target)
    for path in ('linear', 'cosine', 'sigmoid')
    for target in ('eps', 'x0', 'v')
}

FLOW = {
    'fm-gauss': dict(path = 'straight', target = 'u'),
    'fm-1to7': dict(path = 'straight', target = 'u', data_digit = 7, source_digit = 1, source_noise = 0.05),
}

CONFIGS = {**DIFFUSION, **FLOW}
