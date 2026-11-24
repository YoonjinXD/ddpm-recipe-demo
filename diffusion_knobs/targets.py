# what the network is asked to predict, given x_t = a * x0 + s * eps

def make_target(kind, x0, eps, a, s):
    return dict(
        eps = eps,                 # DDPM
        x0 = x0,
        v = a * eps - s * x0,      # Salimans & Ho, 2022
        u = eps - x0,              # flow matching velocity on the straight path
    )[kind]

def to_x0_eps(kind, out, xt, a, s):
    # any prediction can be turned back into (x0, eps) - the divisions are where things break

    if kind == 'eps':
        return (xt - s * out) / a, out

    if kind == 'x0':
        return out, (xt - a * out) / s

    if kind == 'v':
        return a * xt - s * out, s * xt + a * out

    if kind == 'u':
        return xt - s * out, xt + a * out

    raise ValueError(f'unknown target {kind}')
