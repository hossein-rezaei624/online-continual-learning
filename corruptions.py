import numpy as np


def gaussian_noise_deterministic(x, severity=1, seed=None):
    #if seed is not None:
     #   np.random.seed(seed)

    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
