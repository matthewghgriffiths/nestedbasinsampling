

import numpy as np


def calc_n_effective(Es, mean, var, crit=1e-2):
    """
    """
    Es = np.asanyarray(Es)
    M = len(Es)
    corr = np.fft.ifftshift(np.correlate(Es-mean, Es-mean, mode='same'))
    corr /= var * (M - np.arange(M))

    for Mcut in xrange(M-1):
        if corr[Mcut+1] < crit:
            break

    return 1/(1 + 2 * ((1 - np.arange(1, Mcut)/M) * corr[1:Mcut]).sum())
