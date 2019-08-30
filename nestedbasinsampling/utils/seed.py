import numpy as np
import random
try:
    from ..sampling.fortran.noguts import noguts
    has_fortran = True
except ImportError:
    noguts = None
    has_fortran = False


def set_seed(seed=None):
    np.random.seed(seed)
    random.seed(seed)
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    if has_fortran:
        noguts.seed = seed
    np.random.seed(seed)
