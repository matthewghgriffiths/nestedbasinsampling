
import logging
from nestedbasinsampling import (
    NBS_manager, LOG_CONFIG, SamplingError, combineAllRuns, Replica)
from gaussian_system import get_system, system_kws, Ecut

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.ion()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, **LOG_CONFIG)

    system = get_system()
    db = system.get_database('gaussian.sqlite')
    runs = db.runs()
    run = combineAllRuns(runs).split(
        r1=None, r2=Replica(max(r.Emax[-1] for r in runs), None))


    k = 31*3-6
    plt.ion()
    plt.figure()
    Es = run.Emax
    f = 0.5 * k * (np.log(Es) - np.log(1000.))
    plt.plot(run.Emax**(2./k), run.log_frac)
    plt.plot(run.Emax**(2./k), f)
    plt.show()
