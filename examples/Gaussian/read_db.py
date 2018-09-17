
import logging
from nestedbasinsampling import (
    NBS_manager, LOG_CONFIG, SamplingError, combineAllRuns, Replica)
from gaussian_system import get_system, system_kws, Ecut

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.ion()

if 1:
    logging.basicConfig(level=logging.INFO, **LOG_CONFIG)

    system = get_system()
    pot = system.pot
    nuts = system.sampler
    random_coords = system.get_random_configuration

    k = 87
    epsilon = 0.02
    nsamples = 1000
    a = np.arange(nsamples) + 1
    b = nsamples + 1 - a
    l = np.log(a) - np.log(a+b)
    l2 = l + np.log(a+1) - np.log(a+b+1)
    lstd = np.log1p(np.sqrt(np.exp(l2 - 2 * l) - 1))
    Es = np.array([pot.getEnergy(random_coords()) for i in range(nsamples)])

    coords = random_coords()
    results = []
    for i in tqdm(xrange(nsamples)):
        rs = []
        pos = coords
        while len(rs) < 10:
            r = nuts.nuts_step(pos, Ecut, epsilon=0.02)
            if r.naccept:
                rs.append(r)
                pos = rs[-1].coords
        results.append(rs)

    nEs = np.array([[r.energy for r in rs] for rs in results])
    nEs.sort(0)
    for Es in nEs.T[5:]:
        plt.plot(Es**(0.5*k), ((l - 0.5*k*np.log(Es))/lstd))


if 0:


    db = system.get_database('gaussian3.sqlite')
    runs = db.runs()
    run = combineAllRuns(runs).split(
        r1=None, r2=Replica(max(r.Emax[-1] for r in runs), None))




    k = 31*3-6
    plt.ion()
    plt.plot(np.log(run.Emax), run.log_frac)


    Es = run.Emax
    f = 2 * k * (np.log(Es) - np.log(Ecut))
    plt.plot(run.Emax**(0.5/k), run.log_frac)
    plt.plot(run.Emax**(0.5/k), f)
