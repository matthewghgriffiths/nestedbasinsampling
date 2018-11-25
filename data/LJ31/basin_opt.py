
import logging
import numpy as np
from matplotlib import pyplot as plt

from nestedbasinsampling import NBS_LJ, NestedOptimizerKalman, LOG_CONFIG

logging.basicConfig(level=logging.DEBUG, **LOG_CONFIG)

natoms = 31
radius = 2.5
nopt_kws = dict(
    nsteps=2000, MC_steps=5, target_acc=0.4, nsave=20, tol=1e-2,
    iprint=10, nwait=10, debug=True)
x = np.loadtxt("min-133.59.txt")
x -= x.mean(0)
system = NBS_LJ(natoms, radius, nopt_kws=nopt_kws)

for i in range(10):
    res = system.nopt(
        x.flatten(), Ecut=np.inf, target=-133., max_tries=5,
        use_quench=False, basin_optimization=True, nsave=4, nwait=2,
        quench_kw=dict(iprint=-1))
    Emax = res['Emax']
    stepsizes = res['stepsize']
    i = Emax.size - Emax[::-1].searchsorted(0)
    plt.plot(Emax[i:])
