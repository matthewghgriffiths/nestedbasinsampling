# -*- coding: utf-8 -*-
import logging
from collections import defaultdict, deque

from scipy.special import psi, polygamma
import numpy as np

from pele.optimize import lbfgs_cpp

from ..utils import (
    Result, NestedSamplingError, SamplingError, dict_update_keep,
    RollingLinRegress)

logger = logging.getLogger('NBS.NestedOptimizer')

class NestedOptimizerKalman(object):
    """
    X : array
        the starting configuration for the minimization
    pot :
        the potential object
    sampler :
        the nested sampler object
    nsteps : int
        the maximum number of iterations
    tol : float
        the minimization will stop when the rms grad is less than tol
    iprint : int
        how often to print status information
    maxstep : float
        the maximum step size
    events : list of callables
        these are called after each iteration.  events can also be added using
        attachEvent()
    alternate_stop_criterion : callable
        this criterion will be used rather than rms gradiant to determine when
        to stop the iteration
    debug :
        print debugging information
    energy, gradient : float, float array
        The initial energy and gradient.  If these are both not None then the
        energy and gradient of the initial point will not be calculated, saving
        one potential call.
    """
    def __init__(self, X, pot, sampler, stepsize=0.1, target_acc=0.4,
                 MC_steps=20, tol=1e-1, nsave=20, nwait=20,
                 linreg_window=100, basin_optimization=False, basinE=None,
                 alternate_stop_criterion=None, max_tries=0, seed=None,
                 events=None, iprint=-1, nsteps=10000, debug=False,
                 target=None, use_quench=True, save_results=False,
                 energy=None, gradient=None, store_configs=False,
                 quench=lbfgs_cpp, quenchtol=1e-6, quench_kw=None):

        self.X = np.array(X)        # Copy initial coordinates
        self.Xi = self.X.copy()     # save initial coordinates
        self.Xlast = self.X.copy()  # save last coordinates
        self.dX = self.X.copy()     # save difference in coordinates

        self.pot = pot
        self.sampler = sampler
        self.stepsize = stepsize
        self.target_acc = target_acc
        self._p_factor = (1. - target_acc)/target_acc
        self.lin_reg = RollingLinRegress(linreg_window)

        self.nwait = nwait
        self.nsave = nsave
        self.save_results = save_results
        self.all_results = []
        self.last_results = deque(maxlen=nsave)
        assert nsave >= nwait

        # a list of events to run during the optimization
        self.events = [] if events is None else events
        self.iprint = iprint
        self.nsteps = nsteps
        self.MC_steps = MC_steps
        self.max_tries = max_tries
        self.tol = tol
        self.target = target

        self.store_configs = True

        self.basin_optimization = basin_optimization
        self.alternate_stop_criterion = alternate_stop_criterion
        self.working = True
        self.debug = debug  # print debug messages

        self.rms = np.inf
        self.Ediff = np.inf

        self.iter_number = 0
        self.n_tries = 0
        self.nopt = 0
        self.tot_rejects = 0
        self.tot_steps = 0
        self.naccept = defaultdict(int)

        self.nfev = 0
        if self.basin_optimization:
            if basinE is None:
                self.basinE = self.pot.getEnergy(self.Xi)
                self.nfev += 1
            else:
                self.basinE = basinE
            self.E = np.inf if energy is None else energy
            # The target energy must be higher than the basin energy.
            if self.target is not None:
                assert self.target > self.basinE
        else:
            if energy is None:
                self.E, self.G = self.pot.getEnergyGradient(self.X)
                self.nfev += 1
            else:
                self.E = energy
                self.G = gradient

        self.result = Result()
        self.result.naccept = 0
        self.result.nreject = 0
        self.result.initialcoords = self.Xi
        self.result.initialenergy = self.E
        self.result.message = []

        self.Emax = []
        self.stepsizes = []

        self.use_quench = use_quench
        self.quench = quench
        self.quenchtol = quenchtol
        self.quenchkw = {} if quench_kw is None else quench_kw
        dict_update_keep(self.quenchkw,
                         dict(events=self.events, iprint=self.iprint))

    def update_stepsize(self):
        if len(self.last_results) > self.nwait:
            res = self.last_results[-self.nwait-1]
            naccept = res['naccept']
            nreject = res['nreject']
            logit_p = psi(naccept) - psi(nreject)
            self.lin_reg.update(res['energy'],
                                logit_p + np.log(res['stepsize']))
            y = self.lin_reg.predict(self.E)
            self.stepsize = np.exp(y)*self._p_factor

    def one_iteration(self):
        try:
            X = self.Xi if self.basin_optimization else self.X
            res = self.sampler(
                self.E, X, stepsize=self.stepsize, nsteps=self.MC_steps)

            if res.energy < self.E:
                self.last_results.append(res)
                if self.save_results:
                    self.all_results.append(res)

                self.Xlast[:] = self.X
                self.X[:] = res.coords

                self.E = res.energy
                self.G = (res.grad if hasattr(res, 'grad')
                          else self.pot.getGradient(self.X))
                self.rms = np.linalg.norm(self.G)/np.sqrt(self.X.size)
                if len(self.last_results) == self.nsave:
                    self.Ediff = (self.last_results[0].energy -
                                  self.last_results[-1].energy)
                else:
                    self.Ediff = np.inf

                self.curr_accept = res.naccept
                self.curr_reject = res.nreject
                self.Emax.append(res.energy)
                self.stepsizes.append(res.stepsize)
                self.nfev += res.nfev

                self.printState(False)

                for event in self.events:
                    event(coords=self.X, energy=self.E, res=res,
                          rms=self.rms, stepsize=self.stepsize)

                self.update_stepsize()
                self.n_tries = 0
                self.iter_number += 1
            elif res.naccept == 0:
                logger.warning((
                    "no steps were accepted, E={:10.5}, stepsize={:10.5g}, "
                    "halving stepsize").format(self.E, self.stepsize))
                self.stepsize /= 2
                self.n_tries += 1
                if self.n_tries >= self.max_tries:
                    raise SamplineError()
            else:
                raise NestedSamplingError(
                    "Energy of new replica greater than cutoff")

        except SamplingError as e:
            if self.n_tries < self.max_tries:
                logger.warning((
                    "Sampling error, E={:10.5}, stepsize={:10.5g}, niter={:d}"
                    " halving stepsize").format(
                        self.E, self.stepsize, self.iter_number))
                self.n_tries += 1
                self.stepsize /= 2
            else:
                self.result.message.append('Sampling Error')
                if self.use_quench:
                    quenchres = self.quench_config()
                    self.X = quenchres.coords
                    self.E = quenchres.energy
                    self.G = quenchres.grad
                    self.rms = quenchres.rms
                    self.nopt = self.iter_number
                    self.iter_number += quenchres.nsteps
                self.working = False

    def printState(self, force=True):
        """logs the current state"""
        cond = (self.iprint > 0 and
                self.iter_number % self.iprint == 0) or force
        if cond:
            logger.debug((
                "niters={:4d}, Ecut={:10.5g}, rms={:10.5g}, "
                "stepsize={:8.4g}, naccept={:4d}, nreject={:4d}").format(
                    self.iter_number, self.E, self.rms, self.stepsize,
                    self.curr_accept, self.curr_reject))

    def stop_criterion(self):
        """test the stop criterion"""
        if self.working:
            if self.alternate_stop_criterion is None:
                if self.target is not None:
                    return (self.E < self.target)
                else:
                    return (
                        self.rms < self.tol or self.Ediff < self.tol)
            else:
                return self.alternate_stop_criterion(
                    energy=self.E, gradient=self.G,
                    tol=self.tol, coords=self.X)
        else:
            return True

    def run(self):
        logger.info(
            ("Starting NOpt,   E={:10.5g}, "
             "stepsize={:6.3g}").format(self.E, self.stepsize))
        while self.iter_number < self.nsteps and not self.stop_criterion():
            try:
                self.one_iteration()
            except NestedSamplingError:
                self.result.message.append("problem with nested sampler")
                break

        self.nopt = self.iter_number
        if self.use_quench:
            logger.info(
                "Starting quench, E={:10.5g}, niter   = {:5d}".format(
                    self.E, self.iter_number))
            quenchres = self.quench_config()
            self.X = quenchres.coords
            self.E = quenchres.energy
            self.G = quenchres.grad
            self.rms = quenchres.rms
            self.iter_number += quenchres.nsteps
            self.nfev += quenchres.nfev

        res = self.get_result()

        logger.info(
            ("Final config,    E={:10.5g}, nfev ={:9d}, "
             "rms ={:10.5g}").format(res.energy, res.nfev, res.rms))

        return self.get_result()

    def quench_config(self):
        return self.quench(self.X, self.pot,
                           maxstep=np.linalg.norm(self.X-self.Xlast),
                           energy=self.E, gradient=self.G,
                           tol=self.quenchtol, **self.quenchkw)

    def get_result(self):
        res = self.result
        res.coords = self.X
        res.energy = self.E
        res.rms = self.rms
        res.grad = self.G
        res.Emax = np.array(self.Emax)
        res.stepsize = np.array(self.stepsizes)
        res.nsteps = self.nopt
        res.success = self.stop_criterion()
        res.nfev = self.nfev
        if self.save_results:
            res.results = self.all_results
        return res


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from pele.potentials import LJ
    from pele.optimize import lbfgs_cpp

    from nestedbasinsampling.structure.constraints import HardShellConstraint
    from nestedbasinsampling.sampling.takestep import random_structure
    from nestedbasinsampling.sampling.noguts import NoGUTSSampler
    from nestedbasinsampling.optimize.utils import RecordMinimization

    pot = LJ()

    natoms = 75
    radius =  float(natoms) ** (1. / 3)
    rand_config = lambda : random_structure(natoms, radius)
    constraint = HardShellConstraint(radius)

    coords = rand_config()
    Ecut = pot.getEnergy(coords)
    sampler = NoGUTSSampler(
        pot, constraint=constraint,
        max_depth=7, linear_momentum=True, angular_momentum=True)
    s = determine_stepsize(
        coords, pot.getEnergy(coords), sampler, 0.3, nadapt=5, )
    nopt = RecordMinimization(
        pot, NestedOptimizerKalman, nsteps=2000,
        MC_steps=5, target_acc=0.35, nsave=10, tol=1e-2,
        sampler=sampler, stepsize=s[-1], iprint=10, debug=True)

    #prun res = nopt(coords).run()
    #res = nopt(coords).run()
    #r = nopt.store['res'][100]
    #x = r.coords
    #d = r.stepsize

    #nopt(x, stepsize=d).run()

    raise Exception

    nopt = RecordMinimization(pot, NestedOptimizer, nsteps=15, nadapt=8,
        sampler=sampler, stepsize=None, iprint=100, debug=True)
    E3s = [nopt(x).run().energy for i in xrange(10)]


    nopt = RecordMinimization(
        pot, NestedOptimizer,
        sampler=sampler, stepsize=None, nsteps=10, iprint=100, debug=True)
    E10s = [nopt(x).run().energy for i in xrange(10)]

    sampler = GalileanSampler(
        pot, stepsize=0.5, constraint=constraint,
        nsteps=15, nadapt=15, acc_ratio=0.25)
    nopt = RecordMinimization(
        pot, NestedOptimizer,
        sampler=sampler, stepsize=None, nsteps=10, iprint=100, debug=True)
    E15s = [nopt(x).run().energy for i in xrange(10)]

    sampler = GalileanSampler(
        pot, stepsize=0.5, constraint=constraint,
        nsteps=25, nadapt=25, acc_ratio=0.25)
    nopt = RecordMinimization(
        pot, NestedOptimizer,
        sampler=sampler, stepsize=None, nsteps=10000, iprint=100, debug=True)
    E25s = [nopt(x).run().energy for i in xrange(10)]

    res = nopt(x).run()

    print res.accept
    print res.accept['nreject'] * 1./ res.accept['niter'], res.accept['nreflect']*1./res.accept['niter']

    from nestedbasinsampling.sampling.stats import CDF, AndersonDarling
    CDF(E3s).plot(); CDF(E10s).plot(); CDF(E15s).plot(); CDF(E25s).plot()
    AndersonDarling.compareDistributions([E10s, E15s, E25s])

    niter = []
    nreject = []
    nreflect = []
    for r in nopt.store['res']:
        niter.append(sum(
            accept['niter'] for accept in r.accepts))
        nreject.append(sum(
            accept['nreject'] for accept in r.accepts))
        nreflect.append(sum(
            accept['nreflect'] for accept in r.accepts))

    plt.plot(nreject)
    plt.plot(nreflect)
    plt.plot(niter)
    plt.plot(np.clip(res.Emax - res.Emax.min(), 0, 200))
