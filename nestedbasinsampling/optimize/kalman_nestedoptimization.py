# -*- coding: utf-8 -*-

from collections import defaultdict, deque

import numpy as np
from scipy.special import psi, polygamma
from pele.optimize import lbfgs_cpp

from nestedbasinsampling.sampling import SamplingError
from nestedbasinsampling.utils import (Result, NestedSamplingError,
                                       dict_update_keep, LinearKalmanFilter)

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
    pstr = ("NOPT > niters  ={:4d}, Ecut   ={:10.5g}, "
            "rms  ={:10.5g}, stepsize = {:8.4g}")
    sqstr= ("NOPT > Starting quench,      E ={:10.4g}, "
            "niter  = {:9d}")
    qstr = ("NPOT > Quench config,        E ={:10.4g}, nsteps = {:9d}, "
            "rms  ={:10.5g}"


    def __init__(self, X, pot, sampler, stepsize=0.1, target_acc=0.4,
                 MC_steps=20, tol=1e-1, nsave=10, nwait=1,
                 kalman_discount=10., kalman_var=1.,
                 alternate_stop_criterion=None,
                 events=None, iprint=-1, nsteps=10000, debug=False,
                 target=None, use_quench=True,
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
        self.kalman = LinearKalmanFilter(
            z=np.log(stepsize), R=kalman_var)
        self.kalman_discount = kalman_discount
        self.nwait = nwait
        self.nsave = nsave
        self.last_results = deque(maxlen=nsave)
        assert nsave >= nwait

        # a list of events to run during the optimization
        self.events = [] if events is None else events
        self.iprint = iprint
        self.nsteps = nsteps
        self.MC_steps = MC_steps
        self.nsave = nsave
        self.tol = tol
        self.target = target

        self.store_configs = True

        self.alternate_stop_criterion = alternate_stop_criterion
        self.debug = debug  # print debug messages

        if energy is None and gradient is None:
            self.E, self.G = self.pot.getEnergyGradient(self.X)
        elif gradient is None:
            self.E = energy
            self.G = self.pot.getGradient(self.X)
        else:
            self.E = energy
            self.G = gradient

        self.rms = np.inf
        self.Ediff = np.inf

        self.iter_number = 0
        self.nopt = 0
        self.tot_rejects = 0
        self.tot_steps = 0
        self.naccept = defaultdict(int)
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
            naccept = res['naccept']/self.kalman_discount + 0.5
            nreject = res['nreject']/self.kalman_discount + 0.5
            m = psi(naccept) - psi(nreject)
            v = polygamma(1, naccept) + polygamma(1, nreject)
            X1, P1 = self.kalman(m + np.log(res.stepsize), v)
            self.kalman.P = np.clip(self.kalman.P, -0.7, 0.7)
            #X, P = self.kalman.forecast(self.nwait)
            a = X1
            self.stepsize = np.exp(a)*self._p_factor

    def one_iteration(self):
        try:
            res = self.sampler(
                self.E, self.X, stepsize=self.stepsize, nsteps=self.MC_steps)

            assert res.energy < self.E
            self.last_results.append(res)

            self.update_stepsize()

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

            self.printState(False)

            for event in self.events:
                event(coords=self.X, energy=self.E, res=res,
                      rms=self.rms, stepsize=self.stepsize)

            self.iter_number += 1

        except SamplingError:
            if self.debug:
                print "NOPT > Sampling error"
            self.result.message.append('Sampling Error')

            if self.use_quench:
                quenchres = self.quench_config()
                self.X = quenchres.coords
                self.E = quenchres.energy
                self.G = quenchres.grad
                self.rms = quenchres.rms
                self.nopt = self.iter_number
                self.iter_number += quenchres.nsteps


    def printState(self, force=True):
        cond = (self.iprint > 0 and self.iter_number%self.iprint == 0) or force
        if cond:
            print (
                "NOPT > niters  ={:4d}, Ecut   ={:10.5g},  rms  ={:10.5g}, "
                "stepsize = {:8.4g}, naccept={:4d}, nreject={:4d}").format(
                    self.iter_number, self.E, self.rms, self.stepsize,
                    self.curr_accept, self.curr_reject)

    def stop_criterion(self):
        """test the stop criterion"""

        if self.alternate_stop_criterion is None:
            if self.target is not None:
                return ((self.E < self.target) or
                        (self.rms < self.tol) or
                        (self.Ediff < self.tol))
            else:
                return (self.rms < self.tol or self.Ediff < self.tol)
        else:
            return self.alternate_stop_criterion(energy=self.E,
                                                 gradient=self.G,
                                                 tol=self.tol, coords=self.X)

    def run(self):
        while self.iter_number < self.nsteps and not self.stop_criterion():
            try:
                self.one_iteration()
            except NestedSamplingError:
                self.result.message.append("problem with nested sampler")
                break

        if self.iprint > 0:
            print self.sqstr.format(self.E, self.iter_number)

        self.nopt = self.iter_number
        if self.use_quench:
            quenchres = self.quench_config()

            if self.iprint > 0:
                print self.qstr.format(
                    quenchres.energy,quenchres.nsteps,quenchres.rms)

            self.X = quenchres.coords
            self.E = quenchres.energy
            self.G = quenchres.grad
            self.rms = quenchres.rms
            self.iter_number += quenchres.nsteps

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
        return res

def determine_stepsize(coords, Ecut, sampler, stepsize, target=0.4,
                       nsteps=200, nadapt=30):

    f = (1 - target)/target
    kalman = LinearKalmanFilter(z=np.log(stepsize) - np.log(f), R=1.)
    stepsizes = [stepsize]
    for i in xrange(nadapt):
        p = sampler.genstep(coords)
        last = stepsize
        Es, Gs, path, ps, accepts = sampler.integrate(
            coords, p, Ecut, epsilon=stepsize, nsteps=nsteps)
        naccept = accepts.sum()
        if naccept:
            coords = path[accepts][-1]
        nreject = nsteps - naccept
        acc = float(naccept)/nsteps
        n, m = naccept/10. + 0.5, nreject/10. + 0.5
        m = psi(n) - psi(m)
        v = polygamma(1, n) + polygamma(1, m)
        a0 = m + np.log(stepsize)
        a1, v1 = kalman(a0, v)
        stepsize = np.exp(a1)*f
        stepsizes.append(stepsize)

        print "{:6.3g}+/-{:6.2g}, {:6.3g}+/-{:6.2g}, {:3d}, {:3d}, {:6.2g}, {:6.2g}".format(
            a0, v**0.5, a1, v1**0.5, naccept, nreject, last, stepsize)
    return stepsizes

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
