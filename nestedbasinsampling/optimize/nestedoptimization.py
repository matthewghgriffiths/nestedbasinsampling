# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
from pele.optimize import lbfgs_cpp


from nestedbasinsampling.sampling import SamplingError
from nestedbasinsampling.utils import (Result, NestedSamplingError,
                                       dict_update_keep)

class NestedOptimizer(object):

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
    sqstr= "NOPT > Starting quench,E ={:10.4g}, niter  = {:9d}"
    qstr = "NPOT > Quench config,  E ={:10.4g}, nsteps = {:9d}, rms  ={:10.5g}"

    def __init__(self, X, pot, sampler,
                 tol=1e-1, alternate_stop_criterion=None, stepsize=None,
                 events=None, iprint=-1, nsteps=10000, debug=False,
                 MC_steps=None, nadapt=None, target=None, use_quench=True,
                 energy=None, gradient=None, store_configs=False,
                 quench=lbfgs_cpp, quenchtol=1e-6, quench_kw={}):

        self.X = np.array(X)        # Copy initial coordinates
        self.Xi = self.X.copy()     # save initial coordinates
        self.Xlast = self.X.copy()  # save last coordinates
        self.dX = self.X.copy()     # save difference in coordinates

        self.pot = pot
        self.sampler = sampler
        self.stepsize = stepsize

        # a list of events to run during the optimization
        self.events = [] if events is None else events
        self.iprint = iprint
        self.nsteps = nsteps
        self.nadapt = nadapt
        self.MC_steps = MC_steps
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

        self.iter_number = 0
        self.nopt = 0
        self.tot_rejects = 0
        self.tot_steps = 0
        self.accept = defaultdict(int)
        self.result = Result()
        self.result.initialcoords = self.Xi
        self.result.message = []

        self.Emax = []
        self.stepsizes = []

        self.use_quench = use_quench
        self.quench = quench
        self.quenchtol = quenchtol
        self.quenchkw = quench_kw
        dict_update_keep(self.quenchkw,
                         dict(events=self.events, iprint=self.iprint))

    def one_iteration(self):
        try:
            res = self.sampler(self.E, self.X, nsteps=self.MC_steps,
                               nadapt=self.nadapt, stepsize=self.stepsize)

            assert res.energy < self.E

            self.stepsize = res.stepsize

            for accept in res.accepts:
                for k, n in accept.iteritems():
                    self.accept[k] += n

            self.Xlast[:] = self.X
            self.X[:] = res.coords

            self.E = res.energy
            self.G = (res.grad if hasattr(res, 'grad')
                      else self.pot.getGradient(self.X))
            self.rms = np.linalg.norm(self.G)/np.sqrt(self.X.size)

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
            print self.pstr.format(
                self.iter_number, self.E, self.rms, self.stepsize)

    def stop_criterion(self):
        """test the stop criterion"""

        if self.alternate_stop_criterion is None:
            if self.target is not None:
                return (self.E < self.target) or (self.rms < self.tol)
            else:
                return self.rms < self.tol
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
        res.accept = self.accept
        res.coords = self.X
        res.energy = self.E
        res.rms = self.rms
        res.grad = self.G
        res.Emax = np.array(self.Emax)
        res.stepsize = np.array(self.stepsizes)
        res.nsteps = self.nopt
        res.success = self.stop_criterion()
        return res

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    from pele.potentials import LJ
    from pele.optimize import lbfgs_cpp

    from nestedbasinsampling.structure.constraints import HardShellConstraint
    from nestedbasinsampling.sampling.takestep import random_structure
    from nestedbasinsampling.sampling.galilean import GalileanSampler
    from nestedbasinsampling.optimize.utils import RecordMinimization


    natoms = 31
    pot = LJ()
    radius =  float(natoms) ** (1. / 3)
    rand_config = lambda : random_structure(natoms, radius)
    constraint = HardShellConstraint(radius)

    sampler = GalileanSampler(
        pot, stepsize=0.5, constraint=constraint,
        nsteps=15, nadapt=15, acc_ratio=0.25)

    coords = rand_config()

    nopt = RecordMinimization(
        pot, NestedOptimizer, maxsteps=2000, nsteps=15, nadapt=15,
        sampler=sampler, stepsize=None, iprint=100, debug=True)

    #prun res = nopt.run()
    res = nopt(coords).run()
    x = nopt.store['res'][100]['coords']

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

