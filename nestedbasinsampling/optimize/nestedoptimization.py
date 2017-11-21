# -*- coding: utf-8 -*-

import numpy as np
from pele.optimize import lbfgs_cpp

from nestedbasinsampling.sampling import GMCSampler, SamplingError
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

    pstr = "NOPT > niters  ={:4d}, F = {:10.4g}, Ecut   ={:10.5g}, rms  ={:10.5g}"
    sqstr= "NOPT > Starting quench,E ={:10.4g}, niter  = {:9d}"
    qstr = "NPOT > Quench config,  E ={:10.4g}, nsteps = {:9d}, rms  ={:10.5g}"

    def __init__(self, X, pot, sampler,
                 tol=1e-1, alternate_stop_criterion=None,
                 events=None, iprint=-1, nsteps=10000, debug=False,
                 energy=None, gradient=None, store_configs=False,
                 quench=lbfgs_cpp, quenchtol=1e-6, quench_kw={}):

        self.X = np.array(X)        # Copy initial coordinates
        self.Xi = self.X.copy()     # save initial coordinates
        self.Xlast = self.X.copy()  # save last coordinates
        self.dX = self.X.copy()     # save difference in coordinates

        self.pot = pot
        self.sampler = sampler

        # a list of events to run during the optimization
        self.events = [] if events is None else events
        self.iprint = iprint
        self.nsteps = nsteps
        self.tol = tol

        self.store_configs = True

        self.alternate_stop_criterion = alternate_stop_criterion
        self.debug = debug  # print debug messages

        # To store information from the first iteration
        self.initial = True
        self.ireject = 0
        self.isteps = 0

        if energy is None and gradient is None:
            self.E, self.G = self.pot.getEnergyGradient(self.X)
        elif energy is not None:
            self.E = energy
            self.G = self.pot.getGradient(self.X)
        else:
            self.E = energy
            self.G = gradient

        self.rms = np.linalg.norm(self.G) / np.sqrt(self.G.size)
        self.estrms = self.rms

        self.iter_number = 0
        self.f = 1.
        self.tot_rejects = 0
        self.tot_steps = 0
        self.result = Result()
        self.result.initialcoords = self.Xi
        self.result.message = []

        self.Emax = []
        if self.store_configs:
            self.configs = []

        self.quench = quench
        self.quenchtol = quenchtol
        self.quenchkw = quench_kw
        dict_update_keep(self.quenchkw,
                         dict(events=self.events, iprint=self.iprint))

    def one_iteration(self):
        try:
            Xnew, Enew, nsteps, nreject = self.sampler.new_point(self.E, self.X)

            self.f /= 2

            if Enew > self.E:
                ## something went wrong
                raise NestedSamplingError(
                    self.E, Enew, ", nsteps {}, nreject {}".format(nsteps,nreject))

            self.tot_rejects += nreject
            self.tot_steps += nsteps

            self.Xlast[:] = self.X
            self.X[:] = Xnew

            self.E = Enew
            self.G = self.pot.getGradient(self.X)
            self.rms = np.linalg.norm(self.G)/np.sqrt(self.X.size)

            self.Emax.append(Enew)
            if self.store_configs:
                self.configs.append(self.X.copy())

            self.printState(False)

            for event in self.events:
                event(coords=self.X, energy=self.E, rms=self.rms)

            if self.initial:
                self.isteps = nsteps
                self.ireject = nreject
                self.initial = False

            self.iter_number += 1

        except SamplingError:
            if self.debug:
                print "NOPT > Sampling error"
            self.result.message.append('Sampling Error')
            quenchres = self.quench_config()

            self.X = quenchres.coords
            self.E = quenchres.energy
            self.G = quenchres.grad
            self.rms = quenchres.rms
            self.estrms = self.rms
            self.iter_number += quenchres.nsteps

    def printState(self, force=True):
        cond = (self.iprint > 0 and self.iter_number%self.iprint == 0) or force
        if cond:
            print self.pstr.format(self.iter_number, self.f, self.E, self.rms)

    def stop_criterion(self):
        """test the stop criterion"""
        if self.alternate_stop_criterion is None:
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

        quenchres = self.quench_config()

        if self.iprint > 0:
            print self.qstr.format(quenchres.energy,quenchres.nsteps,quenchres.rms)

        self.X = quenchres.coords
        self.G = quenchres.grad
        self.rms = quenchres.rms
        self.estrms = self.rms
        self.iter_number += quenchres.nsteps

        return self.get_result()

    def quench_config(self):
        return self.quench(self.X, self.pot,
                           maxstep=np.linalg.norm(self.X-self.Xlast),
                           energy=self.E, gradient=self.G,
                           tol=self.quenchtol, **self.quenchkw)

    def get_result(self):
        res = self.result
        res.nsteps = self.iter_number
        res.coords = self.X
        res.energy = self.E
        res.rms = self.rms
        res.grad = self.G
        res.Emax = np.array(self.Emax)
        if self.store_configs:
            res.configs = np.array(self.configs)
        res.success = self.stop_criterion()
        return res

class AdaptiveNestedOptimizer(NestedOptimizer):

    def __init__(self, X, pot, sampler,
                 tol=1e-1, alternate_stop_criterion=None,
                 events=None, iprint=-1, nsteps=10000, debug=False,
                 energy=None, gradient=None, store_configs=False,
                 stepsize=None, acc_ratio=0.5, step_factor=1.1, frequency=100,
                 quench=lbfgs_cpp, quenchtol=1e-6, quench_kw={}):

        self.X = np.array(X)        # Copy initial coordinates
        self.Xi = self.X.copy()     # save initial coordinates
        self.Xlast = self.X.copy()  # save last coordinates
        self.dX = self.X.copy()     # save difference in coordinates

        self.pot = pot
        self.sampler = sampler

        # Passing None gives default stepsize
        self.stepsize = 1.0 if stepsize is None else stepsize

        # a list of events to run during the optimization
        self.events = [] if events is None else events
        self.iprint = iprint
        self.nsteps = nsteps
        self.tol = tol

        self.alternate_stop_criterion = alternate_stop_criterion
        self.debug = debug  # print debug messages

        self.store_configs = store_configs

        # To store information from the first iteration
        self.initial = True
        self.ireject = 0
        self.isteps = 0
        self.firststep = self.stepsize

        # Variables for setting adaptive stepsize
        self.initialstep = self.stepsize
        self.step_factor = step_factor
        self.frequency = frequency
        self.acc_ratio = acc_ratio

        self.rel_ratio = self.acc_ratio/(1. - self.acc_ratio)
        self.factor_step = self.step_factor**(1./self.frequency)

        if energy is None and gradient is None:
            self.E, self.G = self.pot.getEnergyGradient(self.X)
        elif energy is not None:
            self.E = energy
            self.G = self.pot.getGradient(self.X)
        else:
            self.E = energy
            self.G = gradient

        self.rms = np.linalg.norm(self.G) / np.sqrt(self.G.size)
        self.estrms = self.rms

        self.iter_number = 0
        self.f = 1.
        self.tot_rejects = 0
        self.tot_steps = 0
        self.result = Result()
        self.result.initialcoords = self.Xi
        self.result.message = []

        self.Emax = []
        if self.store_configs:
            self.configs = []
            self.stepsizes = []

        self.quench = quench
        self.quenchtol = quenchtol
        self.quenchkw = quench_kw
        dict_update_keep(self.quenchkw, dict(iprint=self.iprint))

    def update_stepsize(self, naccept, nreject):
        self.stepsize *= self.factor_step ** (naccept - self.rel_ratio * nreject)

    def one_iteration(self):
        try:
            Xnew, Enew, naccept, nreject = self.sampler.new_point(
                self.E, self.X, stepsize=self.stepsize)

            self.update_stepsize(naccept, nreject)

            self.f /= 2

            if Enew > self.E:
                ## something went wrong
                raise NestedSamplingError(
                    self.E, Enew, ", naccept {}, nreject {}".format(naccept,nreject))

            self.tot_rejects += nreject
            self.tot_steps += naccept

            self.Xlast[:] = self.X
            self.X[:] = Xnew

            self.E = Enew
            self.G = self.pot.getGradient(self.X)
            self.rms = np.linalg.norm(self.G)/np.sqrt(self.X.size)

            self.Emax.append(Enew)
            if self.store_configs:
                self.configs.append(self.X.copy())
                self.stepsizes.append(self.stepsize)

            self.printState(False)

            for event in self.events:
                event(coords=self.X, energy=self.E, rms=self.rms,
                      stepsize=self.stepsize)

            if self.initial:
                self.isteps = naccept
                self.ireject = nreject
                self.initial = False
                self.firststep = self.stepsize

            self.iter_number += 1

        except SamplingError as e:
            nreject, naccept = e.kwargs['nreject'], e.kwargs['naccept']
            self.update_stepsize(naccept, nreject)
            if self.debug:
                print "NOPT > Sampling error, updating stepsize = {:8.3g}".format(self.stepsize)
                print 'initial stepsize', self.firststep, self.iter_number

    def get_result(self):
        res = self.result
        res.nsteps = self.iter_number
        res.coords = self.X
        res.energy = self.E
        res.rms = self.rms
        res.grad = self.G
        res.Emax = np.array(self.Emax)
        if self.store_configs:
            res.configs = np.array(self.configs)
            res.stepsizes = np.array(self.stepsizes)
        res.success = self.stop_criterion()
        return res

class NestedGalileanOptimizer(AdaptiveNestedOptimizer):

    def __init__(self, X, pot,
                 tol=1e-1, alternate_stop_criterion=None, constraint=None,
                 events=None, iprint=-1, nsteps=10000, debug=False,
                 energy=None, gradient=None, sampler_kw={},
                 stepsize=1.0, acc_ratio=0.5, step_factor=1.1, frequency=100,
                 quench=lbfgs_cpp, quenchtol=1e-6, quench_kw={}):

        self.X = np.array(X)
        self.constraint = constraint
        dict_update_keep(sampler_kw, dict(stepsize=stepsize,
                                          nsteps=self.X.size/3,
                                          constraint=constraint))
        self.sampler_kw = sampler_kw
        sampler = GMCSampler(pot, **sampler_kw)

        super(self.__class__, self).__init__(
            X, pot, sampler,
            tol=tol, events=events, iprint=iprint, nsteps=nsteps,
            debug=debug, quench=lbfgs_cpp, quenchtol=1e-6,
            stepsize=stepsize, acc_ratio=acc_ratio,
            step_factor=step_factor, frequency=frequency,
            alternate_stop_criterion=alternate_stop_criterion, quench_kw=quench_kw)

if __name__  == "__main__":

    from pele.potentials import LJ, BasePotential
    from pele.optimize import lbfgs_cpp

    from nestedbasinsampling.structure.constraints import HardShellConstraint
    from nestedbasinsampling.samplers import MCSampler
    from nestedbasinsampling.random.takestep import random_structure

    import matplotlib.pyplot as plt
    from plottingfuncs.plotting import ax3d

    natoms = 31
    radius =  float(natoms) ** (1. / 3)
    rand_config = lambda : random_structure(natoms, radius)
    constraint = HardShellConstraint(radius)
    pot = LJ()

    gsampler = GMCSampler(pot, stepsize=1, nsteps=30,
                               constraint=constraint)

    coords = rand_config()

    opt = AdaptiveNestedOptimizer(coords, pot, gsampler, stepsize=0.5, nsteps=3000, iprint=100)
    res = opt.run()

    fig, ax = ax3d()
    ax.scatter(*coords.reshape(-1,3).T, c='r')
    ax.scatter(*res.coords.reshape(-1,3).T, c='b')

if  __name__  == "__main__":
    from scipy.optimize import rosen, rosen_der
    from nestedbasinsampling.utils import call_counter

    class RosenPot(BasePotential):

        @call_counter
        def getEnergy(self, x):
            return rosen(x)

        @call_counter
        def getGradient(self, x):
            return rosen_der(x)

        @call_counter
        def getEnergyGradient(self, x):
            return self.getEnergy(x), self.getGradient(x)

    pot = RosenPot()

    x = np.linspace(0.,2.0,1025)
    XY = np.array(np.meshgrid(x,x))
    xy = XY.reshape(2,-1)

    E2 = np.log(rosen(xy).reshape(XY[0].shape) + 1e-10)

    coords = np.array([1.5,1.5])

    f, ax = plt.subplots()
    ax.contour(XY[0],XY[1],E2,zorder=0)
    ax.scatter(coords[0], coords[1], c='r',zorder=10)

    def plot(coords=None,**kwargs):
        ax.scatter(coords[0],coords[1], c='r', alpha=0.5, zorder=10)

    opt = NestedGalileanOptimizer(coords, pot, tol=1e-1, iprint=1, nsteps=100, events=[plot],
                                  sampler_kw=dict(stepsize=0.1, nsteps=30))
    res = opt.run()

    print pot.getEnergy.calls, pot.getGradient.calls, pot.getEnergyGradient.calls

    lbfgsres = lbfgs_cpp(coords, pot)

    print pot.getEnergy.calls, pot.getGradient.calls, pot.getEnergyGradient.calls


