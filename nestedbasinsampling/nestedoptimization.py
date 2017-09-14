# -*- coding: utf-8 -*-

import logging
import numpy as np

from pele.optimize import lbfgs_cpp

from nestedbasinsampling.samplers import GalileanSampler
from nestedbasinsampling.database import Database
from nestedbasinsampling.utils import (Result, NestedSamplingError,
                                       dict_update_keep, call_counter)


_logger = logging.getLogger("nestedbasinsampling")


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
    logger : logger object
        messages will be passed to this logger rather than the default
    energy, gradient : float, float array
        The initial energy and gradient.  If these are both not None then the
        energy and gradient of the initial point will not be calculated, saving
        one potential call.
    """

    def __init__(self, X, pot, sampler,
                 tol=1e-1, alternate_stop_criterion=None,
                 events=None, iprint=-1, nsteps=10000, logger=None, debug=False,
                 energy=None, gradient=None,
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
        self.logger = _logger if logger is None else logger

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
        self.result = Result()
        self.result.initialcoords = self.Xi
        self.result.message = []

        self.Emax = [self.E]

        self.quench = quench
        self.quenchtol = quenchtol
        self.quenchkw = quench_kw
        dict_update_keep(self.quenchkw,
                         dict(events=self.events, iprint=self.iprint))

    def one_iteration(self):
        Xnew, Enew, nsteps, nreject = self.sampler.new_point(self.E, self.X)

        if Enew > self.E:
            ## something went wrong
            raise NestedSamplingError(
                self.E, Enew, ", nsteps {}, nreject {}".format(nsteps,nreject))

        self.dX[:] = Xnew - self.X
        self.d = np.linalg.norm(self.dX)

        self.g = (Enew - self.E)/self.d
        self.estrms = self.g/np.sqrt(self.dX.size)

        self.Xlast[:] = self.X
        self.X[:] = Xnew

        self.E = Enew
        self.Emax.append(Enew)

        if self.iprint > 0 and self.iter_number%self.iprint == 0:
            print "After {} iterations Ecut = {:10.12g}, rms = {:10.12g}".format(
                self.iter_number, self.E, self.rms)
            print self.d

        if self.initial:
            self.isteps = nsteps
            self.ireject = nreject
            self.initial = False

        self.iter_number += 1

    def stop_criterion(self):
        """test the stop criterion"""
        if self.alternate_stop_criterion is None:
            if self.estrms < self.tol:
                self.G = self.pot.getGradient(self.X)
                self.rms = np.linalg.norm(self.G) / np.sqrt(self.G.size)
                return self.rms < self.tol
            else:
                return False
        else:
            return self.alternate_stop_criterion(energy=self.energy,
                                                 gradient=self.G,
                                                 tol=self.tol, coords=self.X)

    def run(self):
        while self.iter_number < self.nsteps and not self.stop_criterion():
            try:
                self.one_iteration()
            except NestedSamplingError:
                self.result.message.append("problem with nested sampler")
                break

        quenchres = self.quench_config()

        self.X = quenchres.coords
        self.G = quenchres.grad
        self.rms = quenchres.rms
        self.estrms = self.rms
        self.iter_number += quenchres.nsteps

        return self.get_result()

    def quench_config(self):
        return self.quench(self.X, self.pot,
                           energy=self.E, gradient=self.G,
                           maxstep=self.d, tol=self.quenchtol, **self.quenchkw)

    def get_result(self):
        res = self.result
        res.nsteps = self.iter_number
        res.coords = self.X
        res.energy = self.E
        res.rms = self.rms
        res.grad = self.G
        res.Emax = np.array(self.Emax)
        res.success = self.stop_criterion()
        return res

class NestedGalileanOptimizer(NestedOptimizer):

    def __init__(self, X, pot,
                 tol=1e-1, alternate_stop_criterion=None, constraint=None,
                 events=None, iprint=-1, nsteps=10000, logger=None, debug=False,
                 energy=None, gradient=None, sampler_kw={},
                 quench=lbfgs_cpp, quenchtol=1e-6, quench_kw={}):

        self.X = np.array(X)
        stepsize = np.std(X)
        self.constraint = constraint
        dict_update_keep(sampler_kw, dict(stepsize=stepsize,
                                          nsteps=self.X.size,
                                          verbose=debug,
                                          constraint=constraint))
        self.sampler_kw = sampler_kw
        sampler = GalileanSampler(pot, **sampler_kw)

        super(self.__class__, self).__init__(
            X, pot, sampler,
            tol=tol, events=events, iprint=iprint, nsteps=nsteps,
            logger=logger, debug=debug, quench=lbfgs_cpp, quenchtol=1e-6,
            alternate_stop_criterion=alternate_stop_criterion, quench_kw=quench_kw)

    def one_iteration(self):

        stepsize = self.sampler.find_stepsize(self.X,
                                              energy=self.E, gradient=self.G)
        Xnew, Enew, nsteps, nreject = self.sampler.new_point(self.E,
                                                             self.X,
                                                             stepsize=stepsize)

        if Enew > self.E:
            ## something went wrong
            raise NestedSamplingError(
                self.E, Enew, ", nsteps {}, nreject {}".format(nsteps,nreject))

        self.dX[:] = Xnew - self.X
        self.d = np.linalg.norm(self.dX)

        self.Xlast[:] = self.X
        self.X[:] = Xnew
        self.E = Enew
        self.G = self.pot.getGradient(Xnew)

        self.rms = np.linalg.norm(self.G)/np.sqrt(self.X.size)

        self.Emax.append(Enew)

        if self.iprint > 0 and self.iter_number%self.iprint == 0:
            print "After {} iterations Ecut = {:10.12g}, rms = {:10.12g}".format(
                self.iter_number, self.E, self.rms)

        if self.initial:
            self.isteps = nsteps
            self.ireject = nreject
            self.initial = False

        self.iter_number += 1

    def stop_criterion(self):
        """test the stop criterion"""
        if self.alternate_stop_criterion is None:
            return self.rms < self.tol
        else:
            return self.alternate_stop_criterion(energy=self.E,
                                                 gradient=self.G,
                                                 tol=self.tol, coords=self.X)

if __name__ == "__main__":
    from pele.systems import LJCluster

    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D

    db =  Database(db='test.sql')

    natoms = 31
    system = LJCluster(natoms)

    pot = system.get_potential()

    coords = system.get_random_configuration()
    res = lbfgs_cpp(coords, pot)

    print pot.getEnergy(coords)
    print res

    opt = NestedGalileanOptimizer(coords, pot, tol=1e-1, iprint=1, nsteps=100,
                                  sampler_kw=dict(stepsize=0.1, maxstep=2.0,
                                                  nsteps=10, verbose=True))
    res = opt.run()


def plot3Dscatter(coords, **kwargs):
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.scatter(*coords.T, **kwargs)
    return ax

if False and __name__  == "__main__":
    from scipy.optimize import rosen, rosen_der, minimize#, rosen_hess

    from pele.potentials import BasePotential

    import matplotlib.pyplot as plt


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

    print pot.getEnergy.calls, pot.getGradient.calls, pot.getEnergyGradient.calls



    x = np.linspace(0.,2.0,1025)
    XY = np.array(np.meshgrid(x,x))

    xy = XY.reshape(2,-1)

    e = rosen(xy)
    E2 = np.log(e.reshape(XY[0].shape) + 1e-10)

    coords = np.array([1.5,1.5])

    res = lbfgs_cpp(coords, pot)

#    sampler = GalileanSampler(pot, stepsize=0.1, nsteps=10, verbose=True)
#    opt = NestedOptimizer(coords, pot, sampler, tol=1e-3, iprint=1, nsteps=1000)
#
#    res = opt.run()


    opt = NestedGalileanOptimizer(coords, pot, tol=1e-1, iprint=1, nsteps=100,
                                  sampler_kw=dict(stepsize=0.1, nsteps=10, verbose=True))
    res = opt.run()

    print pot.getEnergy.calls, pot.getGradient.calls, pot.getEnergyGradient.calls

    plt.contour(XY[0],XY[1],E2)

