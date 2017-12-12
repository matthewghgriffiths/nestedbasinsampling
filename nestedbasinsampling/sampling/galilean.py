# -*- coding: utf-8 -*-
from math import log
import numpy as np

from scipy.optimize import fixed_point

from pele.optimize import lbfgs_cpp

from nestedbasinsampling.utils import SamplingError, Result
from nestedbasinsampling.structure.constraints import BaseConstraint
from nestedbasinsampling.sampling.takestep import hypersphere_step

try:
    from nestedbasinsampling.sampling.fortran import galilean
    has_fortran = True
except ImportError:
    has_fortran = False

class BaseSampler(object):
    """
    Class for sampling a potential less than a given energy cutoff
    """

    acc_ratio = 0.5
    r = 3.
    a = 0.672924
    b = 0.0644284
    debug=False

    nsteps = 10
    nadapt = 10

    detect_str = ("adapt_step > target_acc = {:8.4g}, current_acc = {:8.4g}, "
                  + "stepsize = {:8.4g}, nsteps= {:4d}")

    def set_acc_ratio(self, acc_ratio, r=3.):
        """ Sets the parameters as needed by calcStepFactors to
        calculate the factor by which to multiply the stepsize.

        parameters
        ----------
        acc_ratio : float
            The target acceptance ratio
        r : float
            The maximum factor by which the the stepsize can change in
            a single iteration
        """
        self.acc_ratio = acc_ratio
        self.r = r
        self.a, self.b = self._calcStepFactors(self.acc_ratio, r)

    def _calcStepFactors(self, acc_ratio, r):
        """
        Calculates the parameters by which to change the stepsize using the
        formula as suggested in:

            Swendsen, R. H. (2011).
            How the maximum step size in Monte Carlo simulations
                should be adjusted.
            Physics Procedia, 15, 81–86.
            http://doi.org/10.1016/j.phpro.2011.06.004
        """
        def iterStepFactor(ab, acc_ratio=0.5, r=3):
            a,b = ab
            c = (a*acc_ratio + b)**r
            a = (a*acc_ratio + b)**(1./r) - c
            return np.r_[a, c]
        a, b = fixed_point(iterStepFactor, np.r_[0.1,0.1], args=(acc_ratio,r))
        return a, b

    def new_point(self, Ecut, coords, **kwargs):
        raise NotImplementedError

    def __call__(self, Ecut, coords, stepsize=None, nsteps=None, nadapt=None):
        raise NotImplementedError

    def stepsize_factor(self, p):
        """ Calculates the factor by which to change the stepsize given
        an acceptance probability of p and a target acceptance probabiliy
        of self.acc_ratio

        parameters
        ----------
        p : float
            current acceptance probability

        returns
        -------
        stepsize_factor : float
            the factor by which to multiply the stepsize
        """
        return log(self.a*self.acc_ratio + self.b) / log(self.a*p + self.b)

    def adapt_step(self, Ecut, coords, stepsize=None, nsteps=None, nadapt=None):

        stepsize = self.stepsize if stepsize is None else stepsize
        nsteps = self.nsteps if nsteps is None else nsteps
        nadapt = self.nadapt if nadapt is None else nadapt

        newcoords = coords.copy()
        i = 0
        while i < nadapt:
            try:
                newcoords, Enew, naccept, nreject, nr = \
                    self.new_point(
                        Ecut, newcoords, nsteps=nsteps, stepsize=stepsize)[:5]
                i += 1
            except SamplingError as e:
                naccept = e.kwargs['naccept']
                nreject = e.kwargs['nreject']
                if self.debug:
                    print 'DetectStep> warning: sampling failed'

            current_acc = 1. * naccept / (naccept + nreject + nr)
            stepsize = stepsize * self.stepsize_factor(current_acc)

            if self.debug:
                print self.detect_str.format(
                    self.acc_ratio, current_acc, stepsize, nsteps)

        return stepsize, newcoords, Enew, current_acc

    def _fixConstraint(self, coords, **kwargs):
        coords = np.array(coords) # Make a copy
        conpot = self.constraint.getGlobalPotential(coords)
        disp = np.zeros(3)
        res = lbfgs_cpp(disp, conpot, **kwargs)
        pos = coords.reshape(-1,3)
        pos += res.coords[None,:]
        return coords

class py_GalileanSampler(BaseSampler):

    pstr = ("GMC  > nsteps  ={:4d}, nreject ={:4d}, "+
            "E    ={:10.5g}, stepsize ={:10.5g}")

    def __init__(
        self, pot, genstep=hypersphere_step, constraint=None,
        nsteps=10, nadapt=10, stepsize=0.1, acc_ratio=0.1,
        testinitial=True, fixConstraint=False, maxreject=1000, debug=False):

        self.pot = pot
        self.constraint = BaseConstraint() if constraint is None else constraint
        self.gen_p = genstep

        self.niter = 0

        self.testinitial = testinitial

        if callable(fixConstraint):
            self.fixConstraint = fixConstraint
        elif fixConstraint:
            self.fixConstraint = self._fixConstraint
        else:
            def not_implemented(*arg, **kwargs):
                raise NotImplementedError
            self.fixConstraint = not_implemented

        self.nsteps = nsteps
        self.nadapt = nadapt
        self.maxreject = maxreject
        self.stepsize = stepsize

        self.set_acc_ratio(acc_ratio)
        self.debug = debug

    def __call__(self, Ecut, coords, stepsize=None, nsteps=None, nadapt=None):
        """ Samples a new point within the energy contour defined by Ecut
        starting from coords.
        """
        stepsize = self.stepsize if stepsize is None else stepsize
        nsteps = self.nsteps if nsteps is None else nsteps
        nadapt = self.nadapt if nadapt is None else nadapt

        res = Result()
        res.accepts = []
        res.stepsizes = []
        res.energies = []

        newcoords = coords.copy()
        i = 0
        while i < nadapt:
            try:
                newres = self.new_point(
                    Ecut, newcoords, nsteps=nsteps, stepsize=stepsize)
                newcoords = newres.coords
                stepsize = newres.stepsize
                accept = newres.accept
                Enew = newres.energy
                i += 1
            except SamplingError as e:
                if e.kwargs.has_key('naccept'):
                    accept = e.kwargs
                    Enew = None
                    if self.debug:
                        print 'GMC > warning: sampling failed'
                else:
                    raise e

            naccept = accept['naccept']
            niter = accept['niter']
            nreject = accept['nreject']

            current_acc = 1. * naccept / (niter+nreject)
            stepsize = stepsize * self.stepsize_factor(current_acc)

            res.energies.append(Enew)
            res.stepsizes.append(stepsize)
            res.accepts.append(accept)

            if self.debug:
                print self.detect_str.format(
                    self.acc_ratio, current_acc, stepsize, nsteps)

        res.coords = newres.coords
        res.energy = newres.energy
        res.grad = newres.grad
        res.stepsize = stepsize

        return res

    def new_point(self, Ecut, coords, nsteps=None, stepsize=None):

        nsteps = self.nsteps if nsteps is None else nsteps

        if stepsize is None:
            stepsize = self.stepsize

        niter = 0
        naccept = 0
        nreflect = 0
        nreject = 0

        ncon = 0
        nE = 0
        nEcon = 0
        Enew = np.inf

        newcoords = np.array(coords)

        if self.testinitial:
            Estart = self.pot.getEnergy(newcoords)
            Econ = self.constraint.getEnergy(newcoords)
            if Estart > Ecut:
                raise SamplingError(
                    "Starting energy higher than cutoff",
                    Estart=Estart, Ecut=Ecut)
            if Econ > 0:
                try:
                    newcoords = self.fixConstraint(newcoords)
                except NotImplementedError:
                    raise SamplingError(
                        "Starting configuration doesn't satisfy constraint",
                        Estart=Estart, Econstraint=Econ)

        p = self.gen_p(newcoords, stepsize)
        testcoords = np.empty_like(newcoords)

        while(niter < nsteps):

            assert np.isfinite(p).all()

            Eold = Enew # save previous energy
            testcoords[:] = newcoords +  p

            Econ, Gcon = self.constraint.getEnergyGradient(testcoords)
            Enew, Gnew = self.pot.getEnergyGradient(testcoords)

            constraint = Econ <= 0
            Eaccept = Enew <= Ecut

            if Eaccept and constraint:
                newcoords[:] = testcoords
                naccept += 1
                niter += 1

            else:
                if Eaccept and not constraint:
                    n = Gcon/np.linalg.norm(Gcon)
                    ncon += 1
                elif constraint and not Eaccept:
                    n = Gnew/np.linalg.norm(Gnew)
                    nE += 1
                else:
                    n1 = Gnew/np.linalg.norm(Gnew)
                    n2 = Gcon/np.linalg.norm(Gcon)
                    n = n1 + n2
                    n /= np.linalg.norm(n)
                    nEcon += 1

                p -= 2. * n * n.ravel().dot(p.ravel())
                testcoords += p

                Econ = self.constraint.getEnergy(testcoords)
                Enew = self.pot.getEnergy(testcoords)

                constraint = Econ <= 0
                Eaccept = Enew <= Ecut
                if Eaccept and constraint:
                    newcoords[:] = testcoords
                    niter += 1
                    nreflect += 1
                else:
                    Enew = Eold
                    nreject += 1
                    #p *= -1
                    #niter += 1

                p = self.gen_p(newcoords, stepsize)

                if nreject > self.maxreject:
                    raise SamplingError(
                        'HMC  > failed to find point',
                        stepsize=stepsize, Ecut=Ecut, niter=niter,
                        naccept=naccept, nreflect=nreflect, nreject=nreject)

        res = Result(
            energy=Enew, coords=newcoords, stepsize=stepsize, grad=Gnew,
            accept=dict(naccept=naccept, nreflect=nreflect,
                        nreject=nreject, niter=niter))
        return res

class f90_GalileanSampler(py_GalileanSampler):

    galilean = galilean.galilean

    def set_seed(self, seed):
        self.galilean.seed = seed

    def get_seed(self):
        return self.galilean.seed

    def new_point(self, Ecut, coords, nsteps=None, stepsize=None):
        """
        """
        nsteps = self.nsteps if nsteps is None else nsteps
        stepsize = self.stepsize if stepsize is None else stepsize


        energy, grad, newcoords, naccept, nreject, nreflect, niter, info = (
            self.galilean.newpoint(
                Ecut, coords, nsteps, self.maxreject, stepsize,
                self.pot.getEnergyGradient,
                self.constraint.getEnergyGradient))


        if info==1:
            res = Result(
                energy=energy, coords=newcoords,
                stepsize=stepsize, grad=grad,
                accept=dict(naccept=naccept, nreflect=nreflect,
                            nreject=nreject, niter=niter))
            return res
        elif info==2:
            raise SamplingError(
                "GMC > Starting configuration not valid",
                Ecut=Ecut, coords=coords)
        elif info==3:
            raise SamplingError(
                'GMC  > failed to find point',
                stepsize=stepsize, Ecut=Ecut, niter=niter,
                naccept=naccept, nreflect=nreflect, nreject=nreject)
        else:
            raise SamplingError(
                'GMC > Something went wrong',
                stepsize=stepsize, Ecut=Ecut, niter=niter,
                naccept=naccept, nreflect=nreflect, nreject=nreject)

if has_fortran:
    GalileanSampler = f90_GalileanSampler
else:
    GalileanSampler = py_GalileanSampler

if __name__ == "__main__":

    import seaborn as sns
    import matplotlib.pyplot as plt
    from plottingfuncs.plotting import ax3d

    from pele.potentials import BasePotential
    from nestedbasinsampling.sampling.takestep import vector_random_uniform_hypersphere
    from nestedbasinsampling.sampling.stats import CDF, AndersonDarling

    class MyPot(BasePotential):

        def __init__(self, M):
            self.M = np.array(M, float)

        def getEnergy(self, x):
            return  self.M.dot(x).dot(x)

        def getGradient(self, x):
            return 2*self.M.dot(x)

        def getEnergyGradient(self, x):
            G = self.M.dot(x)
            return G.dot(x), 2*G

    class RosenPot(BasePotential):

        a = 0.
        b = 100.

        def getEnergy(self, x):
            y = np.array(x)
            y[0] = self.a - x[0]
            y[1] -= x[0]**2
            y *= y
            y[1:] *= self.b
            return y.sum()

        def getGradient(self, x):
            y = np.array(x)
            y[0] = self.a - x[0]
            y[1] -= x[0]**2

            G = np.array(y)
            G[0] = -2 * y[0] - 4*self.b*x[0]*y[1]
            G[1:] *= 2

            return G

        def getEnergyGradient(self, x):
            y = np.array(x)
            y[0] = self.a - x[0]
            y[1] -= x[0]**2

            G = np.array(y)
            G[0] = -2 * y[0] - 4*self.b*x[0]*y[1]
            G[1:] *= 2 * self.b
            y *= y
            y[1:] *= self.b
            return y.sum(), G

    def e2Rosen(x, a=0, b=100):
        y = np.array(x)
        y[0] = a - x[0]
        y[1] = x[1] + y[0]**2
        return y


    if 0:
        res = 513
        x = np.linspace(-2, 2, res)
        X,Y = np.meshgrid(x,x)

        pot = RosenPot()
        epot = MyPot(np.diag([1,100]))
        Z = np.empty_like(X)
        Ze = np.empty_like(X)
        for i in xrange(res):
            for j in xrange(res):
                Z[i,j] = pot.getEnergy([X[i,j],Y[i,j]])
                Ze[i,j] = epot.getEnergy([X[i,j],Y[i,j]])

        plt.contour(X,Y,np.log(Z))

        Zf = Z.flatten()
        Zfe = Ze.flatten()
        Zf.sort(); Zfe.sort()

        Zf0 = Zf[Zf<20]
        Zfe0 = Zfe[Zfe<20]

        plt.plot(np.linspace(0,1,Zf0.size), Zf0)
        plt.plot(np.linspace(0,1,Zfe0.size), Zfe0)

        plt.plot(Zf);plt.plot(Zfe)

    if 0:
        res = 129
        x = np.linspace(-2, 2, res)
        X,Y,Z = np.meshgrid(x,x,x)

        pot = RosenPot()
        epot = MyPot(np.diag([1,100,100.]))

        E = np.empty_like(X)
        Ee = np.empty_like(X)
        for i in xrange(res):
            for j in xrange(res):
                for k in xrange(res):
                    E[i,j,k] = pot.getEnergy([X[i,j,k],Y[i,j,k],Z[i,j,k]])
                    Ee[i,j,k] = epot.getEnergy([X[i,j,k],Y[i,j,k],Z[i,j,k]])

        Ef = E.flatten()
        Eef = Ee.flatten()

        Ef0 = Ef[Ef < 30.]
        Efe0 = Eef[Eef < 30.]
        Ef0.sort()
        Efe0.sort()

    ndim = 100
    pot = RosenPot()
    epot = MyPot(np.diag([1]+[100]*(ndim-1)))

    L = np.linalg.cholesky(epot.M)
    iL = np.linalg.inv(L)

    scale = 10

    Ehs = [epot.getEnergy(iL.dot(vector_random_uniform_hypersphere(ndim))*scale)
            for _ in xrange(10000)]

    Ecut = scale**2

    x = iL.dot(vector_random_uniform_hypersphere(ndim)) * scale
    y = e2Rosen(x)
    sampler = GalileanSampler(pot, nsteps=100, acc_ratio=0.1)
    sampler = py_GalileanSampler(pot, nsteps=100, acc_ratio=0.1)
    sampler.new_point(Ecut, y)
    res = sampler(Ecut, y, nsteps=8, nadapt=8)


    Es = []
    Eall = []
    stepsizes = []
    accept = dict(naccept=0, niter=0, nreflect=0, nreject=0)

    for i in xrange(1000):
        res = sampler(
            Ecut, y, stepsize=0.1, nsteps=15, nadapt=15)
        Eall.append(res.energies)
        Es.append(res.energy)
        stepsizes.extend(res.stepsizes)
        for a in res.accepts:
            for k, v in a.iteritems():
                accept[k] += v

    CDF(Es).plot()
    CDF(Ehs).plot()
    print AndersonDarling.compareDistributions([CDF(Es), CDF(Ehs)])
    print accept
    print accept['nreject']*1./accept['niter'], accept['nreflect']*1./accept['niter']

    raise

    Ealls = np.array(Eall)
    cdfs = [CDF(_Es) for _Es in Ealls.T]

    [c.plot() for c in cdfs]

    sig = [AndersonDarling.compareDistributions([c, CDF(Ehs)]) for c in cdfs]

    plt.figure()
#    nEs = np.array(Es)
#    nEs -= nEs.mean()
#    autocorr = np.correlate(nEs, nEs, mode='same')
#    autocorr = autocorr[autocorr.size/2:]
#    plt.plot(autocorr)

    steps = np.array(stepsizes)
    steps -= steps.mean()
    autocorr = np.correlate(steps, steps, mode='same')
    plt.plot(autocorr)



