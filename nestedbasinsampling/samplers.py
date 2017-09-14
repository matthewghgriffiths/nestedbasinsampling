# -*- coding: utf-8 -*-

import numpy as np

from nestedbasinsampling.utils import Replica
from nestedbasinsampling.random import vector_random_uniform_hypersphere as rand_hsphere
from nestedbasinsampling.takestep import AdaptiveStepsize, TakestepHyperSphere

class BaseSampler(object):
    """
    Class for sampling a potential less than a given energy cutoff
    """
    def find_stepsize(self, coords, stepsize=None, energy=None, gradient=None):
        raise NotImplementedError

    def new_point(self, Ecut, coords, **kwargs):
        raise NotImplementedError

    def new_points(self, Ecut, N, replicas=None, driver=None):
        """
        Generate N replicas below Ecut with energy < Ecut
        """
        newreplicas = []

        for _ in xrange(N):
            start = np.random.choice(replicas)

            coords = start.x.copy()
            newcoords, Enew = self.new_point(Ecut, coords)[:2]

            replica = Replica(newcoords, Enew,
                              niter=self.niter, from_random=False)

            newreplicas.append(replica)

        return newreplicas

class MCSampler(BaseSampler):

    def __init__(self, pot, takestep=AdaptiveStepsize(TakestepHyperSphere()),
                 nsteps=100, maxtries=1000, verbose=False, constraint=None):

        self.pot = pot
        self.niter = 0
        self.nfev = 0
        self.nsteps = nsteps
        self.maxtries = maxtries
        self.takestep = takestep
        self.constraint = lambda x: True if constraint is None else constraint
        self.verbose = verbose

        self.pstr = "found energy = {:10.12g} after {} steps and {} tries"

    def new_point(self, Ecut, coords, nsteps=None):

        niter = 0
        ntries = 0

        nsteps = self.nsteps if nsteps is None else nsteps

        newcoords = coords.copy()

        while(niter<nsteps):

            self.takestep.takeStep(newcoords)

            acceptMove = self.constraint(newcoords)

            if acceptMove:
                Enew = self.pot.getEnergy(newcoords)
                acceptMove = Enew < Ecut

            if acceptMove:
                niter += 1
            else:
                newcoords[:] = coords
            ntries += 1

            try:
                self.takestep.updateStep(acceptMove)
            except AttributeError:
                pass

            if ntries >= self.maxtries and niter == 0:
                break

        self.niter += 1

        return newcoords, Enew


#    def new_points(self, Ecut, N, replicas, driver=None):
#
#        newreplicas = []
#        for _ in xrange(N):
#            ntries = 0
#            nsteps = 0
#            start = np.random.choice(replicas)
#
#            coords = start.x.copy()
#            newcoords = coords.copy()
#
#            while(nsteps<self.nsteps):
#
#                self.takestep.takeStep(newcoords)
#
#                newE = self.pot.getEnergy(newcoords)
#
#                acceptMove = newE < Ecut
#
#                if acceptMove:
#                    nsteps += 1
#                else:
#                    newcoords[:] = coords
#                ntries += 1
#
#                try:
#                    self.takestep.updateStep(acceptMove)
#                except AttributeError:
#                    pass
#
#                if ntries >= self.maxtries and nsteps == 0:
#                    break
#
#            self.niter += 1
#            replica = Replica(newcoords, newE,
#                              niter=self.niter, from_random=False)
#
#            if self.verbose:
#                print self.pstr.format(newE,nsteps,ntries)
#
#            newreplicas.append(replica)
#
#        return newreplicas

class GalileanSampler(BaseSampler):

    pstr = "HMC: E = {:10.12g}, nsteps = {}, nreject = {}, stepsize = {:10.12g}"

    def __init__(self, pot, takestep=TakestepHyperSphere(), constraint=None,
                 nsteps=100, stepsize=1.0, acc_ratio=0.5, factor=0.2,
                 minstep=1e-5, maxstep=2.0, maxreject=1000, adaptive=True,
                 verbose=False, searchratio=2.):

        self.pot = pot
        self.niter = 0

        self.nsteps = nsteps
        self.maxreject = maxreject

        self.constraint = (lambda x: True) if constraint is None else constraint

        self.stepsize = stepsize
        self.minstep = minstep
        self.maxstep = maxstep
        self.acc_ratio = acc_ratio
        self.adaptive = adaptive
        self.factor = factor

        self.searchratio = searchratio

        #self.takestep = takestep
        self.verbose = verbose

        #self.pstr = "found energy = {:10.12g} after {} steps and {} rejections"

    def gen_p(self, coords, stepsize):
        """
        Generates random direction vector
        """
        p = rand_hsphere(coords.size).reshape(coords.shape)*stepsize
        return p

    def reflect(self, coords, p):

        G = self.pot.getGradient(coords)
        n = np.linalg.norm(G)
        G /= n
        p -= 2 * p.dot(G) * G

    def adjust_stepsize(self, nsteps, nreject, newcoords):
        curr_rate = np.clip((nsteps - nreject + 1.)/(nsteps + 1.),0.,1.)
        self.stepsize *= np.exp(self.factor*(curr_rate-self.acc_ratio))

    def find_stepsize(self, coords, stepsize=None, energy=None, gradient=None):

        stepsize = self.stepsize if stepsize is None else stepsize
        E = self.pot.getEnergy(coords) if energy is None else energy
        G = self.pot.getGradient(coords) if gradient is None else gradient
        G /= np.linalg.norm(G)

        Estep = self.pot.getEnergy(coords - stepsize*G)

        if Estep > E:
            while(Estep > E):
                stepsize /= self.searchratio
                Estep = self.pot.getEnergy(coords - stepsize*G)
                if stepsize < self.minstep:
                    stepsize = self.minstep
                    break
        else:
            while(Estep < E):
                stepsize *= self.searchratio
                Estep = self.pot.getEnergy(coords - stepsize*G)
                if stepsize > self.maxstep:
                    stepsize = self.maxstep*self.searchratio
                    break
            stepsize /= self.searchratio

        return stepsize

    def new_point(self, Ecut, coords, nsteps=None, stepsize=None):

        nsteps = self.nsteps if nsteps is None else nsteps
        stepsize = self.find_stepsize(coords) if stepsize is None else stepsize

        nreject = 0

        newcoords = np.array(coords)
        self.p = self.gen_p(newcoords, stepsize)
        testcoords = np.empty_like(newcoords)

        for i in xrange(nsteps):

            testcoords[:] = newcoords +  self.p
            Enew = self.pot.getEnergy(testcoords)

            while Enew >= Ecut or not self.constraint(testcoords):
                nreject += 1
                self.reflect(testcoords, self.p)
                testcoords[:] = newcoords +  self.p
                Enew = self.pot.getEnergy(testcoords)

                if nreject > self.maxreject:
                    stepsize = self.find_stepsize(coords,
                                                  stepsize/self.searchratio,
                                                  energy=Ecut)
                    return self.new_point(Ecut, coords, nsteps, stepsize)

            newcoords[:] = testcoords

        #if self.adaptive:
        #    self.adjust_stepsize(nsteps, nreject, newcoords)

        if self.verbose:
            print self.pstr.format(Enew, nsteps, nreject, stepsize)

        self.niter += 1

        return newcoords, Enew, nsteps, nreject

#    def new_points(self, Ecut, N, replicas, driver=None):
#
#        newreplicas = []
#
#        for _ in xrange(N):
#            start = np.random.choice(replicas)
#
#            coords = start.x.copy()
#            newcoords, Enew, nsteps, nreject = self.new_point(Ecut, coords,
#                                                              self.nsteps)
#
#            replica = Replica(newcoords, Enew,
#                              niter=self.niter, from_random=False)
#
#            newreplicas.append(replica)
#
#        return newreplicas


if __name__  == "__main__":
    from scipy.optimize import rosen, rosen_der, rosen_hess, minimize

    from pele.potentials import BasePotential

    import numpy as np
    import matplotlib.pyplot as plt


    class RosenPot(BasePotential):

        def getEnergy(self, x):
            return rosen(x)

        def getGradient(self, x):
            return rosen_der(x)

        def getEnergyGradient(self, x):
            return self.getEnergy(x), self.getGradient(x)

    pot = RosenPot()

    x = np.linspace(0.,2.0,1025)
    XY = np.array(np.meshgrid(x,x))

    xy = XY.reshape(2,-1)

    e = rosen(xy)
    E = np.log(e.reshape(XY[0].shape) + 1e-10)

    gsampler = GalileanSampler(pot, stepsize=0.1)
    self = gsampler

    coords = np.array([1.5,1.5])
    Ecut = self.pot.getEnergy(coords)

    plt.contour(XY[0],XY[1],E)
    plt.scatter(coords[0],coords[1], c='g')

    newcoords = np.array(coords)
    Ecut = self.pot.getEnergy(newcoords)
    for i in xrange(100):
        newcoords, Ecut, nsteps, nreject = gsampler.new_point(Ecut, newcoords, 20)
        plt.scatter(newcoords[0],newcoords[1], c='b')
        print Ecut, nreject*1./20, gsampler.stepsize














