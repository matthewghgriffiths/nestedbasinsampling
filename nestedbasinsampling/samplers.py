# -*- coding: utf-8 -*-

import numpy as np

from nestedbasinsampling.utils import Replica, Result
from nestedbasinsampling.random import vector_random_uniform_hypersphere as rand_hsphere
from nestedbasinsampling.takestep import AdaptiveStepsize, TakestepHyperSphere, hypersphere_step
from nestedbasinsampling.constraints import BaseConstraint


class SamplingError(Exception):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return str(self.args) + "\n" + str(self.kwargs)

class BaseSampler(object):
    """
    Class for sampling a potential less than a given energy cutoff
    """
    debug=False
    def find_stepsize(self, coords, stepsize=None, energy=None, gradient=None):
        raise NotImplementedError

    def new_point(self, Ecut, coords, **kwargs):
        raise NotImplementedError

    def new_points(self, Ecut, N, replicas=None, driver=None):
        """
        Generate N replicas below Ecut with energy < Ecut
        """
        newreplicas = []

        while(len(newreplicas)<N):
            try:
                start = np.random.choice(replicas)
                coords = start.x.copy()
                newcoords, Enew = self.new_point(Ecut, coords)[:2]

                replica = Replica(newcoords, Enew,
                                  niter=self.niter, from_random=False)
                newreplicas.append(replica)
            except SamplingError as e:
                if self.debug:
                    print e

        return newreplicas

class ConstrainedMCSampler(BaseSampler):

    def __init__(self, pot, takestep=AdaptiveStepsize(TakestepHyperSphere()),
                 nsteps=100, maxtries=1000, debug=False,
                 constraint=None):

        self.pot = pot
        self.niter = 0
        self.nfev = 0
        self.nsteps = nsteps
        self.maxtries = maxtries
        self.takestep = takestep
        self.constraint = BaseConstraint() if constraint is None else constraint
        self.debug = debug

        self.pstr = "found energy = {:10.12g} after {} steps and {} tries"

    def reflect(self, coords, p, pot, G=None):
        """
        Applies hard shell constraint to momentum vector p

        Parameters
        ----------
        pos : numpy.array
            starting position
        p : numpy.array
            initial momentum, must be same shape as pos and is modified
        pot : Potential
            needs getEnergy and getGradient methods
        G : numpy.array, optional
            gradient at coords
        """
        G = pot.getGradient(coords) if G is None else G
        n = np.linalg.norm(G)
        G /= n
        p -= 2 * p.ravel().dot(G.ravel()) * G


    def new_point(self, Ecut, coords, nsteps=None):

        niter = 0
        ntries = 0

        nsteps = self.nsteps if nsteps is None else nsteps

        savecoords = coords.copy()
        newcoords = coords.copy()

        while(niter<nsteps):

            self.takestep.takeStep(newcoords)

            p = newcoords - savecoords

            constraint = self.constraint(newcoords)

            if constraint:
                Enew = self.pot.getEnergy(newcoords)
                acceptMove = Enew < Ecut

            if acceptMove:
                niter += 1
                savecoords[:] = newcoords
            else:
                ntries += 1
                self.reflect(newcoords, p, self.pot)
                newcoords[:] = savecoords + p

            try:
                self.takestep.updateStep(acceptMove)
            except AttributeError:
                pass

            if ntries >= self.maxtries:
                raise SamplingError("MC Sampler failure",
                                    E=Ecut, ntries=ntries, niter=niter)

        self.niter += 1

        return newcoords, Enew, niter, ntries

class MCSampler(BaseSampler):

    def __init__(self, pot, takestep=AdaptiveStepsize(TakestepHyperSphere()),
                 nsteps=100, maxtries=1000, debug=False,
                 constraint=None):

        self.pot = pot
        self.niter = 0
        self.nfev = 0
        self.nsteps = nsteps
        self.maxtries = maxtries
        self.takestep = takestep
        self.constraint = BaseConstraint() if constraint is None else constraint
        self.debug = debug

        self.pstr = "found energy = {:10.12g} after {} steps and {} tries"

    def new_point(self, Ecut, coords, nsteps=None):

        niter = 0
        ntries = 0

        nsteps = self.nsteps if nsteps is None else nsteps

        savecoords = coords.copy()
        newcoords = coords.copy()

        while(niter<nsteps):

            self.takestep.takeStep(newcoords)

            acceptMove = self.constraint(newcoords)

            if acceptMove:
                Enew = self.pot.getEnergy(newcoords)
                acceptMove = Enew < Ecut

            if acceptMove:
                niter += 1
                savecoords[:] = newcoords
            else:
                ntries += 1
                newcoords[:] = savecoords

            try:
                self.takestep.updateStep(acceptMove)
            except AttributeError:
                pass

            if ntries >= self.maxtries:
                raise SamplingError("MC Sampler failure",
                                    E=Ecut, ntries=ntries, niter=niter)

        self.niter += 1

        return newcoords, Enew, niter, ntries



class GalileanSampler(BaseSampler):

    pstr = "HMC  > nsteps  ={:4d}, nreject ={:4d}, E    ={:10.5g}, stepsize ={:10.5g}"

    def __init__(self, pot, genstep=hypersphere_step, constraint=None,
                 nsteps=100, stepsize=0.1, acc_ratio=0.5, factor=0.2,
                 minstep=1e-5, maxstep=1.0, maxreject=1000, adaptive=True,
                 debug=False, searchratio=1.5, maxdepth=10,
                 armijo=False, armijo_c1=1e-4, armijo_c2=0.9, armijo_s=2):

        self.pot = pot
        self.constraint = BaseConstraint() if constraint is None else constraint

        self.niter = 0

        self.nsteps = nsteps
        self.maxreject = maxreject
        self.maxdepth = maxdepth

        self.armijo = armijo
        self.armijo_c1 = armijo_c1
        self.armijo_c2 = armijo_c2
        self.armijo_s = armijo_s
        self.armijo_step = stepsize

        self.gen_p = genstep

        self.stepsize = stepsize
        self.minstep = minstep
        self.maxstep = maxstep
        self.acc_ratio = acc_ratio
        self.adaptive = adaptive
        self.factor = factor

        self.searchratio = searchratio

        self.debug = debug

    def adjust_stepsize(self, nsteps, nreject, newcoords):
        curr_rate = np.clip((nsteps - nreject + 1.)/(nsteps + 1.),0.,1.)
        self.stepsize *= np.exp(self.factor*(curr_rate-self.acc_ratio))

    def armijo_condition(self, coords, p, step, E, wolfe1, wolfe2):

        newcoords = coords + step*p
        if wolfe2 is None:
            newE = self.pot.getEnergy(newcoords)
            cond = (newE <= E + step*wolfe1)
        else:
            newE, newG = self.pot.getEnergyGradient(newcoords)
            cond = ((newE <= E + step*wolfe1) and (- p.dot(newG) <= wolfe2))
        return cond, newcoords

    def find_stepsize_armijo(self, coords, stepsize=None, energy=None, gradient=None):

        stepsize = self.armijo_step if stepsize is None else stepsize
        E = self.pot.getEnergy(coords) if energy is None else energy
        G = self.pot.getGradient(coords) if gradient is None else gradient
        nG = np.linalg.norm(G)

        assert np.isfinite(G).all()

        p = - G / nG
        pG = p.dot(G)

        wolfe1 = self.armijo_c1 * pG
        wolfe2 = - self.armijo_c2 * pG if self.armijo_c2 is not None else None

        cond, newcoords = self.armijo_condition(coords, p, stepsize,
                                                E, wolfe1, wolfe2)
        cond = cond and self.constraint(newcoords)

        if cond:
            while cond:
                stepsize *= self.armijo_s
                cond, newcoords = self.armijo_condition(coords, p, stepsize,
                                                        E, wolfe1, wolfe2)
                cond = cond and self.constraint(newcoords)
        else:
            while not(cond):
                stepsize /= self.armijo_s
                cond, newcoords = self.armijo_condition(coords, p, stepsize,
                                                        E, wolfe1, wolfe2)
                cond = cond and self.constraint(newcoords)

        self.armijo_step = stepsize
        return stepsize

    def find_stepsize(self, coords, stepsize=None, energy=None, gradient=None):

        coords = np.array(coords) # Making copy of coords
        stepsize = self.stepsize if stepsize is None else stepsize
        E = self.pot.getEnergy(coords) if energy is None else energy
        G = self.pot.getGradient(coords) if gradient is None else gradient
        G /= np.linalg.norm(G)

        assert np.isfinite(G).all()

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

        Econ = self.constraint.getEnergy(coords - stepsize*G)
        if Econ > 0.:
            while(Econ > 0.):
                stepsize /= self.searchratio
                Econ = self.constraint.getEnergy(coords - stepsize*G)
                if stepsize < self.minstep:
                    stepsize = self.minstep
                    break

        stepsize /= self.searchratio

        self.stepsize

        return stepsize

    def reflect_point(self, pos, p, pot, Ecut, newpos=None, energy=None):
        """
        Applies hard shell constraint to point at position pos with momentum p

        Parameters
        ----------
        pos : numpy.array
            starting position
        p : numpy.array
            initial momentum, must be same shape as pos and is modified
        pot : Potential
            needs getEnergy and getGradient methods
        Ecut : float
            Energy contour hard shell constraint
        newpos : numpy.array
            pos + p if already calculated
        energy : float
            pot.getEnergy(pos+p) if already calculated

        Returns
        -------
        newpos : numpy.array
            new position
        p : numpy.array
            new momentum
        E : float
            energy of newpos
        G : numpy.array or None
            gradient at newpos if calculated, else None
        """
        newpos = pos + p if newpos is None else newpos
        E = self.pot.getEnergy(newpos) if energy is None else energy
        G = None

        while(E > Ecut):
            G = self.pot.getGradient(newpos)
            G /= np.linalg.norm(G)
            p -= 2 * p.ravel().dot(G.ravel()) * G
            newpos[:] = pos + p
            E = self.pot.getEnergy(newpos)

        return newpos, p, E, G

    def reflect(self, coords, p, pot, G=None):
        """
        Applies hard shell constraint to momentum vector p

        Parameters
        ----------
        pos : numpy.array
            starting position
        p : numpy.array
            initial momentum, must be same shape as pos and is modified
        pot : Potential
            needs getEnergy and getGradient methods
        G : numpy.array, optional
            gradient at coords
        """
        G = pot.getGradient(coords) if G is None else G
        n = np.linalg.norm(G)
        G /= n
        p -= 2 * p.ravel().dot(G.ravel()) * G

    def new_point(self, Ecut, coords, nsteps=None, stepsize=None, depth=0):

        nsteps = self.nsteps if nsteps is None else nsteps

        if stepsize is None:
            if self.armijo:
                stepsize = self.find_stepsize_armijo(coords)
            else:
                stepsize = self.find_stepsize(coords)

        nreject = 0

        newcoords = np.array(coords)
        p = self.gen_p(newcoords, stepsize)
        testcoords = np.empty_like(newcoords)

        for i in xrange(nsteps):

            assert np.isfinite(p).all()

            testcoords[:] = newcoords +  p

            Econ, Gcon = self.constraint.getEnergyGradient(testcoords)
            Enew, Gnew = self.pot.getEnergyGradient(testcoords)

            constraint = Econ <= 0
            Eaccept = Enew <= Ecut


            while(not constraint or not Eaccept):

                assert np.isfinite(p).all()

                nreject += 1

                if not constraint:
                    self.reflect(testcoords, p, self.constraint, Gcon)
                    assert np.isfinite(p).all()
                    #self.reflect(testcoords, p, self.pot)

                    testcoords[:] = newcoords +  p
                    Econ, Gcon = self.constraint.getEnergyGradient(testcoords)

                    constraint = Econ <= 0
                    if constraint:
                        Enew, Gnew = self.pot.getEnergyGradient(testcoords)
                        Eaccept = Enew <= Ecut

                if not Eaccept:
                    self.reflect(testcoords, p, self.pot, Gnew)
                    assert np.isfinite(p).all()

                    testcoords[:] = newcoords +  p
                    Econ, Gcon = self.constraint.getEnergyGradient(testcoords)
                    Enew, Gnew = self.pot.getEnergyGradient(testcoords)
                    constraint = Econ <= 0
                    Eaccept = Enew <= Ecut

                if nreject > self.maxreject:

                    if self.debug:
                        print ('HMC  > failed to find point', stepsize,
                               stepsize/self.searchratio, Ecut, i, depth)

                    if depth > self.maxdepth:
                        print ('HMC  > failed to find point',
                               stepsize, stepsize/self.searchratio, Ecut, i)
                        raise SamplingError("max depth", stepsize,
                                            stepsize/self.searchratio, Ecut, i)

                    raise SamplingError()

                    if self.armijo_step:
                        stepsize /= self.armijo_s
                        self.armijo_step = stepsize
                    else:
                        stepsize = self.find_stepsize(coords,
                                                      stepsize/self.searchratio,
                                                      energy=Ecut)
                        print "HMC > failed", stepsize, Ecut, i
                        stepsize = self.find_stepsize_armijo(coords)
                        print "HMC > trying", stepsize, Ecut, i

                    return self.new_point(Ecut, coords, nsteps,
                                          stepsize, depth=depth+1)

            newcoords[:] = testcoords

        if self.debug:
            print self.pstr.format(nsteps, nreject, Enew, stepsize)

        self.niter += 1

        return newcoords, Enew, nsteps, nreject

class GMCSampler(GalileanSampler):
    pstr = "HMC  > nsteps  ={:4d}, nreject ={:4d}, E    ={:10.5g}, stepsize ={:10.5g}"
    testinitial = True

    def new_point(self, Ecut, coords, nsteps=None, stepsize=None, depth=0):

        nsteps = self.nsteps if nsteps is None else nsteps

        if stepsize is None:
            if self.armijo:
                stepsize = self.find_stepsize_armijo(coords)
            else:
                stepsize = self.find_stepsize(coords)

        naccept = 0
        ncon = 0
        nE = 0
        nEcon = 0
        nreject = 0

        newcoords = np.array(coords)

        if self.testinitial:
            Estart = self.pot.getEnergy(newcoords)
            Econ = self.constraint.getEnergy(newcoords)
            if Estart > Ecut:
                raise SamplingError(
                    "Starting energy higher than cutoff",
                    Estart=Estart, Ecut=Ecut)
            if Econ > 0:
                raise SamplingError(
                    "Starting configuration doesn't satisfy constraint",
                    Estart=Estart, Econstraint=Econ)

        p = self.gen_p(newcoords, stepsize)
        testcoords = np.empty_like(newcoords)

        while(naccept < nsteps):

            assert np.isfinite(p).all()

            testcoords[:] = newcoords +  p

            Econ, Gcon = self.constraint.getEnergyGradient(testcoords)
            Enew, Gnew = self.pot.getEnergyGradient(testcoords)

            constraint = Econ <= 0
            Eaccept = Enew <= Ecut

            if Eaccept and constraint:
                newcoords[:] = testcoords
                naccept += 1
            else:
                nreject += 1

                if Eaccept and not constraint:
                    n = Gcon/np.linalg.norm(Gcon)
                    testcoords += p - 2. * n * n.ravel().dot(p.ravel())
                    ncon += 1
                elif constraint and not Eaccept:
                    n = Gnew/np.linalg.norm(Gnew)
                    testcoords += p - 2. * n * n.ravel().dot(p.ravel())
                    nE += 1
                else:
                    n1 = Gnew/np.linalg.norm(Gnew)
                    n2 = Gcon/np.linalg.norm(Gcon)
                    n = n1 + n2
                    n /= np.linalg.norm(n)
                    testcoords += p - 2. * n * n.ravel().dot(p.ravel())
                    nEcon += 1

                Econ = self.constraint.getEnergy(testcoords)
                Enew = self.pot.getEnergy(testcoords)

                constraint = Econ <= 0
                Eaccept = Enew <= Ecut
                if Eaccept and constraint:
                    newcoords[:] = testcoords
                    naccept += 1
                else:
                    p = self.gen_p(newcoords, stepsize)

                if nreject > self.maxreject:
                    raise SamplingError('HMC  > failed to find point',
                                        stepsize=stepsize, Ecut=Ecut,
                                        naccept=naccept, nreject=nreject)

        return newcoords, Enew, nsteps, nreject

class ThermalSampler(object):

    pstr = "MC   > nsteps = {:4d} trial E = {:10.12g} trial J = {:10.12g} accept = {:s}"

    def __init__(self, pot, takeStep=None, stepsize=1.0, temperature=1.0,
                 Ecutoff = None, constraint=None, iprint=-1, ntest=1,
                 debug=False, event_after_step=None, acceptTest=None,
                 random=None, storage=None):

        self.pot = pot
        self.stepsize = stepsize
        self.takeStep = TakestepHyperSphere(stepsize) if takeStep is None else takeStep
        self.constraint = BaseConstraint() if constraint is None else constraint
        self.temperature = temperature

        self.ntest = ntest
        self.Ecutoff = Ecutoff

        self.nsteps = 0
        self.naccept = 0
        self.nreject = 0

        self.iprint = iprint
        self.event_after_step = [] if event_after_step is None else event_after_step

        self.acceptTest = self._acceptTest if acceptTest is None else acceptTest
        self.random = np.random.rand if random is None else random

        self.storage = None

        self.result = Result()
        self.result.nfev = 0
        self.result.energy = [np.inf]

    def _acceptTest(self, markovEs, trialEs, coords, trialcoords):

        # Coercing input into list
        try:
            iter(trialEs)
        except TypeError:
            trialEs = [trialEs]
        else:
            trialEs = list(trialEs)

        while len(trialEs) < self.ntest:
            trialEs.append(self.pot.getEnergy(trialcoords))
            self.result.nfev += 1

        if self.Ecutoff is not None:
            testEs = [max((0,E-self.Ecutoff)) for E in trialEs]
        else:
            testEs = trialEs

        self.trialJ = np.exp(-np.array(testEs)/self.temperature).mean()
        acceptstep = self.random() < self.trialJ

        return acceptstep, markovEs, trialEs

    def initialise(self, coords):
        self.coords = np.array(coords)
        self.markovEs = []

    def _mcStep(self):

        self.trial_coords = self.coords.copy()

        self.takeStep.takeStep(self.trial_coords, driver=self)

        self.trialEs = []

        self.acceptstep = self.constraint(self.trial_coords)

        if self.acceptstep:
            self.acceptstep, self.markovEs, self.trialEs = self.acceptTest(
                self.markovEs, self.trialEs, self.coords, self.trial_coords)

        return self.acceptstep, self.trial_coords, self.trialEs

    def takeOneStep(self):

        self.nsteps += 1
        self.markovE_olds = self.markovEs

        acceptstep, newcoords, newEs = self._mcStep()
        self.printStep()

        if self.storage and (self.insert_rejected or acceptstep) and self.config_ok:
            self.storage(newEs, newcoords)

        if acceptstep:
            self.coords = newcoords
            self.markovEs = newEs
            self.naccept += 1

            if np.mean(self.markovEs) < np.mean(self.result.energy):
                self.result.energy = self.markovEs
                self.result.coords = self.coords.copy()
        else:
            self.nreject += 1

        self.takeStep.updateStep(acceptstep, driver=self)
        for event in self.event_after_step:
            event(self.markovEs, self.coords, acceptstep)

    def printStep(self):
        if self.nsteps % self.iprint == 0:
            print self.pstr.format(
                self.nsteps, np.mean(self.trialEs),
                self.trialJ, str(self.acceptstep))

class AdaptiveThermalSampler(ThermalSampler):

    sTstr = "MC   > n = {:4d} E = {:8.5g} J = {:8.3g} T = {:8.3g} accept = {:5s} s = {:8.3f} new = {:5s}"

    def __init__(self, pot, takeStep=None, stepsize=1.0, temperature=1.0,
                 Ecutoff = None, constraint=None, iprint=-1, ntest=1,
                 debug=False, event_after_step=None, acceptTest=None,
                 random=None, storage=None, interval=100,
                 target_new_min_prob=0.8, target_new_min_accept_prob=0.3,
                 sfactor=1.1, Tfactor=1.1, ediff=0.001):

        super(self.__class__, self).__init__(
            pot, takeStep=takeStep, stepsize=stepsize, temperature=temperature,
            Ecutoff = Ecutoff, constraint=constraint, iprint=iprint,
            ntest=ntest, debug=debug, event_after_step=event_after_step,
            acceptTest=acceptTest, random=random, storage=storage)

        self.nsame = 0
        self.nnew = 0

        self.interval = interval
        self.target_new_min_prob = target_new_min_prob
        self.target_new_min_accept_prob = target_new_min_accept_prob
        self.sfactor = sfactor
        self.Tfactor = Tfactor
        self.ediff = ediff

        self.sf = sfactor**(1./self.interval)
        self.Tf = Tfactor**(1./self.interval)
        self.new_min_factor = self.sf ** (self.target_new_min_prob-1.)
        self.same_min_factor = self.sf ** (self.target_new_min_prob)
        self.accept_min_factor = self.Tf ** (self.target_new_min_accept_prob-1.)
        self.reject_min_factor = self.Tf ** (self.target_new_min_accept_prob)

    def compareEs(self, newMarkovEs, oldMarkovEs):

        newE = np.mean(newMarkovEs)
        newstd = np.std(newMarkovEs)
        oldE = np.mean(oldMarkovEs)
        oldstd = np.std(oldMarkovEs)

        std = 0 #np.sqrt((newstd**2+oldstd**2)/(len(newMarkovEs)+len(oldMarkovEs)))
        newmin = abs(newE-oldE) > self.ediff + std
        return newmin

    def updateStep(self, newmin):
        if newmin:
            self.takeStep.stepsize *= self.new_min_factor
        else:
            self.takeStep.stepsize *= self.same_min_factor
        self.stepsize = self.takeStep.stepsize

    def updateTemperature(self, accept):
        if accept:
            self.temperature *= self.accept_min_factor
        else:
            self.temperature *= self.reject_min_factor

    def update(self, newmin, accept):
        self.updateStep(newmin)
        if newmin:
            self.updateTemperature(accept)

    def takeOneStep(self):

        self.nsteps += 1
        self.markovEs_old = self.markovEs

        acceptstep, newcoords, newEs = self._mcStep()
        self.newmin = self.compareEs(newEs, self.markovEs_old)

        if self.newmin:
            self.nnew += 1
        else:
            self.nsame += 1

        self.printStep()

        if self.storage and (self.insert_rejected or acceptstep) and self.config_ok:
            self.storage(newEs, newcoords)

        if acceptstep:
            self.coords = newcoords
            self.markovEs = newEs
            if self.newmin:
                self.naccept += 1

            if np.mean(self.markovEs) < np.mean(self.result.energy):
                self.result.energy = self.markovEs
                self.result.coords = self.coords.copy()
        else:
            self.nreject += 1

        self.update(self.newmin, acceptstep)

        for event in self.event_after_step:
            event(self.markovEs, self.coords, acceptstep)

    def printStep(self):
        if self.nsteps % self.iprint == 0:
            print self.sTstr.format(
                self.nsteps, np.mean(self.trialEs), self.trialJ,
                self.temperature, str(self.acceptstep),
                self.stepsize, str(self.newmin), )

if __name__ == "__main__" and False:

    import matplotlib.pyplot as plt
    from pele.potentials import BasePotential

    from nestedbasinsampling.takestep import TakestepHyperSphere
    from nestedbasinsampling.random import vector_random_uniform_hypersphere
    from nestedbasinsampling.utils import hyperspherevol

    class MyPot(BasePotential):

        def getEnergy(self, x):
            x = np.asanyarray(x)
            self.dim = x.size
            E = 0.5*(x**2).sum()
            return E

        def getGradient(self, x):
            return x

        def getEnergyGradient(self, x):
            return self.getEnergy(x), self.getGradient(x)

        def getVol(self, Vs):
            Vs = np.atleast_1d(Vs)
            vols = np.zeros_like(Vs)
            ltz =  0. > Vs
            Vs[ltz] = 0.
            nz = np.logical_not(ltz)
            r = np.sqrt(2*Vs)
            vols[nz] = hyperspherevol(self.dim, r)
            return vols

    pot = MyPot()

    R = 1.
    ndim = 200

    vol = hyperspherevol(ndim, R)

    rand_config = lambda : R*vector_random_uniform_hypersphere(ndim)

    N = 50000
    coords = np.zeros(ndim)
    coords[0] = 1.
    E = 0.5

    gsampler = GMCSampler(pot, stepsize=0.3)
    samples = []
    Es = []
    new = np.zeros(ndim)+0.01
    for _ in xrange(N):
        new, Enew = gsampler.new_point(E, new, nsteps=1, stepsize=0.3)[:2]
        Es.append(Enew)
        samples.append(new)

    argsort = np.argsort(Es)
    gsamples = np.array(samples)[argsort]
    grs = np.linalg.norm(gsamples,axis=1)
    gEs = np.array(Es)[argsort]


    mcsampler = MCSampler(pot, takestep=TakestepHyperSphere(.05))
    samples = []
    Es = []
    new = np.zeros(ndim)+0.01
    for _ in xrange(N):
        new, Enew = mcsampler.new_point(E, new, nsteps=1)[:2]
        Es.append(Enew)
        samples.append(new)

    argsort = np.argsort(Es)
    mcsamples = np.array(samples)[argsort]
    mcrs = np.linalg.norm(mcsamples,axis=1)
    mcEs = np.array(Es)[argsort]

    plt.plot(grs**ndim - np.linspace(0,1,N,False))
    plt.plot(mcrs**ndim - np.linspace(0,1,N,False))

    ndim = 2
    vol = hyperspherevol(ndim, R)

    rand_config = lambda : R*vector_random_uniform_hypersphere(ndim)

    N = 50000
    coords = np.zeros(ndim)
    coords[0] = 1.
    E = 0.5

    gsampler = GMCSampler(pot, stepsize=0.3)
    samples = []
    Es = []
    new = np.zeros(ndim)+0.01
    for i in xrange(N):
        new, Enew = gsampler.new_point(E, new, nsteps=1, stepsize=0.3)[:2]
        Es.append(Enew)
        samples.append(new)

    argsort = np.argsort(Es)
    gsamples = np.array(samples)[argsort]
    grs = np.linalg.norm(gsamples,axis=1)
    gEs = np.array(Es)[argsort]

    plt.plot(grs**ndim - np.linspace(0,1,N,False))
    plt.plot(mcrs**ndim - np.linspace(0,1,N,False))

    mcsampler = MCSampler(pot, takestep=TakestepHyperSphere(.05))
    samples = []
    Es = []
    new = np.zeros(ndim)+0.01
    for i in xrange(N):
        new, Enew = mcsampler.new_point(E, new, nsteps=1)[:2]
        Es.append(Enew)
        samples.append(new)

    argsort = np.argsort(Es)
    mcsamples = np.array(samples)[argsort]
    mcrs = np.linalg.norm(mcsamples,axis=1)
    mcEs = np.array(Es)[argsort]

    plt.plot(grs**ndim - np.linspace(0,1,N,False))
    plt.plot(mcrs**ndim - np.linspace(0,1,N,False))


if __name__ == "__main__":
    from pele.systems import LJCluster
    from pele.optimize import lbfgs_cpp
    from nestedbasinsampling.constraints import HardShellConstraint
    from nestedbasinsampling.random import random_structure
    from nestedbasinsampling.nestedoptimization import AdaptiveNestedOptimizer

    import matplotlib.pyplot as plt
    from plottingfuncs.plotting import ax3d

    natoms = 31
    radius =  float(natoms) ** (1. / 3)
    system = LJCluster(natoms)
    rand_config = lambda : random_structure(natoms, radius)

    constraint = HardShellConstraint(radius)

    pot = system.get_potential()

    coords = rand_config()
    Ecut = pot.getEnergy(coords);Ecut

    coords = np.array([-0.50644146,  0.00760803,  1.0326462 , -0.00821804, -0.39270969,
        0.32737261, -0.71055378, -0.28714936, -0.50838261,  0.70903135,
       -0.44342951,  1.04899041, -0.92034014,  0.82328437,  0.97173452,
        0.33184024,  0.57969122, -1.37499424, -1.48853395,  0.22759426,
       -0.62954964, -1.0236435 ,  1.25057781,  0.06162295,  0.15295476,
       -0.85302021, -0.85181103,  0.11688046, -1.66710077, -0.02708454,
        0.92162318, -1.15964557,  0.58375724,  1.08896457, -0.28900972,
       -0.40154893, -0.80763631,  0.61597603, -0.84021609,  1.47042294,
       -0.48903092,  0.61555208,  1.46126053,  0.50626641, -0.78150704,
       -0.1576444 , -0.11719317, -1.56205682,  0.23267384,  0.19161563,
       -0.44888349, -1.79242242, -0.62104211, -0.54623554, -0.18795092,
       -1.37098707,  1.26988498,  1.28043729,  0.30747032,  1.20031504,
       -1.50541095,  0.03522178,  0.31342605, -0.2249095 ,  1.35080268,
        0.47983218, -0.34904468,  0.35903654,  0.18884556,  0.34373549,
        0.69641944,  1.12667089,  0.86273415, -0.38793287, -1.65112064,
        0.09375902,  1.22553986, -0.26792141, -1.03598805, -0.95267764,
        0.47834146,  1.75515771,  0.65608507,  0.4158042 ,  0.92622997,
        1.32432563,  0.72034649, -0.95920156, -1.39338951, -0.66670458,
        0.77803577,  0.25376173,  0.213823  ])

    gmc = GMCSampler(pot, stepsize=0.2, nsteps=20, constraint=constraint)
    opt = AdaptiveNestedOptimizer(coords, pot, gmc, stepsize=0.5,
                                  nsteps=3000, iprint=100, tol=5e-1)
    res = opt.run()

    from nestedbasinsampling.nestedoptimization import NestedGalileanOptimizer

    opt = NestedGalileanOptimizer(coords, pot, tol=5e-1, stepsize=0.5, iprint=100, constraint=constraint)
    opt.run()

    ncoords = res.coords + random_structure(natoms, 0.25)
    opt = AdaptiveNestedOptimizer(ncoords, pot, gmc, stepsize=0.5,
                                  nsteps=3000, iprint=100, tol=5e-1)
    res = opt.run()














