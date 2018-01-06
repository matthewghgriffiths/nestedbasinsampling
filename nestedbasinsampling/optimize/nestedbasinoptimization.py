# -*- coding: utf-8 -*-

from collections import defaultdict
from math import log

import numpy as np
from pele.optimize import lbfgs_cpp

from nestedbasinsampling.sampling import SamplingError
from nestedbasinsampling.sampling.takestep import vector_random_uniform_hypersphere
from nestedbasinsampling.utils import (
    Result, NestedSamplingError, dict_update_keep, SortedCollection)

class NestedBasinOptimizer(object):
    """
    """
    pstr = ("NBOPT> niters ={:6d}, E ={:10.5g}, "
            "rms ={:10.5g}, stepsize ={:8.4g}, logF = {:8.4g}")
    def __init__(self, minimum, pot, sampler, nlive=1, energy=np.inf,
                 tol=1e-1, alternate_stop_criterion=None, stepsize=None,
                 events=None, iprint=-1, nsteps=10000, debug=False,
                 MC_steps=None, nadapt=None, target=None, store_configs=False,
                 harmonic_start=True, harmonic_tries=20, harmonic_factor=2.):

        self.minimum = minimum
        self.energy = self.minimum.energy

        self.pot = pot
        self.sampler = sampler
        self.stepsize = stepsize

        self.set_coords()

        self.events = [] if events is None else events
        self.iprint = iprint
        self.nsteps = nsteps
        self.nadapt = nadapt
        self.MC_steps = MC_steps
        self.tol = tol
        self.target = target

        self.harmonic_start = harmonic_start
        self.harmonic_tries = harmonic_tries
        self.harmonic_factor = harmonic_factor

        if self.harmonic_start:
            self.set_harmonic()

        self.store_configs = store_configs


        self.alternate_stop_criterion = alternate_stop_criterion
        self.debug = debug  # print debug messages

        self.E = energy
        self.G = np.inf + np.empty_like(self.coords)
        self.rms = np.inf

        self.iter_number = 0
        self.nopt = 0
        self.tot_rejects = 0
        self.tot_steps = 0
        self.accept = defaultdict(int)
        self.result = Result()
        self.result.message = []

        self.Es = []
        self.nlives = []
        self.stepsizes = []
        self.configs = []

        self.nlive = nlive
        self.logF = 0.
        self.live_points = SortedCollection([], key=lambda res: res.energy)

    def set_harmonic(self):
        self.H = self.pot.getHessian(self.minimum.coords)
        self.u, self.v = np.linalg.eigh(self.H)
        p = self.u > 1e-5
        self.k = p.sum()

        self.up = self.u[p]
        self.vp = self.v[:,p]
        self.up2 = (self.up/2.)**-0.5
        self.Ef = self.harmonic_factor**(1./self.k)

    def random_harmonic_coords(self, E):
        fac = self.Ef * (E-self.minimum.energy)**0.5
        x = vector_random_uniform_hypersphere(self.k) * fac
        newcoords = self.vp.dot(self.up2 * x) + self.minimum.coords
        Enew = self.pot.getEnergy(newcoords)
        return Enew, newcoords

    def set_coords(self):
        self.coords = self.minimum.coords
        if not self.sampler.constraint(self.coords):
            self.coords = self.sampler.fixConstraint(self.coords)

    def initialize(self):
        while len(self.live_points) < self.nlive:
            res = self.sample_point()
            self.live_points.insert(res)

    def sample_point(self):
        if self.harmonic_start:
            for i in xrange(self.harmonic_tries):
                E, coords = self.random_harmonic_coords(self.E)
                if E < self.E:
                    if self.sampler.constraint(coords):
                        break
                    else:
                        try:
                            coords = self.sampler.fixConstraint(coords)
                            break
                        except NotImplementedError:
                            pass
            else:
                coords = self.coords
        else:
            coords = self.coords

        res = self.sampler(self.E, coords, nsteps=self.MC_steps,
                           nadapt=self.nadapt, stepsize=self.stepsize)
        return res

    def one_iteration(self):

        self.nlive = len(self.live_points)

        deadpoint = self.live_points.pop()
        self.E = deadpoint.energy
        self.G = deadpoint.grad
        self.rms = np.linalg.norm(self.G)/np.sqrt(self.G.size)
        self.stepsize = deadpoint.stepsize
        self.logF += log(self.nlive) - log(self.nlive+1)

        self.Es.append(self.E)
        self.nlives.append(self.nlive)
        self.stepsizes.append(self.stepsize)
        if self.store_configs:
            self.configs.append(deadpoint.coords)


        res = self.sample_point()

        self.live_points.insert(res)
        self.printState(False)

        for event in self.events:
            event(coords=self.X, energy=self.E, res=res,
                  rms=self.rms, stepsize=self.stepsize)

        self.iter_number += 1

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

    def printState(self, force=True):
        cond = (self.iprint > 0 and self.iter_number%self.iprint == 0) or force
        if cond:
            print self.pstr.format(
                self.iter_number, self.E, self.rms, self.stepsize, self.logF)

    def run(self):
        self.initialize()
        while self.iter_number < self.nsteps and not self.stop_criterion():
            try:
                self.one_iteration()
            except NestedSamplingError:
                self.result.message.append("problem with nested sampler")
                break

        if self.debug or self.iprint > 0:
            self.printState(True)

        return self.get_result()

    def get_result(self):
        res = self.result
        res.coords = self.coords
        res.energy = self.energy
        res.rms = self.rms
        res.grad = self.G
        res.energies = np.array(self.Es)
        res.Emax = res.energies
        res.nlive = self.nlives
        res.stepsize = np.array(self.stepsizes)
        if self.store_configs:
            res.configs = np.array(self.configs)
        res.nsteps = self.nopt
        res.success = self.stop_criterion()
        return res

if __name__ == '__main__':

    from scipy.special import betaln

    def logBetaEvidence(a1, b1, a2, b2, pa=0.5, pb=0.5):
        logSame = (betaln(a1+a2+2*pa, b1+ b2 +2*pb)
                   - betaln(2*pa,2*pb))
        logDiff = (betaln(a1 + pa, b1 + pb) +
                   betaln(a2 + pa, b2 + pb) - 2*betaln(pa,pb) )
        return logSame, logDiff

    def matchRuns(runs, log_step=1., log_Xmin=None, Emin=None):
        """
        """
        run = combineAllRuns(runs)
        logX = run.log_frac
        rlogX = [r.log_frac for r in runs]
        rlogX2 = [r.log_frac2 for r in runs]

        if log_Xmin is None:
            Emin = max((r.Emax[-1]) for r in runs) if Emin is None else Emin
            imin = -run.Emax[::-1].searchsorted(Emin)
            log_Xmin = logX[imin]

        Lsplits = np.arange(-log_step, log_Xmin, -log_step)
        isplits = logX[::-1].searchsorted(Lsplits)
        Esplits = run.Emax[-isplits]

        risplits = [-r.Emax[::-1].searchsorted(Esplits) for r in runs]
        risplits0 = [np.r_[0,i[:-1]] for i in risplits]

        dlogX = np.array(
            [lX[i1] - lX[i0] for i1, i0, lX in izip(risplits, risplits0, rlogX)])
        dlogX2 = np.array(
            [lX[i1] - lX[i0] for i1, i0, lX in izip(risplits, risplits0, rlogX2)])

        a, b = logmoment2Beta(dlogX, dlogX2)
        return Esplits, np.concatenate((a[:,:,None], b[:,:,None]), axis=2)

    def logmoment2Beta(logm1, logm2):
        """
        """
        logvar = logm2 + np.log(1 - np.exp(2*logm1 - logm2))
        logm3 = logm1 + np.log(1 - np.exp(logm2 - logm1))
        ns = np.exp(logm1 + logm3 - logvar)
        nr = np.exp(logm3 - logvar + np.log(1. - np.exp(logm1)))
        return ns, nr

    def calcRelError(r):
        logX = r.log_frac
        logX2 = r.log_frac2
        return logX, np.log1p(np.sqrt(np.exp(logX2 - 2 * logX) - 1))

    from itertools import izip

    import matplotlib.pyplot as plt
    import seaborn as sns

    from pele.systems import LJCluster
    from nestedbasinsampling.storage import Minimum, Run
    from nestedbasinsampling.sampling.galilean import GalileanSampler
    from nestedbasinsampling.structure.constraints import HardShellConstraint
    from nestedbasinsampling.nestedsampling.combine import combineAllRuns
    from nestedbasinsampling.optimize import NestedOptimizer
    from nestedbasinsampling.sampling.stats import CDF, AndersonDarling, AgglomerativeCDFClustering

    plt.ion()

    natoms = 31
    system = LJCluster(natoms)
    radius =  float(natoms) ** (1. / 3)

    with open("../lj31_10.pkl", 'rb') as f:
        import cPickle
        mincoords = cPickle.load(f)



    acc_ratio = 0.25

    pot = system.get_potential()
    constraint = HardShellConstraint(radius)
    sampler = GalileanSampler(
        pot, constraint=constraint, nsteps=30, nadapt=30,
        maxreject=100, acc_ratio=acc_ratio, fixConstraint=True)

    for i in xrange(len(mincoords)):
        mincoords[i] = sampler.fixConstraint(mincoords[i])

    minima = [Minimum(pot.getEnergy(c), c) for c in mincoords]
    for m in minima:
        m.fvib = system.get_log_product_normalmode_freq(m.coords)
        m.pgorder = system.get_pgorder(m.coords)

    samples = [sampler(-130, sampler.fixConstraint(mc)) for mc in mincoords]

if False:
    mEs = [[NestedOptimizer(
        coords, pot, sampler, nsteps=1000, energy=-70, iprint=100,
        gradient=np.inf).run().energy for _ in xrange(10)]
            for coords in mincoords]

    cdfs = [CDF(Es) for Es in mEs]

    [c.plot() for c in cdfs]

    mE2s = [[NestedOptimizer(
        coords, pot, sampler, nsteps=1000, energy=-70, iprint=100,
        gradient=np.inf).run().energy for _ in xrange(10)]
            for coords in mincoords]

    cdf2s = [CDF(Es) for Es in mE2s]

    [c.plot() for c in cdf2s]


if False:
    m = Minimum()

if False:

    m = Minimum(-134, mincoords[1])
    res0 = NestedBasinOptimizer(
        m, pot, sampler, nlive=50, nsteps=2000, iprint=100,
        energy=-130).run()

    run0 = Run(res0.energies, res0.nlive).split(-70,-140)
    l0, ls0 = calcRelError(run0)
    plt.fill_between(run0.Emax, l0 + ls0, l0 - ls0, alpha=0.5, color='b')
    plt.plot(run0.Emax, run0.log_frac)

    m = Minimum(-132, mincoords[7])
    res8 = NestedBasinOptimizer(
        m, pot, sampler, nlive=50, nsteps=2000, iprint=100,
        energy=-128).run()
    run1 = Run(res8.energies, res8.nlive).split(-70,-140)
    #run0 = runs[0]
    #run1 = runs[8]
    l1, ls1 = calcRelError(run1)
    plt.fill_between(run1.Emax, l1 + ls1, l1 - ls1, alpha=0.5, color='g')
    plt.plot(run0.Emax, run0.log_frac); plt.plot(run1.Emax, run1.log_frac)

    results = [NestedOptimizer(
        mincoords[1], pot, sampler, nsteps=1000, energy=-130, iprint=100,
        gradient=np.inf, target=-134, use_quench=False).run() for _ in xrange(100)]
    run12 = combineAllRuns([Run(res.Emax, np.ones_like(res.Emax)).split(-130,-135) for res in results])
    results = [NestedOptimizer(
        mincoords[7], pot, sampler, nsteps=1000, energy=-128, iprint=100,
        gradient=np.inf, target=-134, use_quench=False).run() for _ in xrange(100)]
    run7 = combineAllRuns([Run(res.Emax, np.ones_like(res.Emax)).split(-70,-135) for res in results])

    plt.plot(run7.Emax, run7.log_frac); plt.plot(run12.Emax, run12.log_frac)

if False:
    i0 = 3
    i1 = 1
    i2 = 2
    energy = -110
    target = -120

    mE0s = [NestedOptimizer(
        mincoords[i0], pot, sampler, nsteps=1, energy=energy, target=target).run()
        for _ in xrange(200)]
    mE1s = [NestedOptimizer(
        mincoords[i1], pot, sampler, nsteps=1, energy=energy, target=target).run()
        for _ in xrange(200)]
    mE2s = [NestedOptimizer(
        mincoords[i2], pot, sampler, nsteps=1, energy=energy, target=target).run()
        for _ in xrange(200)]
    plt.figure()
    CDF([r.energy for r in mE0s]).plot()
    CDF([r.energy for r in mE1s]).plot()
    CDF([r.energy for r in mE2s]).plot()
    AndersonDarling.compareDistributions(
        [[r.energy for r in mE0s], [r.energy for r in mE1s], [r.energy for r in mE2s]])

    res = NestedOptimizer(mincoords[i0], pot, sampler, nsteps=500,
                          energy=np.inf, target=energy, use_quench=False).run()

    res = NestedOptimizer(mincoords[0], pot, sampler, nsteps=1,
                          energy=energy, use_quench=False).run()

    mE3s = [NestedOptimizer(
        res.coords, pot, sampler, nsteps=1, energy=energy, target=target).run()
        for _ in xrange(50)]
    CDF([r.energy for r in mE3s]).plot()

    AndersonDarling.compareDistributions([[r.energy for r in mE0s], [r.energy for r in mE1s]])

    mE0s = [NestedOptimizer(
        mincoords[i0], pot, sampler, nsteps=100, energy=energy).run() for _ in xrange(50)]
    mE1s = [NestedOptimizer(
        mincoords[i1], pot, sampler, nsteps=100, energy=energy).run() for _ in xrange(50)]
    CDF([r.energy for r in mE0s]).plot(); CDF([r.energy for r in mE1s]).plot()
    AndersonDarling.compareDistributions([[r.energy for r in mE0s], [r.energy for r in mE1s]])

    np.unique([np.round(r.energy,2) for r in mE0s]).size

    plt.figure()
    run0 = combineAllRuns([Run(res.Emax, np.ones_like(res.Emax)).split(-70,-135) for res in mE0s])
    run1 = combineAllRuns([Run(res.Emax, np.ones_like(res.Emax)).split(-70,-135) for res in mE1s])
    l0, ls0 = calcRelError(run0)
    l1, ls1 = calcRelError(run1)
    plt.fill_between(run0.Emax, l0 + ls0, l0 - ls0, alpha=0.5, color='b')
    plt.fill_between(run1.Emax, l1 + ls1, l1 - ls1, alpha=0.5, color='g')
    plt.plot(run0.Emax, run0.log_frac); plt.plot(run1.Emax, run1.log_frac)
    AndersonDarling.compareDistributions([[r.energy for r in mE0s], [r.energy for r in mE1s]])
    raise

    mE0s = [NestedOptimizer(
        mincoords[i0], pot, sampler, nsteps=1000, energy=energy).run() for _ in xrange(10)]
    mE1s = [NestedOptimizer(
        mincoords[i1], pot, sampler, nsteps=1000, energy=energy).run() for _ in xrange(10)]
    CDF([r.energy for r in mE0s]).plot(); CDF([r.energy for r in mE1s]).plot()
    AndersonDarling.compareDistributions([[r.energy for r in mE0s], [r.energy for r in mE1s]])
    l0, ls0 = calcRelError(run0)
    l1, ls1 = calcRelError(run1)
    plt.fill_between(run0.Emax, l0 + ls0, l0 - ls0, alpha=0.5, color='b')
    plt.fill_between(run1.Emax, l1 + ls1, l1 - ls1, alpha=0.5, color='g')
    plt.plot(run0.Emax, run0.log_frac); plt.plot(run1.Emax, run1.log_frac)
    AndersonDarling.compareDistributions([[r.energy for r in mE0s], [r.energy for r in mE1s]])

    res = NestedOptimizer(
        mincoords[1], pot, sampler, nsteps=1000, energy=-0, target=-120, use_quench=False).run()

    mE2s = [NestedOptimizer(
        res.coords, pot, sampler, nsteps=1000, energy=energy).run() for _ in xrange(10)]

if False:
    results = []
    runs = []

    import time

    start = time.time()
    for m in minima[:4]:
        nbopt = NestedBasinOptimizer(m, pot, sampler, nlive=100, nsteps=25000,
                                     energy=-50, debug=True, iprint=500)
        res = nbopt.run()
        _run = Run(res.energies, res.nlive).split(np.inf,-140)
        results.append(res)
        runs.append(_run)
        print (time.time() - start)/60
    end = time.time()-start
    print end

    runs=[]
    for res in results:
        runs.append(Run(res.energies, res.nlive).split(np.inf,-140))

    minEs = [pot.getEnergy(pos) for pos in mincoords]
    modes = [system.get_log_product_normalmode_freq(pos) for pos in mincoords]

    diffE = [r.Emax[-1] - E for r, E in zip(runs, minEs)]

    logVolBottom = [r.log_frac[-1] for r in runs]
    logVolHarm = [0.5*(3*natoms-6)*np.log(dE) - logL for dE, logL in zip(diffE, modes)]

    rellogVol = [lV - min(logVolHarm) for lV in logVolHarm]

    colors = sns.color_palette(n_colors=10)
    mE = min(minEs)
    dEs = np.logspace(-1,1.5,20)
    for r, m, c in zip(runs, minima, colors):
        l0, ls0 = calcRelError(r)
        diffE = r.Emax[-1] - m.energy
        lV = -0.5 * m.fvib + 0.5*(3*natoms-6)*np.log(diffE) - l0[-1]
        plt.plot(m.energy + dEs,
                 -0.5 * m.fvib + 0.5*(3*natoms-6)*np.log(dEs), color=c)
        plt.fill_between(r.Emax,
                         l0 + ls0 + lV, l0 - ls0 + lV, alpha=0.5, color=c)
        #plt.plot(r.Emax, l0 + lV, color=c)
    plt.xlim(-140,-100)

    from nestedbasinsampling.nestedsampling.integration import logsumexp, logtrapz

    k = 3*natoms - 6
    minE = min(minEs)
    dE = np.logspace(0,-10,10000)
    logVols = [r.log_frac for r in runs]
    Es = [r.Emax - minE for r in runs]
    dEs = [E[-1] * dE for E in Es]
    Es = [np.r_[E, dE[1:]] for E, dE in zip(Es, dEs)]
    logHarmVols = [
        -0.5 * m.fvib + 0.5*k*np.log(dE) - np.log(m.pgorder)
        for m, dE in zip(minima, dEs)]
    logVols = [
        np.r_[lV + lHV[0] - lV[-1], lHV[1:]]
        for lV, lHV in zip(logVols, logHarmVols)]

    Ts = np.logspace(-3,0,200)
    logZs = [
        logtrapz(-E[None,::-1]/Ts[:,None],lV[::-1],axis=1)
        for E, lV in zip(Es, logVols)]
    logE1s = [
        logtrapz(-E[None,::-1]/Ts[:,None] + np.log(E[None,::-1]),lV[::-1],axis=1)
        for E, lV in zip(Es, logVols)]
    logE2s = [
        logtrapz(-E[None,::-1]/Ts[:,None] + 2*np.log(E[None,::-1]),lV[::-1],axis=1)
        for E, lV in zip(Es, logVols)]

    lZ = logsumexp(logZs, 0)
    lE1 = logsumexp(logE1s, 0)
    lE2 = logsumexp(logE2s, 0)

    U = np.exp(lE1 - lZ) + minE
    U2 = np.exp(lE2 - lZ) + 2*minE*U - minE**2
    V = U - 0.5*k * Ts
    V2 = U2 - U**2 + V**2


    Cv = 0.5 * k + (V2 - V ** 2) * Ts ** -2


    from pele.thermodynamics import minima_to_cv

    plt.figure()
    plt.plot(Ts, Cv)
    cv = minima_to_cv(minima[:4], Ts, k)
    plt.plot(Ts, cv.Cv)


    raise Exception
        #plt.plot(np.log(m.energy + dEs - mE), -0.5 * m.fvib + 0.5*(3*natoms-6)*np.log(dEs), color=c)
        #plt.fill_between(np.log(r.Emax - mE), l0 + ls0 + lV, l0 - ls0 + lV, alpha=0.5, color=c)

    from nestedbasinsampling.nestedsampling.integration import calcRunWeights, calcRunAverageValue
    from nestedbasinsampling.thermodynamics import E2_func, E1_func, E0_func

    def E2_func(E, T=[1.], Emin=-44.):
        T = np.atleast_1d(T)
        E = np.atleast_1d(E)-Emin
        ET = (E[None,:]/T[:,None])
        return ET**2 * np.exp(-ET)

    def E1_func(E, T=[1.], Emin=-44.):
        T = np.atleast_1d(T)
        E = np.atleast_1d(E)-Emin
        ET = (E[None,:]/T[:,None])
        return ET * np.exp(-ET)

    def E0_func(E, T=[1.], Emin=-44.):
        T = np.atleast_1d(T)
        E = np.atleast_1d(E)-Emin
        ET = (E[None,:]/T[:,None])
        return np.exp(-ET)


    Ts = np.logspace(-2,0,200)

    weights = [calcRunWeights(r) for r in runs]
    N = 500
    x = np.logspace(-3*natoms,0,N)
    f = x**(1./(3*natoms-6))
    nEs = [mE + dE*f for mE, dE in zip(minEs, diffE)]

    E2 = lambda E: E2_func(E, Ts, Emin=minEs[0])
    E1 = lambda E: E1_func(E, Ts, Emin=minEs[0])
    E0 = lambda E: E0_func(E, Ts, Emin=minEs[0])

    nE0s = [np.trapz(E0(nE), x, axis=1) for nE in nEs]
    nE1s = [np.trapz(E1(nE), x, axis=1) for nE in nEs]
    nE2s = [np.trapz(E2(nE), x, axis=1) for nE in nEs]

    plt.plot(Ts, sum(nE2s)/sum(nE0s))

    E2s = [calcRunAverageValue(w, E2, False)[0] for w in weights]
    E1s = [calcRunAverageValue(w, E1, False)[0] for w in weights]
    E0s = [calcRunAverageValue(w, E0, False)[0] for w in weights]

    E1 = sum(E*np.exp(lv) for E, lv in zip(E1s, rellogVol))
    E1 += sum(E*np.exp(r.log_frac[-1] + lv) for E, lv, r in zip(nE1s, rellogVol, runs))
    E0 = sum(E*np.exp(lv) for E, lv in zip(E0s, rellogVol))
    E0 += sum(E*np.exp(r.log_frac[-1] + lv) for E, lv, r in zip(nE0s, rellogVol, runs))

    Zbasin = [E0 * np.exp(lV)]

    Cv = sum(E2s)/sum(E0s) + sum(E1s)**2/sum(E0s)**2*Ts
    plt.plot(Ts, Cv)

    nEs = [mE + 30*f for mE  in minEs]
    nE0s = [np.trapz(E0(nE), x, axis=1)*np.exp(lv) for nE, lv in zip(nEs, rellogVol)]
    nE1s = [np.trapz(E1(nE), x, axis=1)*np.exp(lv) for nE, lv in zip(nEs, rellogVol)]
    plt.plot(Ts[1:], np.diff(sum(nE1s)/sum(nE0s)*Ts)/Ts[1:])


    def calcRelError(r):
        logX = r.log_frac
        logX2 = r.log_frac2
        return logX, np.log1p(np.sqrt(np.exp(logX2 - 2 * logX) - 1))

    colors = sns.color_palette(n_colors=10)

    plt.figure()
    for r, c in zip(runs, colors):
        r = r.split(100,-140)
        l0, ls0 = calcRelError(r)
        plt.fill_between(r.Emax, l0 + ls0, l0 - ls0, alpha=0.5, color=c)

    Ematch, AB = matchRuns([r.split(-0, -140) for r in runs], log_step=6.)

    a, b = np.rollaxis(AB, -1)
    t = (a/(a+b))
    stdt = np.sqrt(t*b/(a+b)/(a+b+1))


    plt.figure()
    for t1, stdt1, c in zip(t, stdt, colors):
        plt.fill_between(Ematch, t1 + 1.*stdt1,  t1 - 1.*stdt1, alpha=0.5, color=c)


    comp = np.triu_indices(10, 1)

    logSame, logDiff = logBetaEvidence(a[comp[0]], b[comp[0]], a[comp[1]], b[comp[1]])
    BF = np.exp(logSame - logDiff)
    1. - 1./(1+BF)

    prob = 1./(1+BF)
    probmax = np.maximum.accumulate(prob, axis=1)


    plt.figure()
    plt.plot(Ematch, prob.T)

    plt.figure()
    plt.plot(Ematch, probmax.T)

    inds = set()
    clusters = [set(range(10))]
    for E, pmax in zip(Ematch, probmax.T)[:2]:
        new = set(zip(comp[0][pmax>.25], comp[1][pmax>.25]))
        print E, inds.symmetric_difference(new)
        inds.update(new)

    p_mat = np.zeros((10,10))
    p_mat[comp] = pmax
    p_mat[comp[::-1]] = pmax


    newclusters = []
    unassigned = set(clusters[0])

    basins = dict()
    current = [set(range(10))]
    for E, pmax in zip(Ematch, probmax.T):
        p_mat[comp] = pmax
        p_mat[comp[::-1]] = pmax
        newclusters = []
        for basin in current:
            unassigned = set(basin)
            while unassigned:
                i = unassigned.pop()
                c = [i]
                while unassigned:
                    pmax, j = min((p_mat[i,j], j) for j in unassigned)
                    c.append(j)
                    if p_mat[np.ix_(c,c)].max() > .2:
                        c.pop()
                        break
                    else:
                        unassigned.remove(j)
                newclusters.append(c)
        basins[E] = newclusters
        current = newclusters
    print basins





















