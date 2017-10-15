# -*- coding: utf-8 -*-

import numpy as np

from pele.optimize import lbfgs_cpp

from nestedbasinsampling.utils import dict_update_keep, SortedCollection, Result
from nestedbasinsampling.database import Minimum, Replica, Database
from nestedbasinsampling.samplers import BaseSampler, SamplingError
from nestedbasinsampling.takestep import TakestepHyperSphere, AdaptiveStepsize
from nestedbasinsampling.constraints import BaseConstraint
from nestedbasinsampling.nestedoptimization import NestedGalileanOptimizer, NestedOptimizer


class BasinPotential(object):
    """
    """
    def __init__(self, pot, quench=None, database=None):
        self.pot = pot
        self.quench = quench
        self.database = database
        self.nfev = 0
        if self.database is not None:
            self.storage = self.database.minimumRes_adder()

    def getEnergy(self, coords):
        self.res = self.quench(coords)
        basinE = self.res.energy

        if self.database is not None:
            self.min = self.storage(self.res)
        return basinE

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

        self.results= []

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

        #print trialEs
        print self.trialJ, np.exp(-(np.array(trialEs)-self.Ecutoff)/self.temperature).std()

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
            self.results.append(newEs)
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

    sTstr = "MC   > n = {:4d}  naccept = {:5d} E = {:8.5g} J = {:8.3g} T = {:8.3g} s = {:8.3f} accept = {:5s}"

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
            self.results.append(newEs)
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
                self.nsteps, self.naccept, np.mean(self.trialEs), self.trialJ,
                self.temperature, self.stepsize, str(self.acceptstep), )

_opt_kw = dict(tol=1e-1, alternate_stop_criterion=None,
               events=None, iprint=-1, nsteps=10000, logger=None,
               debug=False, sampler_kw=dict(nsteps=30,stepsize=0.1,debug=False),
               quench=lbfgs_cpp, quenchtol=1e-6,
               quench_kw=dict(nsteps=1000))

_MC_kw = dict(takestep=None, stepsize=1.0, nsteps=100,
              maxreject=1000, verbose=True, constraint=None)

class Replicas(object):
    """
    """
    def __init__(self, database):
        self.db = database
        self.replicas = self.get_replicas()
        self.mindict = self.get_mindict()

    def get_replicas(self):
        return sorted((r for r in self.db.replicas()),
                      key = lambda r: r.quench_energies)

    def get_mindict(self):
        return dict((m.id(), m) for m in self.db.minima())

    def calcWeights(self):
        pass

    def setBins(self, nlowest=10):
        pass

    def calcBasinVolumes(self):
        pass

    def calcBasinFracVolumes(self):
        pass

    def calcBasinFracVolume(self, Vs, Vlo, Vhi):
        pass

    def calcDos(self, Vs):
        pass

class NestedSamplingRun(object):
    """
    """
    def __init__(self, Vmax=[], nlive=None, volume=1., Vcut=np.inf):
        self.Vmax = Vmax
        self.nlive = [1 for V in Vmax] if nlive is None else nlive
        self.volume = volume
        self.Vcut = Vcut

    def combine(self, run):
        """
        Joins this nested sampling run with the nested sampling run passed in

        parameters
        ----------
        run : NestedSamplingRun (or derived class)
            the nested sampling run joining with the current class

        returns
        -------
        newrun : NestedSamplingRun (or derived class)
            the combined nested sampling run
        """
        Vmax1 = self.Vmax
        nlive1 = self.nlive
        Vcut1 = self.Vcut

        Vmax2 = run.Vmax
        nlive2 = run.nlive
        Vcut2 = run.Vcut

        n1, n2 = len(Vmax1), len(Vmax2)
        i1, i2 = 0, 0

        Vmaxnew = []
        nlivenew = []

        while(i1!=n1 or i2!=n2):
            V1 = Vmax1[i1] if i1 < n1 else -np.inf
            live1 = nlive1[i1] if i1 < n1 else 0
            V2 = Vmax2[i2] if i2 < n2 else -np.inf
            live2 = nlive2[i2] if i2 < n2 else 0

            if (V1 > V2):
                Vmaxnew.append(V1)
                nlive = live1
                if V1 < Vcut2:
                    nlive += live2
                nlivenew.append(nlive)
                i1 += 1
            else:
                Vmaxnew.append(V2)
                nlive = live2
                if V2 < Vcut1:
                    nlive += live1
                nlivenew.append(nlive)
                i2 += 1

        Vcut = max(Vcut1, Vcut2)

        return type(self)(Vmax=Vmaxnew, nlive=nlivenew,
                          volume=self.volume, Vcut=Vcut)

    __add__ = combine # So Nested Sampling runs can be easily added together

    def calcBasinFracVolume(self):
        nlive = np.array(self.nlive)
        self.frac = (nlive) / (nlive+1.)
        self.fracVolume = np.cumprod(self.frac)
        self.basinVolume = self.fracVolume*self.volume
        return self.fracVolume

    def calcBasinFracDoS(self, Vi, deltaV=None, err=False):
        pass

class NestedBasinSampling(object):
    """
    """
    def __init__(self, pot, constraint=None, database=None, quencher=None,
                 opt_kw=None, samplers={}, stepsize=1.0, center=False):
        self.pot = pot
        self.constraint = constraint
        self.quencher = self._nestedoptimization if quencher is None else quencher

        self.center = center # if true centers coordinates at origin

        self.samplers = samplers

        if 'MCsample' not in self.samplers:
            self._takestep = TakestepHyperSphere(stepsize=stepsize)
            self.samplers['MCsample'] = None

        self.db = Database() if database is None else database
        self.addminimum = self.db.minimumRes_adder()
        self.addReplica = self.db.replica_adder()

    def _nestedoptimization(self, coords, **kwargs):
        """ performs nested optimisation on coords"""
        dict_update_keep(kwargs, _opt_kw)
        if 'constraint' not in kwargs:
            kwargs['constraint'] = self.constraint
        self._opt = NestedGalileanOptimizer(coords, self.pot, **kwargs)
        #self._sampler = GMCSampler(pot)
        #self._opt = NestedOptimizer(coords, pot, self._sampler, **kwargs)
        self._res = self._opt.run()
        return self._res

    def get_basinpotential(self):
        return BasinPotential(self.pot, self.quencher, database=self.db)


from pele.systems import LJCluster
from pele.potentials import BasePotential
from nestedbasinsampling.constraints import HardShellConstraint
from nestedbasinsampling.random import random_structure
from nestedbasinsampling.samplers import GalileanSampler, MCSampler, GMCSampler
from nestedbasinsampling.nestedoptimization import NestedGalileanOptimizer, AdaptiveNestedOptimizer
import matplotlib.pyplot as plt
from plottingfuncs.plotting import ax3d

from nestedbasinsampling.random import vector_random_uniform_hypersphere

from scipy.signal import savgol_filter

from scipy.stats import anderson_ksamp, ks_2samp


natoms = 31
system = LJCluster(natoms)

gmin  = np.array(
      [-0.57661039, -0.20922621,  1.02839484,  0.03718986, -0.56960802,
        0.19065885, -0.74485456, -0.29376893, -0.5330444 ,  0.51589413,
       -0.38547324,  1.16293184, -0.97689163,  0.82980011,  1.01651968,
        0.44237781,  0.76453522, -1.43379792, -1.58109833,  0.37403757,
       -0.75244974, -1.17803492,  1.13214206, -0.03827416,  0.11377805,
       -0.9574589 , -0.85655503, -0.11345962, -1.63036576, -0.00340774,
        0.81480327, -1.25699092,  0.52149634,  1.03562717, -0.58539809,
       -0.33122239, -0.55082335,  0.76714013, -0.87832192,  1.51033404,
       -0.38728163,  0.64810546,  1.24889779,  0.47289277, -0.66473407,
       -0.28650994, -0.12978066, -1.5346621 ,  0.2469026 ,  0.08722791,
       -0.6274991 , -1.76578221, -0.68578055, -0.42063516, -0.17612401,
       -1.24814226,  1.04274587,  1.2046335 ,  0.48044108,  1.28486289,
       -1.37443511,  0.07223048,  0.30133618, -0.24371389,  1.48979545,
        0.48682862, -0.35623627,  0.45130852,  0.17777359,  0.11400512,
        0.65740392,  1.14976934,  0.85312099, -0.30131762, -1.42034537,
        0.37785698,  1.1214737 , -0.35276543, -0.97313091, -0.96912921,
        0.31447945,  1.72926618,  0.67038536,  0.31580514,  0.85445676,
        1.31467926,  0.62662403, -0.91492618, -1.35463763, -0.73063174,
        0.71348693,  0.27886602,  0.31001412])

coords = np.array(
      [-0.58571338, -0.21208698,  1.03215387,  0.03530758, -0.58065593,
        0.17849478, -0.74055899, -0.29344186, -0.53910233,  0.51320899,
       -0.40068605,  1.16340898, -0.96339651,  0.83163157,  1.01804934,
        0.4415748 ,  0.75506391, -1.42538059, -1.56786152,  0.38222732,
       -0.75483186, -1.17597642,  1.11611787, -0.0484855 ,  0.10792627,
       -0.95807091, -0.87880151, -0.09598989, -1.64694843, -0.02264213,
        0.81304942, -1.25692846,  0.5351692 ,  1.04313459, -0.57641888,
       -0.34599678, -0.56502672,  0.77398128, -0.8819425 ,  1.50490192,
       -0.37983505,  0.65395402,  1.27184629,  0.46784827, -0.66208911,
       -0.2852324 , -0.12804174, -1.52362713,  0.25071717,  0.07572318,
       -0.62833736, -1.76149249, -0.69045391, -0.42221701, -0.1921271 ,
       -1.25038876,  1.04065157,  1.20218778,  0.47994326,  1.27254293,
       -1.39708403,  0.06975819,  0.29934598, -0.24822749,  1.48991752,
        0.51019691, -0.35584382,  0.44550799,  0.16551756,  0.10453712,
        0.66384195,  1.15648211,  0.85713831, -0.28628184, -1.40903507,
        0.36515141,  1.13571479, -0.3584671 , -0.96292667, -0.97836357,
        0.31044   ,  1.7408854 ,  0.68127983,  0.31070714,  0.86275476,
        1.32985186,  0.63221743, -0.90545734, -1.36266071, -0.73065435,
        0.69916299,  0.28328188,  0.32278989])

radius =  float(natoms) ** (1. / 3)
rand_config = lambda : random_structure(natoms, radius)
constraint = HardShellConstraint(radius)


pot = system.get_potential()

gminE = lbfgs_cpp(gmin, pot).energy

nbs = NestedBasinSampling(pot, constraint)
bpot = nbs.get_basinpotential()

#coords = rand_config()
#E = bpot.getEnergy(coords)
#res = bpot.res

gmc = GMCSampler(pot, stepsize=0.2, nsteps=20, constraint=constraint)
quencher = lambda x: AdaptiveNestedOptimizer(x, pot, gmc, stepsize=0.1).run()

bpot = BasinPotential(pot, quencher)

Es = []

E1s = E1

coords1 = rand_config()
E1 = [bpot.getEnergy(coords1) for i in xrange(10)]
coords2 = rand_config()
E2 = [bpot.getEnergy(coords2) for i in xrange(10)]


E1 += [bpot.getEnergy(coords1) for i in xrange(10)]
E2 += [bpot.getEnergy(coords2) for i in xrange(10)]

print np.mean(E1), np.std(E1)
print np.mean(E2), np.std(E2)

samples = [E1, E2]

anderson_ksamp(samples)

raise

for i in xrange(10000):
    Es.append(bpot.getEnergy(rand_config()))
    print i, np.mean(Es), np.std(Es)

Es.sort()
plt.plot(Es)

raise

mc = ThermalSampler(bpot, stepsize=0.2, temperature=1.0, Ecutoff=gminE)
mc.initialise(gmin)

adaptMC = AdaptiveThermalSampler(bpot, stepsize=0.05, temperature=0.5, Ecutoff=gminE,
                                 interval=100,sfactor=1.1,Tfactor=1.1,ntest=1)
adaptMC.initialise(coords)

for i in xrange(1000):
    adaptMC.takeOneStep()

adaptMC.initialise(coords)

for i in xrange(1000):
    adaptMC.takeOneStep()

coords = adaptMC.coords

quencher = lambda x: NestedGalileanOptimizer(x, pot, stepsize=0.5, constraint=constraint).run()
quencher(coords)
quencher(res.coords)

bpot = BasinPotential(pot, quencher)
res = bpot.res

coords = res.coords

x = coords.reshape(-1,3)
x -= x.mean(0)[None,:]

fig, ax = ax3d()
ax.scatter(*gmin.reshape(-1,3).T, c='r')
ax.scatter(*x.T, c='b')


fig, ax = ax3d()
ax.scatter(*gmin.reshape(-1,3).T, c='r')
ax.scatter(*res.initialcoords.reshape(-1,3).T, c='b')
raise

mc.takeOneStep()

for i in xrange(100):
    mc.takeOneStep()

#print E
























#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################




raise Exception('end')


class MyPot(BasePotential):

    def getEnergy(self, x):
        x = np.asanyarray(x)
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
        Vs[ltz == False] = np.inf
        nz = ltz * Vs > -1.
        vols[nz] = - np.log(-Vs[nz])* 2 *np.pi

        return vols

pot = MyPot()

R = 3.
ndim = 200

vol = np.pi * R**2

rand_config = lambda : R*vector_random_uniform_hypersphere(ndim)

Erand = []
for i in xrange(50000):
    Erand.append(pot.getEnergy(rand_config()))
Erand.sort()

plt.plot(Erand, np.linspace(0,1,len(Erand),False)**(2./(ndim)))
plt.plot([0,max(Erand)],[0,1])

constraint = HardShellConstraint(R, ndim=ndim)
constraint.ndim = ndim

_opt_kw['sampler_kw']['constraint'] = constraint
_opt_kw['sampler_kw']['nsteps'] = 50
_opt_kw['tol'] = 5e-3

nbs = NestedBasinSampling(pot, constraint)
bpot = nbs.get_basinpotential()

sampler = GalileanSampler(pot)
mc = MCSampler(pot, takestep=TakestepHyperSphere(.5))



coords = rand_config()
E = pot.getEnergy(coords)

opt = NestedOptimizer(coords, pot, mc)


mc = MCSampler(pot, takestep=TakestepHyperSphere(0.1))
gmc = GMCSampler(pot)

x = coords.copy()
Esamp = []
for i in xrange(100000):
    x, Enew = mc.new_point(E, x, 1)[:2]
    Esamp.append(Enew)

Esamp.sort()
plt.plot(Esamp, np.linspace(0,1,len(Esamp))**(2./(ndim)))


x = coords.copy()
Esamp = []
for i in xrange(10000):
    x, Enew = gmc.new_point(E, x, 1, 0.5)[:2]
    Esamp.append(Enew)

Esamp.sort()
plt.plot(Esamp, np.linspace(0,1,len(Esamp))**(2./(ndim)))


Esamp = []
for i in xrange(10000):
    Esamp.append(pot.getEnergy(np.linalg.norm(coords)*vector_random_uniform_hypersphere(ndim)))

Esamp.sort()
plt.plot(Esamp, np.linspace(0,1,len(Esamp))**(2./(ndim)))


plt.plot([0,E],[0,1])
















Esamp = []
for i in xrange(2000):
    Esamp.append(mc.new_point(E,coords,50)[1])

Esamp.sort()
plt.plot(Esamp, np.linspace(0,1,len(Esamp))**(2./(ndim)))

Esamp = []
for i in xrange(1000):
    Esamp.append(mc.new_point(E,np.linalg.norm(coords)*vector_random_uniform_hypersphere(ndim),20)[1])

Esamp.sort()
plt.plot(Esamp, np.linspace(0,1,len(Esamp))**(2./(ndim)))


Esamp = []
for i in xrange(1000):
    Esamp.append(mc.new_point(E,coords,20)[1])

Esamp.sort()
plt.plot(Esamp, np.linspace(0,1,len(Esamp))**(2./(ndim)))

Esamp = []
for i in xrange(10000):
    Esamp.append(pot.getEnergy(np.linalg.norm(coords)*vector_random_uniform_hypersphere(ndim)))

Esamp.sort()
plt.plot(Esamp, np.linspace(0,1,len(Esamp))**(2./(ndim)))


plt.plot([0,E],[0,1])

Esamp = []
for i in xrange(1000):
    Esamp.append(sampler.new_point(E,coords.copy(),40,1.)[1])

Esamp.sort()
plt.plot(Esamp, np.linspace(0,1,len(Esamp))**(2./(ndim)))

raise


coords = rand_config()
E = bpot.getEnergy(coords)

db = nbs.db

rep_adder = db.replica_adder(10)

for i in xrange(100):
    coords = rand_config()
    Erep = pot.getEnergy(coords)
    qEs = []
    ms = []
    for j in xrange(10):
        qEs.append(bpot.getEnergy(coords))
        db.session.commit()
        ms.append(bpot.min.id())
    print i, Erep, qEs

    rep_adder(Erep, coords, quench_energies=qEs, minima=ms)

run = NestedSamplingRun(volume=vol)

coords = np.array([1,0.])

for i in xrange(500):
    bpot.getEnergy(coords)
    reprun = NestedSamplingRun(Vmax=bpot.res.Emax[1:])
    run += reprun

coords = np.array([1.5,0.])
E = pot.getEnergy(coords)
runs = []
for i in xrange(5):
    runs.append(NestedSamplingRun(volume=vol))

    for i in xrange(25):
        bpot.getEnergy(coords)
        reprun = NestedSamplingRun(Vmax=bpot.res.Emax[1:])
        runs[-1] += reprun

    plt.plot(runs[-1].Vmax, runs[-1].calcBasinFracVolume())
plt.plot(runs[-1].Vmax, pot.getVol(runs[-1].Vmax)/pot.getVol(pot.getEnergy(coords)))

for run in runs:
    plt.plot(run.Vmax, run.calcBasinFracVolume())
plt.plot(runs[-1].Vmax, pot.getVol(runs[-1].Vmax)/pot.getVol(pot.getEnergy(coords)))

sampler = GalileanSampler(pot, stepsize=1.0)
mc = MCSampler(pot)

coords = np.array([1.5,0.])
E = pot.getEnergy(coords)

Esamp = []
for i in xrange(1000):
    Esamp.append(sampler.new_point(E,coords,100)[1])

Esamp.sort()

plt.plot(Esamp, np.linspace(0,1.,len(Esamp),False))


Esamp = []
for i in xrange(1000):
    Esamp.append(mc.new_point(E,coords,100)[1])

Esamp.sort()

plt.plot(Esamp, np.linspace(0,1.,len(Esamp),False))


coords = np.array([1.5,0.,0.,0.,0.,0.])

coords = np.array([1.5,0.,0.,0.,0.,0.,0,0,0,0,0,0,0,0,0])
E = pot.getEnergy(coords)
Esamp = []
for i in xrange(1000):
    Esamp.append(mc.new_point(E,coords,20)[1])

Esamp.sort()
Esamp = np.array(Esamp)
plt.plot((Esamp+1)/(E+1), np.linspace(0,1.,len(Esamp),False)**(1./coords.size))

plt.plot([0,1],[0,1])

plt.plot(Esamp, pot.getVol(Esamp)/pot.getVol(E))








run = NestedSamplingRun(volume=vol)

self = Replicas(db)

for rep in self.replicas:
    reprun = NestedSamplingRun(Vmax=[rep.energy], volume=vol)
    for i in rep.minima:
        reprun += NestedSamplingRun(db.getMinimum(i).user_data['Emax'][1:],
                                    volume=vol,Vcut=rep.energy)
    run += reprun

fracVolume = run.calcBasinFracVolume()

Es = sorted(Es)

Eis = sorted([rep.energy for rep in self.replicas])
fs = [pot.getVol(E)/vol for E in Eis]

plt.plot(Eis, fs)
plt.plot(Eis, np.linspace(0,1,len(Eis),False))

TotFrac = np.interp(Erand,run.Vmax[::-1],
                    run.basinVolume[::-1],0.,run.basinVolume[0])

Frand = np.interp(run.Vmax[::-1], Erand, np.linspace(0,1.,len(Erand)),0,1.)[::-1]

plt.plot(TotFrac,np.linspace(0,vol,len(Erand)))

plt.figure()
plt.semilogx(np.linspace(0,vol,Esort.size),Esort)
plt.semilogx(np.linspace(0,vol,len(Erand)),Erand)
plt.semilogx(fracVolume*run.volume, run.Vmax)


for rep in self.replicas:
    reprun = NestedSamplingRun(Vmax=[rep.energy], volume=vol)
    for i in rep.minima:
        reprun += NestedSamplingRun(db.getMinimum(i).user_data['Emax'][1:],
                                    volume=vol,Vcut=rep.energy)

    plt.semilogx(reprun.calcBasinFracVolume(), reprun.Vmax)















raise

class MyPot(BasePotential):

    a = 0.02

    params = [[-1,np.array([-2,2]),0.5],
         [-2,np.array([2,-2]),0.25]]

    def getEnergy(self, x):
        x = np.asanyarray(x)
        E = self.a * (x**2).sum()
        for a, x0, l in self.params:
            E += a * np.exp(-((x-x0)**2).sum()/l**2 )
        return E

    def getGradient(self, x):
        x = np.asanyarray(x)
        grad = 2 * self.a * x
        for a, x0, l in self.params:
            grad -= 2 * a * (x-x0)/l**2 * np.exp(-((x-x0)**2).sum()/l**2 )
        return grad

    def getEnergyGradient(self, x):
        return self.getEnergy(x), self.getGradient(x)

    def getVol(self, V):

        vol = 0.
        if V > 0:
            vol = 2 * np.pi * V / self.a

pot = MyPot()


R = 6.
res = 6*128 + 1
x = np.linspace(-R,R,res)
X, Y = np.meshgrid(x, x)
Es = np.empty_like(X)

config = (X**2 + Y**2) < R**2


for i in xrange(res):
    for j in xrange(res):
        Es[i,j] = pot.getEnergy([x[i],x[j]])

Esort = Es[config]
Esort.sort()

Erand = []
for i in xrange(500000):
    Erand.append(pot.getEnergy(rand_config()))
Erand.sort()

plt.semilogx(np.linspace(0,1,Esort.size),Esort)
plt.semilogx(np.linspace(0,1,len(Erand)),Erand)


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Es)

rand_config = lambda : R*vector_random_uniform_hypersphere(2)



constraint = HardShellConstraint(R, ndim=2)
constraint.ndim = 2

_opt_kw['sampler_kw']['constraint'] = constraint
_opt_kw['sampler_kw']['nsteps'] = 50
_opt_kw['tol'] = 5e-3

nbs = NestedBasinSampling(pot, constraint)
bpot = nbs.get_basinpotential()

gmc = GMCSampler(pot, stepsize=0.1, nsteps=50, constraint=constraint)
quencher = lambda x: AdaptiveNestedOptimizer(x, pot, gmc, stepsize=0.01,tol=5e-3, frequency=10).run()

bpot = BasinPotential(pot, quencher, database=nbs.db)

coords = rand_config()
E = bpot.getEnergy(coords)

db = nbs.db

rep_adder = db.replica_adder(10)

coords = np.array([R,0])


for i in xrange(40):
    coords = rand_config()
    Erep = pot.getEnergy(coords)
    qEs = []
    ms = []
    for j in xrange(10):
        qEs.append(bpot.getEnergy(coords))
        db.session.commit()
        ms.append(bpot.min.id())
    print i, Erep, qEs

    rep_adder(Erep, coords, quench_energies=qEs, minima=ms)


self = Replicas(db)

self.vol = np.pi * R**2

self.quenchEs = [rep.quench_energies for rep in self.replicas]

minEs = np.sort([m.energy for m in self.mindict.itervalues()])

self.binedges = np.array([-2.,-1,-0,np.inf])
self.nbins = self.binedges.size-1

self.basinvols = np.zeros(self.nbins)
self.weights = [self.vol**-1 for rep in self.replicas]

self.w1 = sum(self.weights)

for i in xrange(self.nbins):
    blo, bhi = self.binedges[i:i+2]
    for w, rep in zip(self.weights, self.replicas):
        qEs = rep.quench_energies
        qEi = sum(1 for E in qEs if blo < E <= bhi)
        self.basinvols[i] += 1. * w * qEi / len(qEs)

self.basinvols *= self.vol/self.w1

#self.basinvols[0] *= 0.13/0.19
#self.basinvols[1] *= 1.0/1.5
#self.basinvols[2] = self.vol - self.basinvols[:-1].sum()

runs = [NestedSamplingRun(volume=v) for v in self.basinvols]

for rep in self.replicas:


for m in self.mindict.values():
    Vmax = m.user_data['Emax']
    nlive = [1 for V in Vmax]
    run = NestedSamplingRun(Vmax, nlive)
    i = np.searchsorted(self.binedges, m.energy) - 1
    runs[i] += run


Efilt = savgol_filter(Esort, 101, 2)
Erand = np.array(Erand)

FracVols = [run.calcBasinFracVolume() for run in runs]
TotFrac = sum(np.interp(Erand,run.Vmax[::-1],
                        run.basinVolume[::-1],0.,run.basinVolume[0])
             for run in runs):

[plt.plot(run.Vmax,run.basinVolume) for run in runs]

plt.semilogx(TotFrac, Erand)

plt.plot(np.linspace(0, self.vol, Erand.size), Erand)

Efilt = savgol_filter(Esort, 101, 2)
diffE = savgol_filter(Esort, 101, 2, deriv=1)


plt.semilogy(Efilt, 1./diffE)


plt.semilogx(TotFrac, Esort)

plt.semilogx(np.linspace(0, self.vol, Esort.size), Efilt)
plt.plot(np.linspace(0, self.vol, Esort.size), Esort)

for run in runs:
    plt.plot(run.basinVolume, run.Vmax)

plt.plot(np.linspace(0, self.vol, Esort.size), Esort)


for run, bv in zip(runs, self.binedges[:self.nbins]):
    plt.plot((run.Vmax -bv), run.basinVolume, )

plt.semilogy(Esort, np.linspace(0, self.vol, Esort.size),)


plt.plot(Esort[:-1], np.diff(Esort))

raise

Vmax1 = self.mindict[1].user_data['Emax']
Vcut1 = np.inf
nlive1 = [1 for V in Vmax1]

Vmax2 = self.mindict[2].user_data['Emax']
Vcut2 = Vmax2[0]+0.01 #np.inf
nlive2 = [1 for V in Vmax2]

run = NestedSamplingRun()
run1 = NestedSamplingRun(Vmax1, nlive1)
run2 = NestedSamplingRun(Vmax2, nlive2)
run12 = run1.combine(run2)

Vmax1 = self.mindict[2].user_data['Emax']
Vcut1 = Vmax1[0] + 0.01
nlive1 = [1 for V in Vmax1]

Vmax2 = self.mindict[1].user_data['Emax']
Vcut2 = Vmax2[0]+0.01 #np.inf
nlive2 = [1 for V in Vmax2]

n1, n2 = len(Vmax1), len(Vmax2)
i1, i2 = 0, 0

Vmaxnew = []
nlivenew = []

while(i1!=n1 or i2!=n2):
    V1 = Vmax1[i1] if i1 < n1 else -np.inf
    live1 = nlive1[i1] if i1 < n1 else 0
    V2 = Vmax2[i2] if i2 < n2 else -np.inf
    live2 = nlive2[i2] if i2 < n2 else 0

    if (V1 > V2):
        Vmaxnew.append(V1)
        nlive = live1
        if V1 < Vcut2:
            nlive += live2
        nlivenew.append(nlive)
        print 'if', i1, i2, V1, V2, live1, live2
        i1 += 1
    else:
        Vmaxnew.append(V2)
        nlive = live2
        if V2 < Vcut1:
            nlive += live1
        nlivenew.append(nlive)
        print 'else', i1, i2, V1, V2, live1, live2
        i2 += 1

zip(nlivenew, Vmaxnew)




for i, qEs in enumerate(self.quenchEs):
    plt.plot([i]*len(qEs), qEs)

raise

def plotcoords(coords, ax=None, fig=None, **kwargs):
    if ax is None:
        fig = plt.figure() if fig is None else fig
        ax = fig.add_subplot(111,projection='3d')

    ax.scatter(*coords.reshape(-1,3).T, **kwargs)

    return fig, ax


natoms = 13
natoms = 31
niter = 100
system = LJCluster(natoms)

gmin  = np.array(
      [-0.57661039, -0.20922621,  1.02839484,  0.03718986, -0.56960802,
        0.19065885, -0.74485456, -0.29376893, -0.5330444 ,  0.51589413,
       -0.38547324,  1.16293184, -0.97689163,  0.82980011,  1.01651968,
        0.44237781,  0.76453522, -1.43379792, -1.58109833,  0.37403757,
       -0.75244974, -1.17803492,  1.13214206, -0.03827416,  0.11377805,
       -0.9574589 , -0.85655503, -0.11345962, -1.63036576, -0.00340774,
        0.81480327, -1.25699092,  0.52149634,  1.03562717, -0.58539809,
       -0.33122239, -0.55082335,  0.76714013, -0.87832192,  1.51033404,
       -0.38728163,  0.64810546,  1.24889779,  0.47289277, -0.66473407,
       -0.28650994, -0.12978066, -1.5346621 ,  0.2469026 ,  0.08722791,
       -0.6274991 , -1.76578221, -0.68578055, -0.42063516, -0.17612401,
       -1.24814226,  1.04274587,  1.2046335 ,  0.48044108,  1.28486289,
       -1.37443511,  0.07223048,  0.30133618, -0.24371389,  1.48979545,
        0.48682862, -0.35623627,  0.45130852,  0.17777359,  0.11400512,
        0.65740392,  1.14976934,  0.85312099, -0.30131762, -1.42034537,
        0.37785698,  1.1214737 , -0.35276543, -0.97313091, -0.96912921,
        0.31447945,  1.72926618,  0.67038536,  0.31580514,  0.85445676,
        1.31467926,  0.62662403, -0.91492618, -1.35463763, -0.73063174,
        0.71348693,  0.27886602,  0.31001412])


radius =  float(natoms) ** (1. / 3)
rand_config = lambda : random_structure(natoms, radius)
constraint = HardShellConstraint(radius)


pot = system.get_potential()
coords = rand_config()

nbs = NestedBasinSampling(pot, constraint)
bpot = nbs.get_basinpotential()

E = bpot.getEnergy(coords)
res = bpot.res
print E

sampler = MCSampler(bpot, stepsize=1.0, nsteps=100, verbose=True)

newcoords, Enew, niter, ntries, nres, nmin = sampler.new_point(res.coords, E+0.2, 10)

f, ax = plotcoords(coords, c='b')
plotcoords(res.coords, ax=ax, c='r')
plotcoords(newcoords, ax=ax, c='g')





