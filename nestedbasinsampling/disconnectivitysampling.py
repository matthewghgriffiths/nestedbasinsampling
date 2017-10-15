# -*- coding: utf-8 -*-

from itertools import chain, izip
from random import choice, seed

import numpy as np
from scipy.stats import anderson_ksamp, ks_2samp

import networkx as nx
from networkx.algorithms.traversal import bfs_successors, bfs_edges

from pele.optimize import lbfgs_cpp

from nestedbasinsampling.utils import dict_update_keep
from nestedbasinsampling.samplers import SamplingError, GMCSampler
from nestedbasinsampling.constraints import BaseConstraint
from nestedbasinsampling.nestedoptimization import \
    BasinPotential, AdaptiveNestedOptimizer
from nestedbasinsampling.disconnectivitydatabase import \
    Minimum, Replica, Run, Database
from nestedbasinsampling.disconnectivitygraphs import \
    ReplicaGraph, BasinGraph, SuperBasin, AndersonDarling


class DisconnectivitySystem(object):
    """
    """
    def __init__(self, pot, constraint=None,
                 sampler=GMCSampler, sampler_kw={},
                 minimizer=AdaptiveNestedOptimizer, minimizer_kw={},
                 globalbasin=None, database=None, startseed=0):

        self.pot = pot
        self.constraint = (BaseConstraint() if constraint is None 
                           else constraint)

        self.sampler = sampler
        self.sampler_kw = sampler_kw
        self.sampler_kw['constraint'] = constraint

        self.minimizer = minimizer
        self.minimizer_kw = minimizer_kw

        self.seed = startseed
        seed(self.seed)
        np.random.seed(self.seed)

    def get_database(self, *args, **kwargs):
        return Database(*args, **kwargs)

    def get_sampler(self, **kwargs):
        dict_update_keep(kwargs, self.sampler_kw)
        sampler = self.sampler(pot, **kwargs)
        return sampler

    def _quench(self, coords, sampler_kw={}, **kwargs):

        dict_update_keep(sampler_kw, self.sampler_kw)
        dict_update_keep(kwargs, self.minimizer_kw)

        ## Saving as module variables to allow inspection of values
        self._sampler = self.sampler(self.pot, **sampler_kw)
        self._opt = self.minimizer(coords, self.pot, 
                                   self._sampler, **kwargs)
        self._res = self._opt.run()
        return self._res

    def get_minimizer(self):
        return self._quench

    def get_basinpotential(self):
        return BasinPotential(self.pot, quench=self.get_minimizer())

class DisconnectivitySampler(object):

    sampleMinStr = \
        "DIS > Sampling minima, Ecut = {:8.4g}, minE = {:8.4g}, logF = {:8.4g}"

    def __init__(self, pot, bpot, sampler, graph=None, random_config=None,
                 database=None, run_adder=None, rep_adder=None, min_adder=None,
                 debug=False):

        self.pot = pot
        self.bpot = bpot
        self.sampler = sampler

        self.database = database
        self.repGraph = ReplicaGraph(database, run_adder=run_adder,
                                 rep_adder=rep_adder, min_adder=min_adder
                                 ) if graph is None else graph

        self.basinGraph = BasinGraph(self.repGraph)

        self.NewRun = self.repGraph.NewRun
        self.NewRep = self.repGraph.NewRep
        self.NewMin = self.repGraph.NewMin

        self.Run = self.repGraph.Run
        self.Replica = self.repGraph.Replica
        self.Minimum = self.repGraph.Minimum

        self.random_config = random_config
        self.debug=debug

    def sampleReplica(self, replica, stepsize=None, nsteps=None):

        Ecut = replica.energy

        if stepsize is None:
            if replica.stepsize is None:
                stepsize = sampler.stepsize
            else:
                stepsize = replica.stepsize

        samples = self.repGraph.findSamples(replica)

        # Generating starting location
        if replica.coords is None: # Sample randomly
            if self.random_config is None:
                raise NotImplementedError("random_config not defined")
            else:
                coords = self.random_config()
                E = self.pot.getEnergy(coords)
        elif samples:
            coords = choice(samples)
            # Continue Markov Chain
            coords, E, naccept, nreject = sampler.new_point(
                Ecut, coords, stepsize=stepsize, nsteps=nsteps)
        else:
            # Perform random walk starting at replica
            coords, E, naccept, nreject = sampler.new_point(
                Ecut, replica.coords, stepsize=stepsize, nsteps=nsteps)

        newrep = self.Replica(E, coords, stepsize=stepsize)
        newrun = self.Run([E],[1], replica, newrep,
                          configs=[coords], stepsizes=[stepsize])

        return newrep, newrun

    def sampleMinima(self, replica, nsamples=20, stepsize=None):

        minima = []
        runs = []

        Ecut = replica.energy
        stepsize = replica.stepsize

        for i in xrange(nsamples):
            parent, prun = self.sampleReplica(replica, stepsize)
            minEnergy = self.bpot.getEnergy(parent.coords,
                                            stepsize=parent.stepsize)
            res = self.bpot.res

            m = self.Minimum(res.energy, res.coords)
            child = self.Replica(res.Emax[-1], res.configs[-1], minimum=m)

            Emax = res.Emax[1:] # np.r_[replica.energy, res.Emax]
            nlive = np.ones_like(Emax, dtype=int)
            stepsizes = res.stepsizes[1:] # np.r_[stepsize, res.stepsizes]
            configs = res.configs[1:] # np.vstack((coords, bpot.res.configs))
            run = self.Run(Emax, nlive, parent, child,
                           stepsizes=stepsizes, configs=configs)

            minima.append(m)
            runs.append(run)

            if self.debug:
                print self.sampleMinStr.format(Ecut, minEnergy,
                                               -len(Emax)*np.log(2))

        return minima, runs

    def compareSuperBasins(self, basin1, basin2):

        min1 = basin1.getMinima()
        min2 = basin2.getMinima()
        minE1 = [m.energy for m in min1]
        minE2 = [m.energy for m in min2]

        return minE1, minE2, anderson_ksamp([minE1, minE2])

import seaborn as sns

from pele.systems import LJCluster
from nestedbasinsampling.constraints import HardShellConstraint
from nestedbasinsampling.random import random_structure

import matplotlib.pyplot as plt
from plottingfuncs.plotting import ax3d


natoms = 31
system = LJCluster(natoms)

radius =  float(natoms) ** (1. / 3)
rand_config = lambda : random_structure(natoms, radius)

constraint = HardShellConstraint(radius)
pot = system.get_potential()

minimizer_kw = dict(tol=1e-1, alternate_stop_criterion=None,
                   events=None, iprint=-1, nsteps=10000, stepsize=0.6,
                   debug=True, quench=lbfgs_cpp, quenchtol=1e-6,
                   quench_kw=dict(nsteps=1000))

sampler_kw = dict(nsteps=30, maxreject=200, debug=True, constraint=constraint)

disSystem = DisconnectivitySystem(pot, constraint=constraint,
                                sampler_kw=sampler_kw, minimizer_kw=minimizer_kw)

sampler = disSystem.get_sampler()
bpot = disSystem.get_basinpotential()
db = disSystem.get_database("dis.sql")


driver = DisconnectivitySampler(pot, bpot, sampler, debug=True,
    random_config=rand_config, database=db)

g = driver.repGraph
bG = driver.basinGraph

basins = driver.basinGraph.basins()

basin = driver.basinGraph.get_lowest_basin()

gbasin = basins[-1]

minDistr = driver.basinGraph.getBasinMinimaDistribution(gbasin)

Es, fs = zip(*((m.energy, f) for m,f in minDistr))

calcCDF = AndersonDarling.calcCDF

E1s, F1s, f1s = calcCDF(Es,fs)
E2s, F2s, f2s = calcCDF(E1s+0.5,fs)

Es = np.r_[E1s, E2s]
Es.sort(kind='mergesort')

F1i = np.searchsorted(E1s, Es,side='right')
F1c = np.r_[F1s,0.][F1i-1]

F2i = np.searchsorted(E2s, Es,side='right')
F2c = np.r_[F2s,0.][F2i-1]

plt.ion()

plt.plot(E1s, F1s)
plt.plot(E2s, F2s)


plt.plot(Es, F1c)
plt.plot(Es, F2c)



















raise

replica = driver.Replica(np.inf, None)
minima, runs = driver.sampleMinima(replica, nsamples=10)

driver.basinGraph.update()

m1, m2 = minima[:2]

for path in nx.all_simple_paths(g.repGraph, replica, m1):
    print [r.energy for r in path]



raise

graph = driver.graph

graph = NestedGraph()
rep1 = graph.NewRep(1, [1])
m1 = graph.NewMin(-1,[1])
rep2 = graph.NewRep(0, [2],minimum=m1)
m2 = graph.NewMin(-2,[1])
rep3 = graph.NewRep(-1,[3],minimum=m2)

run1 = graph.NewRun([0.],[1],rep1, rep2)
run2 = graph.NewRun([-1.],[1],rep2, rep3)

graph.addMinima([m1,m2])
graph.addReplicas([rep1,rep2,rep3])
graph.addRuns([run1, run2])

graph.runs()
graph.replicas()
graph.minima()

raise
driver.sampleSuperbasin(globalbasin, nsamples=30)

minima = globalbasin.getMinima()

m = min(minima, key= lambda x: x.energy)

run = sum(m.parents, Run())

run1, run2, newrep = run.split(len(run)/2)

newbasin = SuperBasin(newrep, parent=globalbasin)

driver.sampleSuperbasin(newbasin, nsamples=10)

driver.compareSuperBasins(globalbasin, newbasin)

raise

run3, run4, newrep2 = run.split(len(run2)/2)

newbasin2 = SuperBasin(newrep2, parent=globalbasin)

driver.sampleSuperbasin(newbasin2, nsamples=20)

e1, e2, andres = driver.compareSuperBasins(globalbasin, newbasin2)

e1.sort()
e2.sort()
plt.plot(e1, np.linspace(0,1,len(e1)));plt.plot(e2, np.linspace(0,1,len(e2)))


raise


coords = rand_config()
E = pot.getEnergy(coords)
rep = Replica(coords, E)

driver.globalbasin.addReplica(rep)

m, run = driver.minimizeReplica(rep)

run1, run2, newrep = run.split(len(run)/2)

print 0
#driver.sampleReplica(rep, 3)
print 1
#driver.sampleReplica(newrep, 3)
print 2

driver.compareDistributions(rep, newrep)

raise

run3, run4, newrep2 = run.split(len(run1)/2)

driver.sampleReplica(newrep2, 20)


e1, e2, andres = driver.compareDistributions(rep, newrep2)

driver.sampleReplica(rep, 30)
driver.sampleReplica(newrep2, 30)
e1, e2, andres = driver.compareDistributions(rep, newrep2)

e1.sort()
e2.sort()
plt.plot(e1, np.linspace(0,1,len(e1)));plt.plot(e2, np.linspace(0,1,len(e2)))

sns.distplot(e1);sns.distplot(e2)
andres
#bE = bpot.getEnergy(coords)
#res = bpot.res

#m = Minimum(res.coords, res.energy, parents=[rep])
#db.addMinimum(m)
#
#run = Run(res.Emax, configs=res.configs, stepsizes=res.stepsizes,
#                        parent=rep, child=m)
#db.addRun(run)
#
#run1, run2, newrep = run.split(len(run)/2)
#
#globalbasin.addReplica(rep)


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

























