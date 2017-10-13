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
from nestedbasinsampling.nestedoptimization import BasinPotential, AdaptiveNestedOptimizer

from nestedbasinsampling.disconnectivitydatabase import Minimum, Replica, Run, Database

class NestedGraph(object):

    def __init__(self, database=None,
                 run_adder=None, rep_adder=None, min_adder=None):

        database = Database() if database is None else database
        # These methods will be used to create new objects
        self.NewRun = database.addRun     if run_adder is None else run_adder
        self.NewRep = database.addReplica if rep_adder is None else rep_adder
        self.NewMin = database.addMinimum if min_adder is None else min_adder

        # Create new graph from database
        self.loadFromDatabase(database, newGraph=True)

    def loadFromDatabase(self, database, newGraph=True):
        if newGraph:
            self.repGraph = nx.DiGraph()

        self.database = database

        minima = self.database.minima()
        replicas = self.database.replicas()
        runs = self.database.runs()

        self.addMinima(minima)
        self.addReplicas(replicas)
        self.addRuns(runs)

    def Minimum(self, energy, coords):
        m = self.NewMin(energy, coords)
        self.addMinima([m])
        return m

    def Replica(self, energy, coords, minimum=None, stepsize=None):
        rep = self.NewRep(energy, coords, minimum=minimum, stepsize=stepsize)
        self.addReplicas([rep])
        return rep

    def Run(self, Emax, nlive, parent, child,
            volume=1., configs=None, stepsizes=None):
        run = self.NewRun(Emax, nlive, parent, child,
                          volume=volume, configs=configs, stepsizes=stepsizes)
        self.addRuns([run])
        return run

    def addMinima(self, minima):
        for m in minima:
            self.repGraph.add_node(m, energy=m.energy)

    def addReplicas(self, replicas):
        for rep in replicas:
            self.repGraph.add_node(rep, energy=rep.minimum)
            if rep.minimum is not None:
                self.repGraph.add_edge(rep, rep.minimum, type='min',
                                    Erep=rep.energy, Emin=rep.minimum.energy)

    def addRuns(self, runs):
        for run in runs:
            self.repGraph.add_edge(run.parent, run.child, run=run, type='run',
                                Ecut=run.parent.energy, Emin=run.child.energy)

    def replicas(self, order=True):
        if order:
            return sorted((n for n in self.repGraph.nodes() if type(n) is Replica),
                          key=lambda r: r.energy)
        else:
            return [n for n in self.repGraph.nodes() if type(n) is Replica]

    def minima(self, order=True):
        if order:
            return sorted((n for n in self.repGraph.nodes() if type(n) is Minimum),
                          key=lambda r: r.energy)
        else:
            return [n for n in self.repGraph.nodes() if type(n) is Minimum]

    def runs(self):
        edges = self.repGraph.edge
        runs = []
        for node1, edges1 in edges.iteritems():
            for node2, attr in edges1.iteritems():
                if attr['type'] is 'run':
                    runs.append(attr['run'])
        return runs

    def findSamples(self, replica):
        successors = self.repGraph.successors(replica)
        runs = [self.repGraph.get_edge_data(replica, rep)['run']
                for rep in successors if type(rep) is Replica]
        samples = []
        for run in runs:
            if len(run.Emax) == 1:
                samples.append(run.child.coords)
            else:
                samples.append(run.configs[1])
        return samples

    def findPredecessors(self, node1, node2):
        pred1 = set(sum(bfs_edges(self.repGraph.reverse(), node1),()))
        return pred1.intersection(sum(bfs_edges(self.repGraph.reverse(), node1),()))

    def splitRun(self, run, Ecut, replace=False):

        configs = run.configs
        nlive = run.nlive
        stepsizes = run.stepsizes
        Emax = run.Emax
        volume = run.volume
        parent = run.parent
        child = run.child

        i = - np.searchsorted(Emax[::-1], Ecut) - 1

        if self.database is not None and replace:
            self.database.removeRun(run)
            NewRep = self.Replica
            NewRun = self.Run
        else:
            NewRep = Replica
            NewRun = Run

        newrep = NewRep(configs[i], Emax[i], stepsize=stepsizes[i])
        run1 =   NewRun(Emax[:i+1], nlive[:i+1], parent, child, volume=volume,
                        configs=configs[:i+1],stepsizes=stepsizes[:i+1])
        run2 =   NewRun(Emax[i+1:], nlive[i+1:], newrep, child,
                        configs=configs[i+1:], stepsizes=stepsizes[i+1:])

        if replace:
            self.repGraph.remove_edge(parent, child)

        return run1, run2, newrep

    def draw(self, **kwargs):
        pos = nx.nx_agraph.graphviz_layout(self.repGraph, prog='dot')
        nx.draw(self.repGraph, pos, **kwargs)

class SuperBasin(object):
    """ SuperBasin object.

    A super basin is defined by a single replica. All the states accessible via
    downhill paths from this replica are part of the superbasin.

    Attributes
    ----------
    replicas : frozenset of Replica
        a frozenset of replicas sampled from superbasin
    """
    def __init__(self, replicas=frozenset()):
        self.replicas = frozenset(replicas)
        self.energies = sorted(rep.energy for rep in self.replicas)
        self.energy = self.energies[0]

    def __iter__(self):
        return iter(self.replicas)

    def __add__(self, new):
        new = [new] if type(new) is Replica else new
        replicas = self.replicas.union(new)
        return self.__class__(replicas=replicas)

    def __hash__(self):
        return hash(self.replicas)

class BasinGraph(object):

    def __init__(self, graph):
        self.graph = graph
        self.initialise()

    def initialise(self):

        self.basinGraph = nx.DiGraph()

        replicas = self.graph.replicas()
        minima = self.graph.minima()

        self.repnodes = dict((rep, SuperBasin([rep])) for rep in replicas)
        self.repnodes.update((m,m) for m in minima)

        for node in self.repnodes.values():
            self.add_node(node)
        for parent, child in self.graph.repGraph.edges_iter():
            self.add_edge(self.repnodes[parent], self.repnodes[child])

    def add_node(self, node, **kwargs):
        self.basinGraph.add_node(node, energy=node.energy, **kwargs)

    def add_edge(self, parent, child, **kwargs):
        self.basinGraph.add_edge(parent, child, Eparent=parent.energy,
                                 Echild=child.energy, **kwargs)

    def update(self, newGraph=False):
        if newGraph:
            return self.initialise()

        replicas = self.graph.replicas()
        minima = self.graph.minima()

        newrepnodes = dict((rep, SuperBasin([rep])) for rep in replicas
                           if rep not in self.repnodes)
        newrepnodes.update((m,m) for m in minima if m not in self.repnodes)

        self.repnodes.update(newrepnodes)

        for node in newrepnodes.itervalues():
            self.add_node(node, energy=node.energy)

        for key in newrepnodes.iterkeys():
            edges = self.graph.repGraph.edge[key]
            for childkey in edges:
                self.add_edge(self.repnodes[key], self.repnodes[childkey])

    def joinBasins(self, basins):
        newbasin = sum(basins, [])

        self.basinGraph.add_node(newbasin)

        predecessors = set(sum((self.basinGraphs.predecessors(b)
                               for b in basins), [])).difference(basins)
        successors = set(sum((self.basinGraphs.successors(b)
                              for b in basins), [])).difference(basins)

        for parent in predecessors:
            self.add_edge(parent, newbasin)
        for child in successors:
            self.add_edge(newbasin, child)

    def draw(self, **kwargs):
        pos = nx.nx_agraph.graphviz_layout(self.basinGraph, prog='dot')
        nx.draw(self.basinGraph, pos, **kwargs)

def combineRuns(run1, run2):
    Emax1 = run1.Emax
    nlive1 = run1.nlive
    Ecut1 = run1.Ecut
    vol1 = run1.volume

    Emax2 = run2.Emax
    nlive2 = run2.nlive
    Ecut2 = run2.Ecut
    vol2 = run2.volume

    # If the configurations/stepsizes have been stored these need to be
    # combined as well
    config = ((run2.configs is not None or len(run2)==0) and
              (run1.configs is not None or len(run1)==0))
    stepsize = ((run2.stepsizes is not None or len(run2)==0) and
                (run1.stepsizes is not None or len(run1)==0))
    configs = [] if config else None
    stepsizes = [] if stepsize else None

    n1, n2 = len(Emax1), len(Emax2)
    i1, i2 = 0, 0

    Emaxnew = []
    nlivenew = []

    while(i1!=n1 or i2!=n2):
        E1 = Emax1[i1] if i1 < n1 else -np.inf
        live1 = nlive1[i1] if i1 < n1 else 0
        E2 = Emax2[i2] if i2 < n2 else -np.inf
        live2 = nlive2[i2] if i2 < n2 else 0

        if (E1 > E2):
            Emaxnew.append(E1)
            nlive = live1
            if E1 < Ecut2:
                nlive += live2
            nlivenew.append(nlive)
            if config:
                configs.append(run1.configs[i1])
            if stepsize:
                stepsizes.append(run1.stepsizes[i1])
            i1 += 1
        else:
            Emaxnew.append(E2)
            nlive = live2
            if E2 < Ecut1:
                nlive += live1
            nlivenew.append(nlive)
            if config:
                configs.append(run2.configs[i2])
            if stepsize:
                stepsizes.append(run2.stepsizes[i2])
            i2 += 1

    Ecut, volume = max(((Ecut1,vol1), (Ecut2,vol2)))
    configs = np.array(configs) if config else None
    stepsizes = np.array(stepsizes) if config else None

    return Run(Emax=Emaxnew, nlive=nlivenew, configs=configs,
               volume=volume, Ecut=Ecut, stepsizes=stepsizes)

class DisconnectivitySystem(object):
    """
    """
    def __init__(self, pot, constraint=None,
                 sampler=GMCSampler, sampler_kw={},
                 minimizer=AdaptiveNestedOptimizer, minimizer_kw={},
                 globalbasin=None, database=None, startseed=0):

        self.pot = pot
        self.constraint = BaseConstraint() if constraint is None else constraint

        self.sampler = sampler
        self.sampler_kw = sampler_kw
        self.sampler_kw['constraint'] = constraint

        self.minimizer = minimizer
        self.minimizer_kw = minimizer_kw

        self.seed = startseed
        seed(self.seed)
        np.random.seed(self.seed)

    def get_database(self):
        return Database()

    def get_sampler(self, **kwargs):
        dict_update_keep(kwargs, self.sampler_kw)
        sampler = self.sampler(pot, **kwargs)
        return sampler

    def _quench(self, coords, sampler_kw={}, **kwargs):

        dict_update_keep(sampler_kw, self.sampler_kw)
        dict_update_keep(kwargs, self.minimizer_kw)

        ## Saving as module variables to allow inspection of values
        self._sampler = self.sampler(self.pot, **sampler_kw)
        self._opt = self.minimizer(coords, self.pot, self._sampler, **kwargs)
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
        self.graph = NestedGraph(database, run_adder=run_adder,
                                 rep_adder=rep_adder, min_adder=min_adder
                                 ) if graph is None else graph

        self.basinGraph = BasinGraph(self.graph)

        self.NewRun = self.graph.NewRun
        self.NewRep = self.graph.NewRep
        self.NewMin = self.graph.NewMin

        self.Run = self.graph.Run
        self.Replica = self.graph.Replica
        self.Minimum = self.graph.Minimum

        self.random_config = random_config
        self.debug=debug

    def sampleReplica(self, replica, stepsize=None, nsteps=None):

        Ecut = replica.energy

        if stepsize is None:
            if replica.stepsize is None:
                stepsize = sampler.stepsize
            else:
                stepsize = replica.stepsize

        samples = self.graph.findSamples(replica)

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
db = disSystem.get_database()
driver = DisconnectivitySampler(pot, bpot, sampler, debug=True,
    random_config=rand_config, database=db)

g = driver.graph

replica = driver.Replica(np.inf, None)
minima, runs = driver.sampleReplica(replica, nsamples=3)

m1, m2 = minima[:2]

for path in nx.all_simple_paths(g.repGraph, replica, m1):
    print [r.energy for r in path]

bG = g.basinGraph

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

























