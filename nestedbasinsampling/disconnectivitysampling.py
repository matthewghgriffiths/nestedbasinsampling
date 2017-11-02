# -*- coding: utf-8 -*-

from itertools import chain, izip
from random import choice, seed, sample

import numpy as np
from scipy.stats import anderson_ksamp, ks_2samp

import networkx as nx
from networkx.algorithms.traversal import bfs_successors, bfs_edges

from pele.optimize import lbfgs_cpp, LBFGS_CPP

from nestedbasinsampling.utils import dict_update_keep, dict_update_copy
from nestedbasinsampling.samplers import \
    SamplingError, GMCSampler, DetectStep
from nestedbasinsampling.constraints import BaseConstraint
from nestedbasinsampling.nestedoptimization import \
    BasinPotential, AdaptiveNestedOptimizer, RecordMinimization
from nestedbasinsampling.disconnectivitydatabase import \
    Minimum, Replica, Run, Database
from nestedbasinsampling.disconnectivitygraphs import \
    ReplicaGraph, BasinGraph, SuperBasin
from nestedbasinsampling.stats import CDF, AndersonDarling, AgglomerativeCDFClustering
from nestedbasinsampling.nestedsampling import \
    findRunSplit, joinRuns, combineRuns

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

    def get_replicaGraph(self, database=None, **kwargs):
        database = self.get_database() if database is None else database
        return ReplicaGraph(database, **kwargs)

    def get_sampler(self, **kwargs):
        dict_update_copy(kwargs, self.sampler_kw)
        sampler = self.sampler(pot, **kwargs)
        return sampler

    def _quench(self, coords, sampler_kw={}, **kwargs):

        sampler_kw = dict_update_copy(sampler_kw, self.sampler_kw)
        kwargs = dict_update_copy(kwargs, self.minimizer_kw)

        pot = self.pot if 'pot' not in kwargs else kwargs.pop('pot')
        ## Saving as module variables to allow inspection of values
        self._sampler = self.sampler(pot, **sampler_kw)
        self._opt = self.minimizer(coords, pot,
                                   self._sampler, **kwargs)
        self._res = self._opt.run()
        return self._res

    def get_minimizer(self):
        return self._quench

    def get_basinpotential(self):
        return BasinPotential(self.pot, quench=self.get_minimizer())

class DisconnectivitySampler(object):
    """ This class controls overall the sampling of the disconnectivity graph

    Attributes
    ----------

    pot : BasePotential
        The potential being explored
    quench : callable
        Performs some type of minimization (can be stochastic or deterministic)
    sampler : BaseSampler
        samples the basin of a replica
    repGraph : ReplicaGraph
        stores the results of the sampling
    basinGraph : BasinGraph
        aggregates the results of the sampling

    Methods
    -------
    """

    sampleMinStr = \
        "DIS > Sampling minima, Ecut = {:8.4g}, minE = {:8.4g}, logF = {:8.4g}"
    branchStr = \
        "DIS > Sampling branches, Esplit = {:8.4g}, sig = {:8.4g}"

    def __init__(self, pot, quench, sampler, repGraph=None, basinGraph=None,
                 reject_sig=1e-1, accept_sig=0.3, splitf=0.5,
                 nsamples=10, nbranch=None, detectStep=None, fixConstraint=None,
                 random_config=None, debug=False):

        self.pot = pot
        self.quench = quench # Nested Optimisation or direct minimizer (e.g. LBFGS)
        self.minimizer = RecordMinimization(self.pot, minimizer=self.quench)
        self.sampler = sampler
        self.detectStep = detectStep
        self.fixConstraint = fixConstraint

        self.reject_sig = reject_sig
        self.accept_sig = accept_sig

        self.splitf = 0.5
        self.nsamples = nsamples
        self.nbranch = nsamples if nbranch is None else nbranch

        self.repGraph = ( ReplicaGraph(Database()) if repGraph is None
                          else repGraph )
        self.basinGraph = ( BasinGraph(self.repGraph, target_sig=reject_sig)
                            if basinGraph is None else basinGraph)

        self.Run = self.repGraph.Run
        self.Replica = self.repGraph.Replica
        self.Minimum = self.repGraph.Minimum
        self.Path = self.repGraph.Path

        self.random_config = random_config
        self.debug=debug

    def globalReplicaBasin(self):
        """
        """
        reps = (rep for rep in self.repGraph.replicas() if rep.coords is None)
        try:
            replica = reps.next()
            basin = self.basinGraph.repdict[replica]
        except StopIteration:
            replica = self.Replica(np.inf, None)
            basin = self.basinGraph.SuperBasin([replica])
        return replica, basin

    def sampleReplica(self, replica, stepsize=None, nsteps=None):

        Ecut = replica.energy

        # Setting the stepsize
        if stepsize is None:
            if replica.stepsize is None:
                if self.detectStep is None:
                    stepsize = sampler.stepsize
                elif replica.coords is not None:
                    stepsize = self.detectStep(Ecut, replica.coords)[0]
                    replica.stepsize = stepsize
                else:
                    stepsize = sampler.stepsize
            else:
                stepsize = replica.stepsize

        # If we have already sampled from replica, then we can use those as
        # a starting point
        samples = self.repGraph.findSamples(replica)

        # Generating starting location
        if replica.coords is None: # Sample randomly
            if self.random_config is None:
                raise NotImplementedError("random_config not defined")
            else:
                coords = self.random_config()
                E = self.pot.getEnergy(coords)
        else:
            coords = choice(samples) if samples else replica.coords
            # Continue Markov Chain
            try:
                coords, E, naccept, nreject = sampler.new_point(
                    Ecut, coords, stepsize=stepsize, nsteps=nsteps)
            except SamplingError as e:
                if self.debug:
                    print "DIS > sampleReplica warning:",  SamplingError
                    print "DIS > stepsize = {:8.5g}".format(stepsize)
                return self.sampleReplica(replica, stepsize, nsteps)


        res = self.minimizer(coords)

        Emax = res.energy_s if hasattr(res, 'energy_s') else None
        nlive = (res.nlive if hasattr(res, 'nlive') else
                 np.ones_like(Emax, dtype=int))
        stepsizes = res.stepsize if hasattr(res, 'stepsize') else None
        configs = res.coords_s if hasattr(res, 'coords_s') else None
        stored = np.arange(len(configs))

        repcoords = res.coords if configs is None else configs[-1]

        child = self.Replica(res.energy, repcoords)
        m = self.Minimum(res.energy, res.coords)
        path = self.Path(child.energy, child, m)

        runs, replicas = self.Run(Emax, nlive, replica, child, stored=stored,
                                  stepsizes=stepsizes, configs=configs)

        if self.debug:
            print self.sampleMinStr.format(Ecut, res.energy,
                                           -len(Emax)*np.log(2))

        return m, runs, path, replicas

    def insertNewESplit(self, basin):
        replica = max(self.basinGraph.getBasinBranchReplicas(basin)[0],
                      key=lambda rep: rep.energy)
        print replica
        print list(self.basinGraph.genPaths(basin, replica))
        path = min(self.basinGraph.genPaths(basin, replica),
                       key=lambda p: p[0].energy)
        run = repGraph.pathtoRun(path)
        Esplit = findRunSplit(run)[0]
        return repGraph.insertEnergySplit(Esplit)

    def sampleBasin(self, basin, nsamples=None, nbranch=None, stepsize=None):
        """
        """

        nbranch = self.nbranch if nbranch is None else nbranch
        nsamples = self.nsamples if nsamples is None else nsamples

        replicas, Esplit = self.basinGraph.getBasinBranchReplicas(basin)

        if len(replicas) == 0:
            basinreplica = choice(list(basin.replicas))
            self.sampleReplica(basinreplica, stepsize=stepsize)
            replicas, Esplit = self.basinGraph.getBasinBranchReplicas(basin)

        if Esplit is None:
            self.insertNewESplit(basin)
            replicas, Esplit = self.basinGraph.getBasinBranchReplicas(basin)

        while(len(replicas) < nbranch):
            basinreplica = choice(list(basin.replicas))
            self.sampleReplica(basinreplica, stepsize=stepsize)
            replicas, Esplit = self.basinGraph.getBasinBranchReplicas(basin)

        sampled, unsampled = [], []
        for r in replicas:
            (unsampled, sampled)[repGraph.nsuccessors(r) > nsamples].append(r)

        if len(sampled) < nbranch:
            newsamples = sample(unsampled, nbranch - len(sampled))
            for r in newsamples:
                for i in xrange(nsamples-self.repGraph.nsuccessors(r)):
                    self.sampleReplica(r, stepsize=stepsize)

            sampled += newsamples

        unsampled = list(set(replicas).difference(sampled))

        return sampled, unsampled, Esplit


import seaborn as sns

from pele.systems import LJCluster
from nestedbasinsampling.constraints import HardShellConstraint, CombinedPotConstraint
from nestedbasinsampling.takestep import random_structure
from nestedbasinsampling.alignment import CompareStructures

import matplotlib.pyplot as plt
from plottingfuncs.plotting import ax3d

###############################################################################
###################################SETUP#######################################
###############################################################################

natoms = 31
system = LJCluster(natoms)
radius =  float(natoms) ** (1. / 3)
rand_config = lambda : random_structure(natoms, radius)
constraint = HardShellConstraint(radius)
pot = system.get_potential()
minimizer_kw = dict(tol=1e-1, alternate_stop_criterion=None,
                   events=None, iprint=-1, nsteps=10000, stepsize=0.3,
                   debug=True, quench=lbfgs_cpp, quenchtol=1e-6,
                   quench_kw=dict(nsteps=1000))
sampler_kw = dict(nsteps=30, maxreject=200, debug=True, stepsize=0.1,
                  constraint=constraint, fixConstraint=True)
disSystem = DisconnectivitySystem(pot, constraint=constraint,
                                  sampler_kw=sampler_kw, minimizer_kw=minimizer_kw,
                                  startseed=1,)
sampler = disSystem.get_sampler(nsteps=1000, maxreject=5000, fixConstraint=True,
                                constraint=constraint)
detectStep = DetectStep(disSystem.get_sampler(maxreject=100), nadapt=30,
                    step_factor=1.5, stepsize=0.1, debug=False)
compare = CompareStructures(niter=100)
db = disSystem.get_database(compareMinima=compare)
repGraph = disSystem.get_replicaGraph(db)
quench = lbfgs_cpp # disSystem._quench
driver = DisconnectivitySampler(pot, quench, sampler, repGraph,
                                nsamples=25, nbranch=10, detectStep=detectStep,
                                debug=True, random_config=rand_config)
basinGraph = driver.basinGraph

###############################################################################
###############################################################################
###############################################################################

self = driver
nsamples = 10

if len(db.minima()) == 0:
    replica, basin = self.globalReplicaBasin()

sampled, unsampled, Esplit = self.sampleBasin(basin)

repsets, repCDFs = self.repGraph.getReplicaMinimaCDFs(sampled+unsampled)

sig = AndersonDarling.compareDistributions(repCDFs)[0]
agglom = AgglomerativeCDFClustering(repCDFs)
anyMingtEsplit = any((m[0].energy > Esplit
                      for m in basinGraph.genConnectedMinima(basin)))

if sig < self.reject_sig or anyMingtEsplit:
    ps, clusteri, clusters = agglom.getMaxLikelihood()
    # Add new basins
    print ps
    newbasins = [self.basinGraph.SuperBasin(
        reduce(lambda x, y: x.union(y), (repsets[i][0] for i in ind)))
                 for p, ind in izip(ps, clusteri) if  p > self.accept_sig]
    print newbasins
    if len(newbasins) == 0:
        self.insertNewESplit(basin)
    else:
        nclusters = [sum(len(repsets[i]) for i in ind) for ind in clusteri]
        if any(n==1 for n in nclusters):
            self.insertNewESplit(basin)

elif sig > self.accept_sig:
    self.basinGraph.addtoBasin(basin, sampled)

else:
    ps, clusteri, clusters = agglom.getMaxClusters(self.accept_sig)
    retest = filter(lambda p: len(p[0]) == 1 or p[1] < self.accept_sig,
                    izip(clusteri, ps))


agglom.plot()

raise

s, u, e = self.sampleBasin(newbasins[0])



rsets, rCDFs = self.repGraph.getReplicaMinimaCDFs(s+u)

agg = AgglomerativeCDFClustering(rCDFs)

raise

###############################################################################
# Finding sampling region
###############################################################################

Esplit, replicas = self.getBasinSamples(basin)

if len(replicas) == 0:
    minima, runs, paths, replicas = self.sampleBasin(basin, nsamples=1)

if Esplit is None:
    replica = max(replicas, key=lambda rep: rep.energy)
    path = min(basinGraph.genPaths(basin, replica),
               key=lambda p: p[0].energy)
    run = repGraph.pathtoRun(path)
    Esplit = findRunSplit(run)[0]
    Esplit = -127.
    repGraph.insertEnergySplit(Esplit)
    replicas = list(basinGraph.genConnectedReplicas(basin, Esplit=Esplit))

while(len(replicas) < nsamples):
    self.sampleBasin(basin, nsamples=nsamples-len(replicas))
    replicas = list(basinGraph.genConnectedReplicas(basin, Esplit=Esplit))

###############################################################################
# Sampling region
###############################################################################

if len(replicas) > nsamples:
    replicas = sample(replicas, nsamples)

for rep in replicas:
    for i in xrange(nsamples):
        self.sampleReplica(rep)

###############################################################################
# Check basins
###############################################################################

repCDFs = map(self.repGraph.getMinimaCDF, replicas)
sig = AndersonDarling.compareDistributions(repCDFs)[0]
agglom = AgglomerativeCDFClustering(repCDFs)
anyMingtEsplit = any((m[0].energy > Esplit
                      for m in basinGraph.genConnectedMinima(basin)))

if sig < self.reject_sig or anyMingtEsplit:
    ps, clusteri, clusters = agglom.getMaxLikelihood()
    # Add new basins
    newbasins = [self.basinGraph.SuperBasin([replicas[i] for i in ind])
                 for p, ind in izip(ps, clusteri) if  p > self.accept_sig]
    [self.basinGraph.add_edge(basin, b, Esplit=Esplit) for b in newbasins]

    if any(p <= self.accept_sig for p in ps):
        pass
        # insert energy split
elif sig > self.accept_sig:
    self.basinGraph.addtoBasin(basin, replicas)
else:
    ps, clusteri, clusters = agglom.getMaxClusters(self.accept_sig)
    retest = filter(lambda p: len(p[0]) == 1 or p[1] < self.accept_sig,
                    izip(clusteri, ps))
















raise

# Find next Esplit to sample at
minreplica = min(basin.replicas, key=lambda r: r.energy)
Esplits = repGraph.Esplits
i = Esplits.searchsorted(minreplica.energy)

if i==0:
    # Lower than all the thresholds
    try:
        maxrep = max(basinGraph.genConnectedReplicas(basin),
                     key=lambda rep: rep.energy)
        path = min(basinGraph.genPaths(basin, maxrep),
                   key=lambda p: p[0].energy)
        run = repGraph.pathtoRun(path)

        Esplit = findRunSplit(run)[0]
        run1s, run2s, replicas1 = repGraph.insertEnergySplit(Esplit)

        replicas = list(basinGraph.genConnectedReplicas(basin,
                                                        Ethresh=Esplit))
    except ValueError:
        # If there aren't any connected replicas, sample one.
        replicas = []
        #minima, runs, paths, replicas = driver.sampleMinima(replica, nsamples=1)
        #maxrep = max(replicas, key=lambda rep: rep.energy)
        #path = min(basinGraph.genPaths(basin, maxrep), key=lambda p: p[0].energy)
        #run = repGraph.pathtoRun(path)
if i > Esplits.size:
    replicas = list(basinGraph.genConnectedReplicas(basin,
                                                    Ethresh=Esplits[i-1]))

print replicas
print repGraph.Esplits


raise
###############################################################################

# Find daughter basins:

## Need to implement this
if Esplit is None:
    Esplit = basinGraph.getEsplit

# Find replicas connected to basin in above energy split
replicas = basinGraph.getConnectedReplicas(basin, Esplit=Esplit)

# If there's not enough sample some more
while(len(replicas) < nsamples):
    driver.sampleBasin(basin, nsamples=nsamples-len(replicas))
    replicas = basinGraph.getConnectedReplicas(basin, Esplit=Esplit)

if len(replicas) > nsamples:
    replicas = sample(replicas, nsamples)

driver.sampleMinima(replicas)

repCDFs = map(driver.repGraph.getMinimaCDF, replicas)


sig = AndersonDarling.compareDistributions(repCDFs)[0]
agglom = AgglomerativeCDFClustering()

anyMingtEsplit = any((m[0].energy > Esplit
                      for m in basinGraph.genConnectedMinima))

if sig < reject or anyMingtEsplit:
    ps, clusteri, clusters = agglom.getMaxLikelihood()
    newbasins = [Superbasin([replicas[i] for i in ind])
                 for ind in clusteri]
elif sig > target:
    newbasin = basin + Superbasin(replicas)
else:
    ps, clusteri, clusters = agglom.getMinClusters()
    # resample clusters












###############################################################################

raise Exception

self=repGraph
Emax = run.Emax
nlive= run.nlive
parent=run.parent
child=run.child
stored=run.stored
configs=run.configs
stepsizes=runs.stepsizes

Esplits = np.array([500,-10,-100,-120,-140])

i1 = Esplits.searchsorted(Emax[0], side='left')
i2 = Esplits.searchsorted(Emax[-1], side='left')

maxmin, f = max(basinGraph.genConnectedMinima(basin), key=lambda mf: mf[0].energy)
path = min(basinGraph.genPaths(basin, maxmin), key=lambda p: p[0].energy)
run = repGraph.pathtoRun(path)

Esplits = repGraph.Esplits

if Esplits is not None:
    i = Esplits.searchsorted(run.Emax[0])


run1, run2, rep = repGraph.splitRun(run, Esplit)

Esplit, i = findRunSplit(run)




run2, run3, rep = repGraph.splitRun(run1, -100)

self = repGraph

raise

run1s, run2s, replicas1 = repGraph.insertEnergySplit(Esplit)

raise

Esplits = repGraph.Esplits
i = Esplits.searchsorted(run.Emax[-1])

if i < Esplits.size:
    run1, run2, rep = repGraph.splitRun(run, Esplits[i])[0]

Esplit, i = findRunSplit(run)



for rep in replicas1:
    driver.sampleMinima(rep, nsamples=10)

agglom1 = AgglomerativeCDFClustering(map(driver.repGraph.getMinimaCDF, replicas1))
agglom1.plot()

ps, clusteri1, clusters1 = agglom1.getMaxLikelihood()
print AndersonDarling.compareDistributions(map(driver.repGraph.getMinimaCDF, replicas1))

raise
run1s, run2s, replicas2 = repGraph.insertEnergySplit(-90)


for rep in replicas2:
    driver.sampleMinima(rep, nsamples=5)


CDFs = map(driver.repGraph.getMinimaCDF, replicas2)

agglom = AgglomerativeCDFClustering(CDFs)

fig, axes = agglom.plot()
fig.tight_layout()
fig.subplots_adjust(hspace=0.05)




globalH = driver.repGraph.getMinimaCDF(replica)

AndersonDarling.compareDistributions(map(driver.repGraph.getMinimaCDF, replicas1))

for c in agglom1.cdfs:
    c.plot()




###############################################################################





























raise


sig, replicas = driver.sampleBasinBranches(basin, replicas=replicas)

replicas = []
sig, replicas = driver.sampleBasinBranches(basin, -90, nsamples=10)

CDFs = map(driver.repGraph.getMinimaCDF, replicas)
H = sum(CDFs[1:], CDFs[0])

sameCDF = [c for c in CDFs if
           AndersonDarling.compareDistributions([c,H])[0]>driver.reject_sig]
H1 = sum(sameCDF[1:], sameCDF[0])

differentCDF = [c for c in CDFs if
           AndersonDarling.compareDistributions([c,H])[0]<driver.reject_sig]

[c.plot() for c in sameCDF]
[c.plot(c='k',ls=':') for c in differentCDF]

H1.plot(lw=5)

#sig, replicas = driver.sampleBasinBranches(basin, -110, replicas=replicas)




allreplicas = repGraph.replicas()









































































raise

print blo, bhi

CDFs = {}
for rep in newreplicas:
    minf = repGraph.genConnectedMinima(rep)
    Es, fs = zip(*((m.energy, f) for m,f in minf))
    CDFs[rep] = CDF(Es, fs)
    CDFs[rep].plot()

raise



from itertools import groupby

minima = repGraph.minima()
grouped = dict((k,list(v)) for k,v in
                groupby(minima, key=lambda m: np.round(m.energy, 2)))

for E, mins in grouped.iteritems():
    dists = []
    for m1 in mins:
        for m2 in mins:
            dists.append(compare.align(m1, m2)[0])
    print E, dists


r = repGraph.getCombinedRun(replica)



repGraph.plot(arrows=False, with_labels=False, node_size=0)

pos = nx.nx_agraph.graphviz_layout(repGraph.graph, prog='dot')
pos = dict((r, (p[0], np.clip(r.energy, None, -100)))
                        for r,p in pos.iteritems())

nx.draw(repGraph.graph, pos, arrows=True, with_labels=False, node_size=10)


raise

run1, run2 = runs[:2]


basin = basinGraph.repdict[replica]

m, f = max(basinGraph.genConnectedMinima(basin), key=lambda m: m[0].energy)

paths = basinGraph.findPaths(basin, m)

runs = map(repGraph.pathtoRun, paths)
run = reduce(combineRuns, runs)

Esplit = findRunSplit(run, 0.5)[0]

runs = [run for run in repGraph.getConnectedRuns(replica)
        if len(run.Emax) > 1 and run.Emax[0] > Esplit > run.Emax[-1]]

for r in runs:
    r1, r2, rep = repGraph.splitRun(r, Esplit, replace=True)


print paths
print runs


raise

Esplit = findRunSplit(run1, 0.5)[0]

print len(repGraph.getConnectedRuns(replica))

runs = [run for run in repGraph.getConnectedRuns(replica)
        if len(run.Emax) > 1 and run.Emax[0] > Esplit > run.Emax[-1]]

print len(runs)

for r in runs:
    r1, r2, rep = repGraph.splitRun(r, Esplit, replace=False)



for r in runs:
    r1, r2, rep = repGraph.splitRun(r, Esplit, replace=True)


print len(repGraph.getConnectedRuns(replica))



repGraph.draw()











driver.basinGraph.update()
basins = driver.basinGraph.basins()
basin = driver.basinGraph.get_lowest_basin()
gbasin = basins[-1]
minDistr = driver.basinGraph.getBasinMinimaDistribution(gbasin)

cdf = CDF(*zip(*[(m.energy, f) for m,f in minDistr]))

cdf1 = CDF(np.ones(10))
cdf2 = CDF([2])

raise
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

























