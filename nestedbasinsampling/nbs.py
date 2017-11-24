# -*- coding: utf-8 -*-


from bisect import bisect, insort
from itertools import chain, izip
from random import choice, sample

import numpy as np

import networkx as nx

from pele.optimize import lbfgs_cpp

from nestedbasinsampling.sampling.stats import CDF, AndersonDarling, AgglomerativeCDFClustering
from nestedbasinsampling.optimize import RecordMinimization
from nestedbasinsampling.storage import Minimum, Replica, Run, Database
from nestedbasinsampling.graphs import ReplicaGraph, BasinGraph, SuperBasin, FunctionGraph
from nestedbasinsampling.nestedsampling import findRunSplit, combineRuns
from nestedbasinsampling.utils.errors import SamplingError
from nestedbasinsampling.nestedbasinsystem import NestedBasinSystem

class NestedBasinSampling(object):
    """ This class controls overall the sampling of the disconnectivity graph

    Attributes
    ----------

    pot : BasePotential
        The potential being explored
    nopt : callable
        Performs a nested optimisation run
    sampler : BaseSampler
        samples the basin of a replica
    repGraph : ReplicaGraph
        stores the results of the sampling
    basinGraph : BasinGraph
        aggregates the results of the sampling

    quench : callable, optional
        Performs a standard minimization

    Methods
    -------
    """

    sampleMinStr = ("NBS > Sampling minima, Ecut = {:10.5g}, "+
                    "minE = {:10.5g}, logF = {:8.4g}")
    branchStr = ("NBS > Sampling branches, basinE = {:8.4g}, "+
                 "maxChildE = {:8.4g}, n = {:3d}")
    danglingStr = ("NBS > Sampling dangling basins, basinE = {:8.4g}, "+
                   "danglingE = {:8.4g}, n = {:3d}")
    basinBottomStr = ("NBS > Sampling basin, basinE = {:8.4g}, "
                      "bottomE = {:8.4g}, newBasinE = {:8.4g}")

    def __init__(self, pot, nopt, sampler,
                 repGraph=None, basinGraph=None, saveConfigs=False,
                 quench=None, savenopt=1, savequench=0, useQuench=False,
                 percentile=1e-2, reject_sig=1e-3, accept_sig=0.5,
                 splitBranch=0.5, func=None, minstep=1e-10,
                 nsamples=10, nbranch=None,
                 detectStep=None, fixConstraint=None, random_config=None,
                 debug=False, iprint=1):

        self.pot = pot
        self.sampler = sampler

        # Optimizers
        self.useQuench = useQuench
        self.saveConfigs = saveConfigs
        self._nopt = nopt
        self.nopt = RecordMinimization(self.pot, minimizer=self._nopt)
        self.savenopt = savenopt
        self._quench = quench
        self.quench = RecordMinimization(self.pot, minimizer=self._quench)
        self.savequench = savequench

        # Other stuff
        self.random_config = random_config
        self.detectStep = detectStep
        self.fixConstraint = fixConstraint
        self.func = func

        self.minstep = minstep

        self.percentile = percentile
        self.reject_sig = reject_sig
        self.accept_sig = accept_sig

        self.splitBranch = splitBranch
        self.nsamples = nsamples
        self.nbranch = nsamples if nbranch is None else nbranch

        self.repGraph = ( ReplicaGraph(Database()) if repGraph is None
                          else repGraph )
        self.basinGraph = ( BasinGraph(self.repGraph)
                            if basinGraph is None else basinGraph)

        self.Run = self.repGraph.Run
        self.Replica = self.repGraph.Replica
        self.Minimum = self.repGraph.Minimum
        self.Path = self.repGraph.Path

        self.debug = debug
        self.iprint = iprint
        self.nsampleReplica = 0

    def globalReplicaBasin(self):
        """
        """
        reps = (rep for rep in self.repGraph.replicas() if rep.coords is None)
        try:
            replica, = reps
            basin = self.basinGraph.repdict[replica]
        except ValueError:
            replica = self.Replica(np.inf, None)
            basin = self.basinGraph.SuperBasin([replica])
        return replica, basin

    def addMinimum(self, energy, coords):
        m = self.repGraph.Minimum(energy, coords)
        basin = self.basinGraph.newMinimum(m)
        return m, basin

    def createParentBasin(self, basin, basinE, stepsize=None):
        replica = choice(list(basin.replicas))
        res = self.sampler(basinE, replica.coords, stepsize=stepsize)

        basinreplica = self.repGraph.Replica(
            basinE, res.coords, res.stepsize)
        newreplica = self.repGraph.Replica(
            res.energy, res.coords, res.stepsize)
        self.repGraph.Run([res.energy],[1],basinreplica, newreplica)
        self.repGraph.Path(basinE, newreplica, replica)

        newbasin = self.basinGraph.SuperBasin([basinreplica])

        return newbasin

    def getClosestBasin(self, basin, Ecut):
        """
        """
        subgraph = self.basinGraph.graph.subgraph(chain(
            [basin], self.basinGraph.genPreceedingBasins(basin),
            self.basinGraph.genSucceedingBasins(basin)))
        edges = [(b1, b2) for b1, edges in subgraph.adjacency_iter()
                 for b2 in edges if b1.energy >= Ecut > b2.energy]
        basins = [b1 for b1, b2 in edges
                  if b1.energy >= Ecut >=
                  self.basinGraph.getEnergyPercentile(b1, self.percentile)]
        if basins:
            newbasin = max(basins,
                        key=lambda b: self.basinGraph.basinOutDegree(b))
        else:
            edge = min(edges,
                       key=lambda b2: min(self.basinGraph.getMinimaSet(b2[0])))
            newbasin, newrep = self.basinGraph.splitBasins(edge[::-1], Ecut)

        return newbasin

    def sampleReplica(self, replica, stepsize=None,
                      nsteps=None, useQuench=None, saveConfigs=None):
        """
        """
        # Setting whether to quench or nested optimisation
        useQuench = self.useQuench if useQuench is None else useQuench
        saveConfigs = self.saveConfigs if saveConfigs is None else saveConfigs
        nsave = self.savequench if useQuench else self.savenopt

        if isinstance(replica, Minimum):
            return replica, None, None

        # Determining the stepsize to sample with
        if stepsize is None:
            if getattr(replica, 'stepsize', None) is None:
                if self.detectStep is None:
                    stepsize = self.sampler.stepsize
                elif replica.coords is not None:
                    stepsize, coords, Enew = self.detectStep(replica.energy,
                                                            replica.coords)[:3]
                    if self.pot.getEnergy(replica.coords) < Enew:
                        stepsize = self.detectStep(replica.energy, coords,
                                                   stepsize=stepsize)[0]
                    if stepsize < self.minstep:
                        raise SamplingError(
                           "stepsize too small stepsize = {:8.4g}".format(
                               stepsize))

                    replica.stepsize = stepsize
                else:
                    stepsize = self.sampler.stepsize
            else:
                stepsize = replica.stepsize

        if stepsize < self.minstep:
            raise SamplingError(
                "stepsize too small stepsize = {:8.4g}".format(stepsize))

        Ecut = replica.energy

        # Generating starting location
        if replica.coords is None: # Sample randomly
            if self.random_config is None:
                raise NotImplementedError("random_config not defined")
            else:
                coords = self.random_config()
                E = self.pot.getEnergy(coords)
        else:
            coords = replica.coords
            try:
                sampleres = self.sampler(Ecut, coords, stepsize=stepsize)
                coords = sampleres.coords
                E = sampleres.energy
                stepsize = sampleres.stepsize
            except SamplingError as e:
                if self.debug:
                    print "NBS > sampleReplica warning:",  e
                    print "NBS > stepsize = {:8.5g}".format(stepsize)

                if self.detectStep is not None:
                    naccept = e.kwargs['naccept']
                    nreject = e.kwargs['nreject']
                    stepsize = self.detectStep.adjust_stepsize(
                        naccept, nreject, stepsize)

                return self.sampleReplica(replica, stepsize, nsteps)

        start = self.Replica(E, coords, stepsize=stepsize)

        if useQuench:
            res = self.quench(coords)
        else:
            res = self.nopt(coords, stepsize=stepsize)

        # Saving the results
        child, m, run, path = self.repGraph.addMinimisation(
            replica, start, res, useQuench=useQuench,
            saveConfigs=saveConfigs, nsave=nsave)

        newbasin = self.basinGraph.newMinimum(m)

        self.nsampleReplica += 1
        if self.debug and (self.nsampleReplica % self.iprint) == 0:
            print self.sampleMinStr.format(Ecut, res.energy,
                                           -res.nsteps*np.log(2))
        return m, newbasin, run, path


    def sampleBasin(self, basin, useQuench=None,
                    nsamples=None, stepsize=None, nsteps=None):
        """
        """
        nsamples = self.nsamples if nsamples is None else nsamples
        ms, newbasins, runs, paths = zip(*
            [self.sampleReplica(choice(list(basin.replicas)), nsteps=nsteps,
                                useQuench=useQuench, stepsize=stepsize)
            for i in xrange(nsamples)])

        return ms, newbasins, runs, paths

    def sampleAboveReplica(self, replica, Ethresh, **kwargs):
        """
        """
        # Generating random starting point uniformly below Ethresh
        res = self.sampler(Ethresh, replica.coords, nsteps=30, nadapt=30)

        # Creating new replica, path and basin associated with this point
        newreplica = self.repGraph.Replica(
            res.energy, res.coords, stepsize=res.stepsize)
        self.repGraph.Path(newreplica.energy, newreplica, replica)
        newbasin = self.basinGraph.SuperBasin([newreplica])

        # Sampling the new basin
        self.sampleBasin(newbasin, **kwargs)
        return newbasin

    def groupBasins(self, basins):
        """
        """
        basinCDFs = [reduce(lambda x,y:x+y,
                            map(self.repGraph.getMinimaCDF, basin.replicas))
                     for basin in basins]

        agglom = AgglomerativeCDFClustering(basinCDFs)
        # The likelihood of all the basins being the same
        sig = agglom.significance

        if sig > self.accept_sig:
            newbasins = [self.basinGraph.mergeBasins(basins)]
        elif sig < self.reject_sig:
            ps, clusteri, clusters = agglom.getMaxLikelihood()
            newbasins = [
                self.basinGraph.mergeBasins([children[i] for i in ind])
                for p, ind in izip(ps, clusteri) if  p > self.accept_sig]
        else:
            newbasins = []

        return newbasins, agglom

    def compareBasinPair(self, basin1, basin2, useQuench=None, useAll=False):

        useQuench = self.useQuench if useQuench is None else useQuench
        runs = False if useAll else not useQuench
        paths = False if useAll else useQuench

        sig, cdf1, cdf2 = self.basinGraph.compareBasinPair(
            basin1, basin2, runs=runs, paths=paths)

        recalc = False
        if cdf1 is None or cdf1.n < self.nsamples:
            self.sampleBasin(basin1, useQuench=useQuench)
            recalc = True
        if cdf2 is None or cdf2.n < self.nsamples:
            self.sampleBasin(basin2, useQuench=useQuench)
            recalc = True

        if sig is None or recalc:
            sig, cdf1, cdf2 = self.basinGraph.compareBasinPair(
                basin1, basin2, runs=runs, paths=paths)

        while self.accept_sig > sig > self.reject_sig:
            c = cmp(cdf1.n, cdf2.n)
            if c <= 0:
                self.sampleBasin(basin1, nsamples=1)
            if c >= 0:
                self.sampleBasin(basin2, nsamples=1)

            sig, cdf1, cdf2 = self.basinGraph.compareBasinPair(
                basin1, basin2, runs=runs, paths=paths)

        return sig, cdf1, cdf2

    def classifyBasin(self, basin):
        pass

    def findBasinBranches(self, basin, useQuench=None, nsamples=1):
        """
        """

        useQuench = self.useQuench if useQuench is None else useQuench
        runs = not useQuench
        paths = useQuench

        children = []
        newchildren = self.basinGraph.graph.successors(basin)

        if self.debug:
            print self.branchStr.format(
                basin.energy, max(newchildren).energy, len(newchildren))

        while len(newchildren) > 1:

            # If we aren't generating any new children basins stop
            if set(children) == set(newchildren):
                break
            else:
                children = newchildren

            # bisecting the branches to the children basins
            for child in children:
                run = self.basinGraph.getConnectingRun(basin, child)
                Ecut = findRunSplit(run, self.splitBranch)[0]
                self.getClosestBasin(child, Ecut)

            # Sampling the 'dangling' basins
            dangling = sorted(
                b for b in self.basinGraph.graph.successors(basin)
                if self.basinGraph.basinOutDegree(
                    b, runs=runs, paths=paths) < self.nsamples)

            while dangling:

                for b in dangling:
                    if self.debug:
                        print self.danglingStr.format(
                            basin.energy, b.energy, len(dangling))
                    driver.sampleBasin(b, nsamples=nsamples)

                dangling = sorted(
                    b for b in self.basinGraph.graph.successors(basin)
                    if self.basinGraph.basinOutDegree(
                        b, runs=runs, paths=paths) < self.nsamples)

            newchildren = self.basinGraph.graph.successors(basin)

            if self.debug and children:
                print self.branchStr.format(
                    basin.energy, max(children).energy, len(children))

    def findBasinBottom(self, basin, idiff=1, useQuench=None):
        """
        """
        useQuench = self.useQuench if useQuench is None else useQuench

        # Updating graph to include minima etc.
        #self.basinGraph.updateGraph()
        bottom = basin
        bottomE = self.basinGraph.getEnergyPercentile(bottom, self.percentile)

        #Getting list of replicas connecting basin to the highest min
        nextbasin = max(self.basinGraph.genSucceedingBasins(basin))
        if nextbasin.energy > bottomE:
            return [basin,nextbasin]

        replicas = self.basinGraph.genConnectingReplicas(basin, nextbasin).next()

        path = self.repGraph.replicastoPath(replicas)
        Es = path.energies
        imin, imax = 0, Es.size - 1
        basins = [basin,nextbasin]

        while imax - imin > idiff:
            # Binary search along the energies
            itest = (imax-imin)/2 + imin
            Ecut = Es[itest]
            newbasin = self.getClosestBasin(basin, Ecut)
            if newbasin.energy > bottomE:
                self.basinGraph.removeBasin(newbasin)
                break
            #self.basinGraph.updateGraph()
            insort(basins, newbasin)

            if self.debug:
                print self.basinBottomStr.format(bottom.energy,
                                                 bottomE, newbasin.energy)

            sig, cdf1, cdf2 = self.compareBasinPair(basin, newbasin)

            print 'sig', sig

            if sig > self.accept_sig:
                bottom = newbasin
                imin = itest
            else:
                imax = itest

            bottomE = self.basinGraph.getEnergyPercentile(bottom,
                                                          self.percentile)

        basins.sort(key=lambda b: -b.energy)
        cdfs = (self.basinGraph.getMinimaCDF(
            b, runs=not useQuench,paths=useQuench) for b in basins)
        basins, cdfs = izip(*((basin, cdf) for basin, cdf in zip(basins, cdfs)
                             if cdf is not None))
        grouped = []
        i0 = 0
        for i1 in xrange(1,len(cdfs)):
            if i1>i0:
                sig = AndersonDarling.compareDistributions(cdfs[i0:i1+1])[0]
                if sig < self.accept_sig:
                    grouped.append(basins[i0:i1])
                    i0 = i1
        grouped.append(basins[i0:])

        newbasins = [self.basinGraph.mergeBasins(bs) for bs in grouped]
        #self.basinGraph.updateGraph()
        return newbasins

    def compareBasins(self, basins):
        cdfs = [self.basinGraph.getMinimaCDF(basin) for basin in basins]
        return cdfs

    @property
    def funcGraph(self):
        """
        """
        if not hasattr(self, '_funcGraph'):
            self.evalFunc()
        elif not nx.is_isomorphic(self.basinGraph.graph,
                                  self._funcGraph.graph):
            self.evalFunc()

        return self._funcGraph

    def evalFunc(self):
        """
        """
        self._funcGraph = FunctionGraph(basinGraph, func)
        f, f2, fstd, df = self.funcGraph()
        return f, fstd, self._funcGraph

    def calcVarianceReduction(self):
        """
        """
        fgraph = self.funcGraph
        basins = fgraph.graph.nodes()
        f = fgraph.integral
        dfbasins = sorted(((fgraph.graph.node[b]['dsvarf']/f/f).sum(), b)
                        for b in fgraph.graph.nodes())
        dfs, basins = zip(*dfbasins)
        return dfs, basins


###############################################################################
###################################SETUP#######################################
###############################################################################

from pele.systems import LJCluster
from nestedbasinsampling.thermodynamics import heat_capacity_func, partition_func

import seaborn as sns
import matplotlib.pyplot as plt
from plottingfuncs.plotting import ax3d

natoms = 31
system = LJCluster(natoms)
radius =  float(natoms) ** (1. / 3)

acc_ratio = 0.25

nopt_kw = dict(
    tol=1e-1, alternate_stop_criterion=None, events=None, iprint=-1,
    maxsteps=10000, stepsize=0.1, debug=True, quench=lbfgs_cpp,
    quenchtol=1e-6, quench_kw=dict(nsteps=1000))

nopt_sampler_kw = dict(nsteps=15, nadapt=15, maxreject=200, acc_ratio=acc_ratio)

sampler_kw = dict(nsteps=30, nadapt=30, maxreject=100,
                  acc_ratio=acc_ratio, fixConstraint=True)

nbsystem = NestedBasinSystem(
    system, radius=radius, nopt_kw=nopt_kw, nopt_sampler_kw=nopt_sampler_kw,
    sampler_kw=sampler_kw, startseed=1)

pot = nbsystem.get_potential()
nopt = nbsystem.get_nestedoptimizer()
sampler = nbsystem.get_sampler()
quench = nbsystem.get_minimizer()
rand_config = nbsystem.get_random_configuration()
compare = nbsystem.get_compare_structures()
db = nbsystem.get_database(compareMinima=compare, accuracy=1e-2)
repGraph = nbsystem.get_replica_graph(database=db)

Ts = np.logspace(-2,0,50)
Emin = -133.
Cfunc = lambda E: heat_capacity_func(E, Ts, Emin=Emin)
Pfunc = lambda E: partition_func(E, Ts, Emin=Emin)

nbs_kw = dict(useQuench=False, nsamples=5, nbranch=5, splitBranch=0.2,
              accept_sig=0.2, reject_sig=1e-3, debug=True)

driver = NestedBasinSampling(
    pot, nopt, sampler, repGraph, quench=quench,
    random_config=rand_config, func=Cfunc, **nbs_kw)

basinGraph = driver.basinGraph

###############################################################################
###############################################################################
###############################################################################

self = driver
plt.ion()

###### Finding bottom of global basin #########
#replica, basin = driver.globalReplicaBasin()

#newbasins = driver.sampleBasin(basin, nsamples=1)[1]
#driver.findBasinBranches(basin)

if 1:
    gmin = np.array(
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

    m, basin = driver.addMinimum(pot.getEnergy(gmin), gmin)

#nextm = repGraph.minima()[1]
#Ediff = nextm.energy - m.energy


self.sampleAboveReplica(m, -123., nsamples=5)

raise

#Ediff = .2
#minima = set([m])
#Ecut = -121
#
#res = self.sampler(Ecut, m.coords, nsteps=30, nadapt=30, stepsize=0.1)
#Enew, coords, stepsize = res.energy, res.coords, res.stepsize
#newReplica = m
#
#def sampleEnergyThresh(self, basin, Ethresh, **kwargs):
#    replica = min(basin)
#    res = self.sampler(Ethresh, replica.coords, nsteps=30, nadapt=30)
#    newreplica = self.repGraph.Replica(
#        res.energy, res.coords, stepsize=res.stepsize)
#    newbasin = self.basinGraph.SuperBasin([newreplica])
#    basinms, newbasins, run, path = self.sampleBasin(newbasin, **kwargs)
#    return newbasin



while True:
    res = self.sampler(Ecut, m.coords,
                       nsteps=30, nadapt=30, stepsize=stepsize)
    Enew, coords, stepsize = res.energy, res.coords, res.stepsize
    lastReplica = newReplica
    newReplica = self.repGraph.Replica(Enew, coords, stepsize=stepsize)
    self.repGraph.Path(newReplica.energy, newReplica, lastReplica)
    newm, newbasin, run, path = self.sampleReplica(newReplica, useQuench=False)
    if m != newm:
        newbasin1 = self.basinGraph.SuperBasin([lastReplica])
        newbasin2 = self.basinGraph.SuperBasin([newReplica])
        print 'found branch'
        raise
        basinms, newbasins, run, path = self.sampleBasin(newbasin1, useQuench=False, nsamples=10)
        basinms, newbasins, run, path = self.sampleBasin(newbasin2, useQuench=False, nsamples=10)
        while True:
            basinms, newbasins, run, path = self.sampleBasin(newbasin2, useQuench=False,
                                                             nsamples=1)
            if m in basinms:
                break
        print 'connected'
        break
    else:
        lastEcut = Ecut
        Ecut += Ediff

raise


minima.update(ms)

m = ms[0]
basin = newbasin

Ecut = m.energy + Ediff
res = self.sampler(Ecut, m.coords, stepsize=0.1)
Enew, coords, stepsize = res.energy, res.coords, res.stepsize

raise

while True:
    res = self.sampler(Ecut, m.coords,
                       nsteps=100, nadapt=15, stepsize=stepsize)
    Enew, coords, stepsize = res.energy, res.coords, res.stepsize
    newReplica = self.repGraph.Replica(Enew, coords, stepsize=stepsize)
    mnew, mbasin, run, path = self.sampleReplica(newReplica)
    if mnew != m:
        while True:
            m2, mbasin, run, path = self.sampleReplica(newReplica)
            if m2 == mnew:
                break
        print 'found branch'
        break
    else:
        Ecut += Ediff

raise

basins = basinGraph.basins()
basins = [b for b in basinGraph.basins() if basinGraph.graph.out_degree(b) > 5]
[basinGraph.graph.out_degree(b) for b in basinGraph.basins()]


basins = [b for b in basinGraph.basins() if basinGraph.graph.out_degree(b) > 1]
newbasin = max(basins)
driver.findBasinBranches(newbasin)

raise








print 'start'
newbasins = driver.sampleBasin(basin, nsamples=5)[1]

for run in repGraph.runs():
    plt.semilogy(run.Emax-run.Emax.min())

minima = basinGraph.getMinimaSet(basin)
mbasin = min(basinGraph.genSucceedingBasins(basin))
bmin = min(minima)

raise

splitbasin = mbasin

run = basinGraph.getConnectingRun(basin, splitbasin)

Ecut = findRunSplit(run, self.splitBranch)[0]

edge = (splitbasin, basin)
newbasin, newrep = self.basinGraph.splitBasins(edge, Ecut)

res = driver.sampleBasin(newbasin, nsamples=1)


Ts = np.logspace(-2,.3,50)
Emin = -133.
Cfunc = lambda E: heat_capacity_func(E, Ts, Emin=Emin)
Pfunc = lambda E: partition_func(E, Ts, Emin=Emin)

fgraph = FunctionGraph(basinGraph, Cfunc)
Zgraph = FunctionGraph(basinGraph, Pfunc)

plt.plot(Ts, fgraph.integral/Zgraph.integral)

raise

driver.findBasinBranches(basin)

basinGraph.plot()

basins = basinGraph.basins()

basins = [b for b in basins if basinGraph.graph.out_degree(b) > 1]
cdfs = [basinGraph.getMinimaCDF(b) for b in basins]

agglom = AgglomerativeCDFClustering(cdfs)

sig, clusteri, clustercdfs = agglom.getMaxLikelihood()

closebasins = [basins[i] for i in clusteri[-1]]

closecoords = []
for b in closebasins:
    r,= b
    if r.coords is not None:
        closecoords.append(r.coords.copy())

for x in closecoords:
    for y in closecoords:
        print bnb(x, y, niter=100)[0]

x, y, z = closecoords
d, x, y = bnb(x, y, niter=1000)[:3]
fig, ax = ax3d()
ax.scatter(*x.reshape(-1,3).T)
ax.scatter(*y.reshape(-1,3).T, c='r')
#basins = basinGraph.basins()
#basin = max(b for b in basins if basinGraph.graph.out_degree(b) > 1)
#driver.findBasinBranches(basin)

#run graphs/functionGraph.py

Ts = np.logspace(-2,.3,50)
Emin = -133.
Cfunc = lambda E: heat_capacity_func(E, Ts, Emin=Emin)
Pfunc = lambda E: partition_func(E, Ts, Emin=Emin)

fgraph = FunctionGraph(basinGraph, Cfunc)
Zgraph = FunctionGraph(basinGraph, Pfunc)

plt.plot(Ts, fgraph.integral/Zgraph.integral)

b0, b1 = basinGraph.basins()[-2:-4:-1]
edge = fgraph.graph.edge[b0][b1]



if False:

    self = basinGraph

    for b1, edges in self.basinGraph.graph.adjacency_iter():
        for b2, edge in edges.iteritems():
            run = self.basinGraph.getConnectingRun(b1, b2)
            edge['run'] = run
            edge['weights'] = run.calcWeights()













###############################################################################
###############################################################################
###############################################################################

if False:

    coords = np.array(
          [-0.54322426, -0.38203861,  0.52203412,  1.26544146,  0.4464261 ,
           -0.32877187, -0.1064755 , -0.7742898 , -0.4522979 ,  0.40450016,
            0.09264135, -0.98264503,  1.28655638, -0.20185368,  0.60569126,
            0.283552  ,  1.02068545, -0.33608509,  0.36110853,  0.03219396,
            0.09663129, -0.30217412,  0.7273169 ,  0.59385779,  0.82869275,
            0.83867761,  0.64556041,  1.02439098, -0.66292951, -0.40059499,
           -0.56433927,  0.26624146, -0.41242867,  0.31771689, -0.02825336,
            1.1759076 ,  0.43866499, -0.95629744,  0.52934806])
    energy = pot.getEnergy(coords)

    m = repGraph.Minimum(energy, coords)

    dE = 0.5

    Ecut = energy + dE

if False:
    natoms = 31
    gmin = np.array(
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

























