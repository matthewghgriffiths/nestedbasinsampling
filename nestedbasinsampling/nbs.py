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
from nestedbasinsampling.utils import len_iter, SortedCollection
from nestedbasinsampling.utils.errors import SamplingError, GraphError
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

    sampleMinStr = (
        "NBS > Sampling minima,  Ecut = {:8.4g}, minE = {:10.5g}, "+
        "logF = {:8.4g}")
    basinBottomStr = (
        "NBS > Sampling basin, basinE = {:8.4g}, bottomE = {:8.4g}, "+
        "newBasinE = {:8.4g}")
    basinCompareStr = (
        "NBS > Comparing basins, b1E  = {:8.4g}, n1 = {:3d}, "+
        "b2E = {:8.4g}, n2 = {:3d}, sig = {:8.4g}")

    branchStr = (
        "NBS > Sampling branches, basinE = {:8.4g}, "+
        "maxChildE = {:8.4g}, n = {:3d}")
    danglingStr = (
        "NBS > Sampling dangling basins, basinE = {:8.4g}, "+
        "danglingE = {:8.4g}, n = {:3d}")

    def __init__(self, pot, nopt, sampler,
                 repGraph=None, basinGraph=None, saveConfigs=False,
                 quench=None, savenopt=1, savequench=0, useQuench=False,
                 basin_res=1e-3, reject_sig=1e-3, accept_sig=0.5,
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

        if self.savequench:
            self.nopt = RecordMinimization(self.pot, minimizer=self._nopt)
            self.quench = RecordMinimization(self.pot, minimizer=self._quench)
        else:
            self.nopt = self._nopt
            self.quench = self._quench

        # Other stuff
        self.random_config = random_config
        self.detectStep = detectStep
        self.fixConstraint = fixConstraint
        self.func = func

        self.minstep = minstep

        self.basin_res = basin_res
        self.log_res = - np.log(basin_res)
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
            self.basinGraph.tree.add_node(basin)
        return replica, basin

    def addMinimum(self, energy, coords):
        m = self.repGraph.Minimum(energy, coords)
        basin = self.basinGraph.newMinimum(m)
        return m, basin

    def findParentBasin(self, basin, basinE, stepsize=None):
        """
        """
        # Finding where to insert the new basin
        # If a parent basin is found within basin_res then return that basin
        last = basin
        parents = self.basinGraph.graph.predecessors(last)
        if parents:
            while parents:
                parent, = parents
                parentbottom = self.basinGraph.getEnergyFraction(
                    parent, self.basin_res)

                if parent.energy >= basinE > parentbottom:
                    return parent
                elif parent.energy > basinE:
                    self.basinGraph.graph.remove_edge(parent, last)
                    break
                else:
                    parents = self.basinGraph.graph.predecessors(parent)
                    if parents:
                        last = parent
        else:
            parent = None


        replica = min(basin.replicas)
        # Generating random starting point uniformly below basinE
        res = self.sampler(basinE, replica.coords, stepsize=stepsize)

        basinreplica = self.repGraph.Replica(
            basinE, res.coords, res.stepsize)
        newreplica = self.repGraph.Replica(
            res.energy, res.coords, res.stepsize)

        self.repGraph.Run([res.energy],[1],basinreplica, newreplica)
        self.repGraph.Path(basinE, newreplica, replica)

        newbasin = self.basinGraph.SuperBasin([basinreplica])

        if parent is not None:
            self.basinGraph.connectBasins(parent, newbasin)
        self.basinGraph.connectBasins(newbasin, last)

        if basin in self.basinGraph.tree:
            self.basinGraph.tree.add_node(newbasin)
            self.basinGraph.connectTreeBasins(basin, newbasin)

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
            return replica, None, None, self.basinGraph.basin[replica]

        if not useQuench:
            # Determining the stepsize to sample with
            if stepsize is None:
                if getattr(replica, 'stepsize', None) is None:
                    if self.detectStep is None:
                        stepsize = self.sampler.stepsize
                    elif replica.coords is not None:
                        stepsize, coords, Enew = self.detectStep(
                            replica.energy, replica.coords)[:3]
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
        child, m, run, path = self.repGraph.addMinimization(
            replica, start, res, useQuench=useQuench,
            saveConfigs=saveConfigs, nsave=nsave)

        if not useQuench:
            newbasin = self.basinGraph.newMinimum(m)
        else:
            newbasin = None

        self.nsampleReplica += 1
        if self.debug and (self.nsampleReplica % self.iprint) == 0:
            print self.sampleMinStr.format(Ecut, res.energy,
                                           -res.nsteps*np.log(2))
        return m, run, path, newbasin


    def sampleBasin(self, basin, useQuench=None,
                    nsamples=None, stepsize=None, nsteps=None):
        """
        """
        useQuench = self.useQuench if useQuench is None else useQuench
        nsamples = self.nsamples if nsamples is None else nsamples

        ms, runs, paths, newbasins = zip(*
            [self.sampleReplica(choice(list(basin.replicas)), nsteps=nsteps,
                                useQuench=useQuench, stepsize=stepsize)
            for i in xrange(nsamples)])

        if not useQuench:
            for child in newbasins:
                self.basinGraph.connectBasins(basin, child)

        return ms, runs, paths, newbasins

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
                self.basinGraph.mergeBasins([basins[i] for i in ind])
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

        if self.debug:
            print self.basinCompareStr.format(
                basin1.energy, cdf1.n, basin2.energy, cdf2.n, sig)

        while self.accept_sig > sig > self.reject_sig:
            c = cmp(cdf1.n, cdf2.n)
            if c <= 0:
                self.sampleBasin(basin1, nsamples=1)
            if c >= 0:
                self.sampleBasin(basin2, nsamples=1)

            sig, cdf1, cdf2 = self.basinGraph.compareBasinPair(
                basin1, basin2, runs=runs, paths=paths)

            if self.debug:
                print self.basinCompareStr.format(
                    basin1.energy, cdf1.n, basin2.energy, cdf2.n, sig)

        return sig, cdf1, cdf2

    def compareBasins(self, basins, useQuench=None):

        useQuench = self.useQuench if useQuench is None else useQuench
        runs = not useQuench
        paths = useQuench

        sigs = dict()
        cdfs = dict()

        recalc = True
        while recalc:
            recalc = False
            for i, basin1 in enumerate(basins):
                sigs[basin1] = dict()
                for j, basin2 in enumerate(basins[:i]):
                    sig, cdf1, cdf2 = self.basinGraph.compareBasinPair(
                        basin1, basin2, runs=runs, paths=paths)
                    sigs[basin1][basin2] = sig
                    cdfs[basin1] = cdf1
                    cdfs[basin2] = cdf2

                    if cdf1 is None or cdf1.n < self.nsamples:
                        self.sampleBasin(basin1, useQuench=useQuench)
                        recalc = True
                    if cdf2 is None or cdf2.n < self.nsamples:
                        self.sampleBasin(basin2, useQuench=useQuench)
                        recalc = True
                    if not recalc and self.debug:
                        print self.basinCompareStr.format(
                            basin1.energy, cdf1.n, basin2.energy, cdf2.n, sig)


        while any(self.accept_sig > sig > self.reject_sig
                  for val in sigs.itervalues() for sig in val.itervalues()):
            for i, basin1 in enumerate(basins):
                for j, basin2 in enumerate(basins[:i]):
                    cdf1 = cdfs[basin1]
                    cdf2 = cdfs[basin2]
                    c = cmp(cdf1.n, cdf2.n)
                    if c <= 0:
                        self.sampleBasin(
                            basin1, nsamples=1, useQuench=useQuench)
                    if c >= 0:
                        self.sampleBasin(
                            basin2, nsamples=1, useQuench=useQuench)

                    sig, cdf1, cdf2 = self.basinGraph.compareBasinPair(
                        basin1, basin2, runs=runs, paths=paths)

                    sigs[basin1][basin2] = sig
                    cdfs[basin1] = cdf1
                    cdfs[basin2] = cdf2

                    if self.debug:
                        print self.basinCompareStr.format(
                            basin1.energy, cdf1.n, basin2.energy, cdf2.n, sig)

        return sigs, cdfs

    def classifyBasin(self, basin):
        pass

    def findNextBasin(self, basin, useQuench=True):
        """
        """

        if self.debug:
            print "NBS > findNextBasin, energy = {:10.5g}".format(
                basin.energy)
        try:
            childruns = (
                (child, self.basinGraph.connectingRun(basin, child))
                for child in self.basinGraph.graph.successors_iter(basin))
            child, run = max(
                (child, run) for child, run in childruns if run.nlive.size > 1)
        except ValueError:
            self.sampleBasin(basin, useQuench=False, nsamples=1)
            child = max(self.basinGraph.graph.successors_iter(basin))
            run = self.basinGraph.connectingRun(basin, child)

        logX = run.log_frac
        log_res = self.log_res

        toplogX = 1.
        bottomlogX = logX[-1]
        basins = SortedCollection([child, basin])
        basinlogs = dict([(child, bottomlogX), (basin, toplogX)])

        while toplogX - bottomlogX > log_res:
            newlogX = toplogX - (toplogX - bottomlogX)*self.splitBranch
            Ecut = run.frac_energy(newlogX, True)

            newbasin, newrep = self.basinGraph.splitBasins(basins, Ecut)
            basins.add(newbasin)
            basinlogs[newbasin] = newlogX

            for testbasin in basins[::-1]:
                sig, cdf1, cdf2 = self.compareBasinPair(
                    basin, testbasin, useQuench=useQuench)

                if sig > self.accept_sig:
                    toplogX = basinlogs[testbasin]
                else:
                    bottomlogX = basinlogs[testbasin]
                    break

        basins = basins[::-1]
        cdfs = (self.basinGraph.getMinBasinCDF(b) for b in basins)
        basins, cdfs = izip(*((basin, cdf) for basin, cdf in zip(basins, cdfs)
                             if cdf is not None and cdf.n > 1))

        #Grouping basins found
        grouped = []
        i0 = 0
        for i1 in xrange(1,len(cdfs)):
            if i1>i0:
                sig = AndersonDarling.compareDistributions(cdfs[i0:i1+1])[0]
                if sig < self.accept_sig:
                    grouped.append(basins[i0:i1])
                    i0 = i1
        grouped.extend([b] for b in basins[i0:])

        newbasins = [self.basinGraph.mergeBasins(bs) for bs in grouped]

        # adding newbasins to tree
        self.basinGraph.tree.add_node(newbasins[0])
        self.basinGraph.tree.add_node(newbasins[1])
        self.basinGraph.connectTreeBasins(newbasins[0], newbasins[1])

        return newbasins

    def findConnectingBasin(self, minbasin, useQuench=None):
        """
        """
        useQuench = self.useQuench if useQuench is None else useQuench

        if minbasin not in self.basinGraph.tree:
            self.basinGraph.tree.add_node(minbasin)

        if self.debug:
            print "NBS > findConnectingBasin, energy = {:10.5g}".format(
                minbasin.energy)

        gbasin = max(self.basinGraph.tree)
        basin = None

        # If minbasin is already connected to the tree via a run
        if self.basinGraph.nConnectingRuns(gbasin, minbasin):
            parent = gbasin
            successors = [
                b for b in self.basinGraph.tree.successors_iter(parent)
                if self.basinGraph.nConnectingRuns(b, minbasin)]
            while successors:
                parent = max(successors, key=lambda b:
                    self.basinGraph.nConnectingRuns(b, minbasin))
                successors = [
                    b for b in self.basinGraph.tree.successors_iter(parent)
                    if self.basinGraph.nConnectingRuns(b, minbasin)]
            self.basinGraph.connectTreeBasins(parent, minbasin)
            basin = parent

        in_edges = self.basinGraph.tree.in_edges(minbasin)
        while all(self.basinGraph.nConnectingRuns(b1, b2) == 0
                  for b1, b2 in in_edges):

            # Finding the basins to compare against
            if in_edges:
                (parent, _), = in_edges
                # Get the children of the basin that minbasin is connected
                # to that are connected to basin via a run
                testbasins = [
                    b for b in self.basinGraph.tree.successors(parent)
                    if self.basinGraph.nConnectingRuns(parent, b) > 0]
                if not testbasins:
                    self.findNextBasin(parent, useQuench=useQuench)
                    (parent, _), = self.basinGraph.tree.in_edges(minbasin)
                    testbasins = [
                        b for b in self.basinGraph.tree.successors(parent)
                        if self.basinGraph.nConnectingRuns(parent, b) > 0]

                assert minbasin not in testbasins
            else:
                testbasins = [gbasin]

            basinE = np.mean([b.energy for b in testbasins])
            if minbasin.energy >  basinE:
                break

            basin = self.findParentBasin(minbasin, basinE)

            if basin in testbasins:
                connected = [
                    b for b in testbasins
                    if self.basinGraph.nConnectingRuns(b, minbasin) > 0]
                if connected:
                    self.basinGraph.connectTreeBasins(connected[0], minbasin)
                else:
                    self.basinGraph.connectTreeBasins(basin, minbasin)
                    in_edges = self.basinGraph.tree.in_edges(minbasin)
                    continue
                    #basin = self.findParentBasin(minbasin, basinE)
                    if basin in testbasins:
                        raise GraphError(
                            "basin already in tree, basinE = {:10.5g}".format(
                                basin.energy))

            sigs, cdfs = self.compareBasins(
                testbasins+[basin], useQuench=useQuench)
            acceptbasins = [b for b, sig in sigs[basin].iteritems()
                            if sig > self.accept_sig]

            if len(acceptbasins) == 1:
                self.basinGraph.mergeBasins(acceptbasins + [basin])
            elif len(acceptbasins) > 1:
                if any(sigs[b1][b2] < self.reject_sig
                       for i, b1 in enumerate(acceptbasins)
                       for b2 in acceptbasins[:i]):
                    raise GraphError(
                       "Non transitive basin matching, try "+
                       "increasing accept sig,"+
                       "reducing reject sig, or"+
                       "testing the sampler",
                       sigs=sigs)
                elif all(sigs[b1][b2] > self.accept_sig
                         for i, b1 in enumerate(acceptbasins)
                         for b2 in acceptbasins[:i]):
                    self.basinGraph.mergeBasins(acceptbasins + [basin])
            else:
                break

            in_edges = self.basinGraph.tree.in_edges(minbasin)

        return basin

    def findBasinBranches(self, basin,
                          useQuench=None, nbranch=None, nsamples=None):
        """
        """

        useQuench = self.useQuench if useQuench is None else useQuench
        nbranch = self.nbranch if nbranch is None else nbranch
        nsamples = self.nsamples if nsamples is None else nsamples

        children = basinGraph.tree.successors(basin)
        Ecut = np.mean([b.energy for b in children])

        if self.debug:
            print "NBS > finding basin Branches, basinE = {:8.4g}".format(
                basin.energy)

        self.sampleBasin(basin, useQuench=False, nsamples=nbranch)

        connectReps = (
            reps for b in children for reps in
            self.basinGraph.connectingReplicas(basin, b, runs=True))
        connectEdges = (
            self.repGraph.graph[rs[0]][rs[1]] for rs in connectReps)
        connectRuns = set(
            edge['run'] for edge in connectEdges if edge.has_key('run'))

        # Finding complete runs
        edges = (e for r in basin for e in self.repGraph.graph[r].itervalues())
        runs = (edge['run'] for edge in edges if edge.has_key('run'))
        runs = [run for run in runs if run.nlive.size > 1]

        # Creating new basins to sample new runs
        uncategorised = [run for run in runs if run not in connectRuns]
        runreplicas = (
            self.repGraph.connectingReplicas(
                run.parent, run.child, runs=True).next()
            for run in uncategorised)
        splitReps = (
            repGraph.splitReplicas(reps, Ecut) for reps in runreplicas)
        newbasins = [basinGraph.SuperBasin([r]) for r in splitReps]

        allbasins = newbasins + children
        for b in allbasins:
            cdf = basinGraph.getMinBasinCDF(b)
            if cdf.n < driver.nsamples:
                self.sampleBasin(b, useQuench=True, nsamples = driver.nsamples-cdf.n)

        cdfs = [basinGraph.getMinBasinCDF(b) for b in allbasins]

        agglom = AgglomerativeCDFClustering(cdfs)

        sigs, clusteri, clustercdfs = agglom.getMaxClusters(pcutoff=driver.accept_sig)

        mergedbasins = [
            self.basinGraph.mergeBasins([allbasins[i] for i in ind])
            for sig, ind in izip(sigs, clusteri) if sig > driver.accept_sig]
        unmergedbasins = [
            allbasins[i] for sig, ind in izip(sigs, clusteri)
            if sig < driver.accept_sig for i in ind]

        for b in mergedbasins:
            if b not in basinGraph.tree:
                basinGraph.tree.add_node(b)
                basinGraph.connectTreeBasins(basin, b)

        return mergedbasins, unmergedbasins

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
        self._funcGraph = FunctionGraph(basinGraph, self.func)
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
sampler_kw = dict(
    nsteps=15, nadapt=15, maxreject=100, acc_ratio=acc_ratio, fixConstraint=True)
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

nbs_kw = dict(useQuench=True, nsamples=15, splitBranch=0.5,
              accept_sig=0.3, reject_sig=1e-2, debug=True)

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
replica, basin = driver.globalReplicaBasin()

#newbasins = driver.findNextBasin(basin, useQuench=True)

#[basinGraph.getMinBasinCDF(b).plot() for b in newbasins[:-1]]

#Loading 10 lowest structures
with open("lj31_10.pkl", 'rb') as f:
    import cPickle
    mincoords = cPickle.load(f)

    for coords in mincoords:
        driver.addMinimum(pot.getEnergy(coords), coords)

    targetmins = repGraph.minima()[:10]

#minbasin = basinGraph.basin[targetmins[0]]
#parent = driver.findConnectingBasin(minbasin, useQuench=True)

newbasins = driver.findNextBasin(basin, useQuench=True)

m = targetmins[0]
minbasin = basinGraph.basin[m]
driver.findConnectingBasin(minbasin, useQuench=True)

gbasin = max(basinGraph.tree)
basinGraph.nConnectingRuns(gbasin, minbasin)

while True:
    minbasin = basinGraph.basin[m]
    basin, = basinGraph.tree.predecessors_iter(minbasin)

    print basin.energy, basinGraph.nConnectingRuns(gbasin, basin), basinGraph.nConnectingRuns(basin, minbasin)

    ms, runs, paths, newbasins = driver.sampleBasin(basin, useQuench=False, nsamples=1)

    minbasin = basinGraph.basin[m]
    driver.findConnectingBasin(minbasin, useQuench=True)

    if m in ms:
        break

gbasin = max(basinGraph.tree)
merged, unmerged = driver.findBasinBranches(gbasin)

raise

Ecut = -110

m1 = targetmins[0]
res = sampler(Ecut, m1.coords, nadapt=15, nsteps=15)
r1 = repGraph.Replica(Ecut, res.coords, stepsize=res.stepsize)


m2 = targetmins[3]
res = sampler(Ecut, m2.coords, nadapt=15, nsteps=15)
r2 = repGraph.Replica(Ecut, res.coords, stepsize=res.stepsize)

run1s = [driver.sampleReplica(r1, useQuench=False)[1] for _ in xrange(10)]
run2s = [driver.sampleReplica(r2, useQuench=False)[1] for _ in xrange(10)]

run1 = combineAllRuns(run1s)
run2 = combineAllRuns(run2s)
rcom = combineAllRuns(run1s + run2s)

E1s = run1.Emax
F1s = (run1.nlive/(run1.nlive+1.)).cumprod()

E2s = run2.Emax
F2s = (run2.nlive/(run2.nlive+1.)).cumprod()

Ecs = rcom.Emax
Fcs = (rcom.nlive/(rcom.nlive+1.)).cumprod()


Ecut = -115

m1 = targetmins[0]
res = sampler(Ecut, m1.coords, nadapt=15, nsteps=15)
r1 = repGraph.Replica(Ecut, res.coords, stepsize=res.stepsize)


m2 = targetmins[3]
res = sampler(Ecut, m2.coords, nadapt=15, nsteps=15)
r2 = repGraph.Replica(Ecut, res.coords, stepsize=res.stepsize)

run1s = [driver.sampleReplica(r1, useQuench=False)[1] for _ in xrange(10)]
run2s = [driver.sampleReplica(r2, useQuench=False)[1] for _ in xrange(10)]

run1 = combineAllRuns(run1s)
run2 = combineAllRuns(run2s)
rcom = combineAllRuns(run1s + run2s)

E1s = run1.Emax
F1s = (run1.nlive/(run1.nlive+1.)).cumprod()

E2s = run2.Emax
F2s = (run2.nlive/(run2.nlive+1.)).cumprod()

Ecs = rcom.Emax
Fcs = (rcom.nlive/(rcom.nlive+1.)).cumprod()

plt.semilogy(E1s, F1s);plt.semilogy(E2s, F2s);plt.semilogy(Ecs, Fcs)


Ecut = -120

m1 = targetmins[0]
res = sampler(Ecut, m1.coords, nadapt=15, nsteps=15)
r1 = repGraph.Replica(Ecut, res.coords, stepsize=res.stepsize)

m2 = targetmins[3]
res = sampler(Ecut, m2.coords, nadapt=15, nsteps=15)
r2 = repGraph.Replica(Ecut, res.coords, stepsize=res.stepsize)

run1s = [driver.sampleReplica(r1, useQuench=False)[1] for _ in xrange(10)]
run2s = [driver.sampleReplica(r2, useQuench=False)[1] for _ in xrange(10)]

run1 = combineAllRuns(run1s)
run2 = combineAllRuns(run2s)
rcom = combineAllRuns(run1s + run2s)

E1s = run1.Emax
F1s = (run1.nlive/(run1.nlive+1.)).cumprod()

logF1s = (np.log(run1.nlive) - np.log(run1.nlive+1.)).cumsum()
logF12s = (np.log(run1.nlive) - np.log(run1.nlive+2.)).cumsum()
F1dF12s = np.exp(2*logF1s - logF12s)
F1std = np.exp(0.5 * logF12s) * np.sqrt(1. - F1dF12s)

F12s = np.exp(logF12s)

a1 = (F1s**2 - F1s * F12s)/F1std**2
b1 = (1. - F1s)*(F1s - F12s)/F1std**2



w1s = -np.diff(np.r_[1.,F1s])

cdf1 = CDF(E1s, w1s)

E2s = run2.Emax
F2s = (run2.nlive/(run2.nlive+1.)).cumprod()

logF2s = (np.log(run2.nlive) - np.log(run2.nlive+1.)).cumsum()
logF22s = (np.log(run2.nlive) - np.log(run2.nlive+2.)).cumsum()
F2dF22s = np.exp(2*logF2s - logF22s)
F2std = np.exp(0.5 * logF22s) * np.sqrt(1. - F2dF22s)


F22s = np.exp(logF22s)

a2 = (F2s**2 - F2s * F22s)/F2std**2
b2 = (1. - F2s)*(F2s - F22s)/F2std**2

inds = E2s[::-1].searchsorted(E1s)

a2s = np.r_[a2[::-1],0][inds-1]
b2s = np.r_[b2[::-1],1.][inds-1]

from scipy.special import gamma, beta

from scipy.special import gammaln, betaln

BF = gamma(a1+a2s-1) * gamma(b1+b2s - 1)/beta(a1, b1)/beta(a2s, b2s)/gamma(a1+a2s+b1+b2s - 2)
logBF = ( gammaln(a1+a2s-1) + gammaln(b1+b2s - 1)
         - betaln(a1, b1) - betaln(a2s, b2s) - gammaln(a1+a2s+b1+b2s - 2))

p = 1./(1.+np.exp(logBF))

plt.semilogy(E1s, F1s)
plt.semilogy(E1s, F1s+F1std)
plt.semilogy(E1s, F1s-F1std)

plt.semilogy(E2s, F2s)
plt.semilogy(E2s, F2s+F2std)
plt.semilogy(E2s, F2s-F2std)


w2s = -np.diff(np.r_[1.,F2s])

cdf2 = CDF(E2s, w2s)



Ecs = rcom.Emax
Fcs = (rcom.nlive/(rcom.nlive+1.)).cumprod()

wcs = -np.diff(np.r_[1.,Fcs])


inds = E1s[::-1].searchsorted(Ecs, side='left')

w0 = wcs
H0 = Fcs - w0/2.

F = np.r_[F1s[::-1],0.][inds-1]
F -= np.where(E1s[::-1][inds-1] == Ecs,
              np.r_[w1s[::-1],0.][inds-1]/2., 0.)
n = np.arange(1, F.size+1)
A2akN = (n * ( w0 * (F-Fcs)**2 / (H0*(1.-H0)- w0/4.) )).cumsum() * (n-1)/n

(np.arange(1, F.size+1) * ( w0 * (F-Fcs)**2 / (H0*(1.-H0)- w0/4.) )).cumsum()

plt.semilogy(E1s, F1s);plt.semilogy(E2s, F2s);plt.semilogy(Ecs, Fcs)


gbasin = max(basinGraph.tree)

ms, runs, paths, newbasins = driver.sampleBasin(gbasin, useQuench=False, nsamples=10)

run = combineAllRuns(runs)

rlogFs = [np.log(r.nlive).cumsum() - np.log(r.nlive+1.).cumsum()
          for r in runs]

Es = run.Emax
logFs = np.log(run.nlive).cumsum() - np.log(run.nlive+1.).cumsum()

Fs = np.exp(logFs)
ws = - np.diff(np.r_[1,np.exp(logFs)])

ws / (Fs * (1- Fs))

r = runs[0]

inds = r.Emax[::-1].searchsorted(Es[:-1], side='left')


F = np.r_[np.exp(rlogFs[0][::-1]),0.][inds-1]

A = (F - Fs[:-1])**2 * ws[:-1] / (Fs[:-1] * (1- Fs[:-1]))

basin = max(basinGraph.tree)
children = basinGraph.tree.successors(basin)
while len(children) == 1:
    basin, = children
    children = basinGraph.tree.successors(basin)

replicas = [r for b in children for r in b]

Ecut = np.mean([r.energy for r in replicas])

samples =  [[sampler(Ecut, r.coords).energy for _ in xrange(500)]
             for r in replicas]
cdfs = [CDF(sample) for sample in samples]

while basinGraph.nConnectingRuns(gbasin, basin) == 0:
    basin, = basinGraph.tree.predecessors_iter(basin)


branch = list(basinGraph.treeBranch(minbasin))
[b.energy for b in branch]



basin = branch[-1]
driver.sampleBasin(basin, useQuench=False, nsamples=10)

children = basinGraph.tree.successors(basin)
Ecut = np.mean([b.energy for b in children])

connectReps = (reps for b in children
               for reps in basinGraph.connectingReplicas(basin, b, runs=True))
connectEdges = (repGraph.graph[rs[0]][rs[1]] for rs in connectReps)
connectRuns = set(edge['run'] for edge in connectEdges if edge.has_key('run'))

# Finding complete runs
edges = (edge for r in basin for edge in repGraph.graph[r].itervalues())
runs = (edge['run'] for edge in edges if edge.has_key('run'))
runs = [run for run in runs if run.nlive.size > 1]

# Finding basins to sample
uncategorised = [run for run in runs if run not in connectRuns]
runreplicas = [repGraph.connectingReplicas(run.parent, run.child, runs=True).next()
               for run in uncategorised]
splitReps = [repGraph.splitReplicas(reps, Ecut) for reps in runreplicas]
newbasins = [basinGraph.SuperBasin([r]) for r in splitReps]

allbasins = newbasins + children
for b in allbasins:
    cdf = basinGraph.getMinBasinCDF(b)
    cdf.plot()
    if cdf.n < driver.nsamples:
        self.sampleBasin(b, useQuench=True, nsamples = driver.nsamples-cdf.n)

cdfs = [basinGraph.getMinBasinCDF(b) for b in allbasins]

agglom = AgglomerativeCDFClustering(cdfs)

sigs, clusteri, clustercdfs = agglom.getMaxClusters(pcutoff=driver.accept_sig)

mergedbasins = [self.basinGraph.mergeBasins([allbasins[i] for i in ind])
            for sig, ind in izip(sigs, clusteri) if sig > driver.accept_sig]
unmergedbasins = [allbasins[i] for sig, ind in izip(sigs, clusteri)
                  if sig < driver.accept_sig for i in ind]

for b in mergedbasins:
    if b not in basinGraph.tree:
        print b.energy
        basinGraph.tree.add_node(b)
        basinGraph.connectTreeBasins(basin, b)

[[r.energy for r in reps] for reps in runreplicas]

raise
for m in targetmins[6:]:
    minbasin = basinGraph.basin[m]
    driver.findConnectingBasin(minbasin, useQuench=True)
    #break

plt.figure()
basinGraph.draw()



m = targetmins[0]
m = targetmins[3]
minbasin = basinGraph.basin[m]

basins = [b for b in basinGraph.tree.nodes()
          if not any(isinstance(r, Minimum) for r in b)]

basinGraph.draw()

for b1, edges in basinGraph.tree.adjacency_iter():
    for b2, edge in edges.iteritems():
        sig = basinGraph.compareBasinPair(b1, b2, paths=True)[0]
        if self.accept_sig > sig > self.reject_sig:
            self.compareBasinPair(b1, b2)

gbasin = max(basinGraph.tree)
active = set([gbasin])

while active:
    basin = active.pop()
    children = self.basinGraph.tree.successors(basin)
    cdfs = (self.basinGraph.getMinBasinCDF(b) for b in children)
    children, cdfs = zip(*((b, cdf) for b, cdf in izip(children, cdfs)
                           if cdf is not None or cdf.n > 1))
    if len(cdfs) > 1:
        agglom = AgglomerativeCDFClustering(cdfs)
        sigs, inds, cdfs = agglom.getMaxClusters(pcutoff=self.accept_sig)
        print sigs, inds
        children = [
            self.basinGraph.mergeBasins([children[i] for i in ind])
            for sig, ind in izip(sigs, inds) if sig > self.accept_sig]

    active.update(children)



raise

from scipy.special import factorial
from scipy.stats import dirichlet



self=agglom

a = 1

_clusterps = ((self.nodep[F] for F in a) for a in self.state)
clusterps = [[p if p else 1./self.ncdfs for p in ps]
             for ps in _clusterps]
state = self.state

alphas = [np.array([a*p*len(s) for p, s in izip(*elems)])
          for elems in izip(clusterps, state)]
ps = [np.array(map(len, s))*1./self.ncdfs for s in state]
[dirichlet(al).pdf(p) * al.prod() for al, p in izip(alphas, ps)]

def chinese_restaurant(ns):
    ns = np.asanyarray(ns)
    n = ns.sum()
    return factorial(ns-1).prod()/ns.sum()


testbasin = newbasins[0]


useQuench = True

in_edges = self.basinGraph.graph.in_edges(minbasin)


if minbasin not in self.basinGraph.tree:
    self.basinGraph.tree.add_node(minbasin)

parents = self.basinGraph.tree.predecessors(minbasin)

while all(self.basinGraph.nConnectingRuns(b1, minbasin) == 0 for b1 in parents):

    parents = self.basinGraph.tree.predecessors(minbasin)
    if parents:
        parent, = parents
        testbasins = [b for b in self.basinGraph.tree.successors(parent)
                      if self.basinGraph.nConnectingRuns(parent, b) > 0]
        if not testbasins:
            self.findNextBasin(parent, useQuench=useQuench)
            (parent, _), = self.basinGraph.tree.in_edges(minbasin)
            testbasins = [b for b in self.basinGraph.tree.successors(parent)
                          if self.basinGraph.nConnectingRuns(parent, b) > 0]

        assert minbasin not in testbasins
    else:
        testbasins = [max(self.basinGraph.tree.nodes())]

    basinE = np.mean([b.energy for b in testbasins])
    basin = self.findParentBasin(minbasin, basinE)

    if basin in testbasins:
        raise GraphError(
            "basin already in tree, basinE = {:10.5g}".format(basin.energy))

    sigs, cdfs = self.compareBasins(testbasins+[basin], useQuench=useQuench)
    acceptbasins = [b for b, sig in sigs[basin].iteritems()
                    if sig > self.accept_sig]

    if len(acceptbasins) == 1:
        self.basinGraph.mergeBasins(acceptbasins + [basin])
    elif len(acceptbasins) > 1:
        if any(sigs[b1][b2] < self.reject_sig
               for i, b1 in enumerate(acceptbasins)
               for b2 in acceptbasins[:i]):
            raise GraphError(
               "Non transitive basin matching, try increasing accept sig"+
               "and reducing reject sig", sigs=sigs)
        elif all(sigs[b1][b2] > self.accept_sig
                 for i, b1 in enumerate(acceptbasins)
                 for b2 in acceptbasins[:i]):
            self.basinGraph.mergeBasins(acceptbasins + [basin])
    else:
        break


    parents = self.basinGraph.tree.predecessors(minbasin)


parent, = basinGraph.tree.predecessors(minbasin)
parent.energy
self.sampleBasin(parent, useQuench=False)

self.findBasinBottom(parent, useQuench=useQuench)

raise





for i, testbasin in enumerate(basins):
    newbasin = self.findParentBasin(minbasin, testbasin.energy)
    self.sampleBasin(newbasin, useQuench=True)
    sig, cdf1, cdf2 = self.compareBasinPair(testbasin, newbasin, useQuench=True)
    if sig < self.reject_sig:
        break
    elif sig > self.accept_sig:
        self.basinGraph.mergeBasins([testbasin, newbasin])

raise

self.sampleBasin(newbasin, useQuench=False, nsamples=20)

driver.findBasinBottom(newbasin, useQuench=True)



def checkTreeReplicas(self, parent, child):
    newbranch = sorted(set(chain(
        self.treeBranch(parent), self.treeBranch(child))))

    # Ensuring that each replica only has the single correct predecessor
    for r1, r2 in izip(newbranch[1:], newbranch[:-1]):
        if len(self.tree.predecessors(r2)) > 1:
            print r2.energy, len(self.tree.predecessors(r2))
        for pre in self.tree.predecessors(r2):
            if pre != r1:
                print pre.energy, r1.energy, r2.energy
                return True
    else:
        return False

def connectTreeReplicas(self, parent, child):
    newbranch = sorted(set(chain(
        self.treeBranch(parent), self.treeBranch(child))))

    # Ensuring that each replica only has the single correct predecessor
    for r1, r2 in izip(newbranch[1:], newbranch[:-1]):
        if len(self.tree.predecessors(r2)) > 1:
            print r2.energy, len(self.tree.predecessors(r2))
        for pre in self.tree.predecessors(r2):
            if pre != r1:
                print pre.energy, r1.energy, r2.energy, pre.id(), r1.id()
                self.tree.remove_edge(pre, r2)
            if not self.tree.has_edge(r1, r2):
                print r1.energy, r2.energy
                self.tree.add_edge(r1, r2)

for run in repGraph.runs():
    checkTreeReplicas(repGraph, run.parent, run.child)
    connectTreeReplicas(repGraph, run.parent, run.child)


for path in repGraph.paths():
    if checkTreeReplicas(repGraph, path.parent, path.child):
        break

for path in repGraph.paths():
    connectTreeReplicas(repGraph, path.parent, path.child)


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

























