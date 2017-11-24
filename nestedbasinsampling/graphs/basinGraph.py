# -*- coding: utf-8 -*-

from collections import defaultdict
from itertools import chain, izip, groupby
from functools import total_ordering
from math import exp, log, sqrt

import numpy as np
from scipy.special import gammaln
from scipy.integrate import quad
import networkx as nx
import matplotlib.pyplot as plt

from pele.utils.disconnectivity_graph import DisconnectivityGraph

from nestedbasinsampling.storage import Minimum, Replica, Run, Database, TransitionState
from nestedbasinsampling.sampling.stats import AndersonDarling, CDF
from nestedbasinsampling.nestedsampling import combineAllRuns, splitRun
from nestedbasinsampling.utils import iter_minlength
from nestedbasinsampling.utils.sortedcollection import SortedCollection
#from .functionGraph import NumericIntegrator

@total_ordering
class SuperBasin(object):
    """ SuperBasin object.

    A super basin is defined by a single replica. All the states
    accessible via downhill paths from this replica are part of
    the superbasin. The objects of this class are hashable
    so that they can included in networkx graphs.

    Attributes
    ----------
    replicas : frozenset of Replica
        a frozenset of replicas sampled from superbasin
    """
    def __init__(self, replicas=frozenset()):
        self.replicas = frozenset(replicas)
        self.energies = sorted(rep.energy for rep in self.replicas)
        self.energy = self.energies[0]

    def __len__(self):
        return len(self.replicas)

    def __iter__(self):
        return iter(self.replicas)

    def __add__(self, new):
        new = [new] if type(new) is Replica else new
        replicas = self.replicas.union(new)
        return self.__class__(replicas=replicas)

    def __eq__(self, basin):
        return self.replicas == basin.replicas

    def __ne__(self, basin):
        return self.replicas != basin.replicas

    def __gt__(self, basin):
        return self.energy > basin.energy

    def __hash__(self):
        return hash(self.replicas)

    def __str__(self):
        return (
            "SuperBasin([" +
            ", \n            ".join(
                "Replica(energy={:6.4g})".format(r.energy)
                for r in sorted(self.replicas)) + "])")

class BasinGraph(object):
    """ This class joins replicas in the ReplicaClass together
    as a set of super basins.
    """
    def __init__(self, replicaGraph, Emax=None, disconnect_kw={}):
        """
        """
        self.repGraph = replicaGraph
        self.Emax = None
        self._connectgraph = nx.Graph()
        self._disconnectgraph = None
        self.disconnect_kw = disconnect_kw
        self.initialize()

    def initialize(self):
        """
        """
        self.graph = nx.DiGraph()
        self.repdict = {}

    def basins(self, order=True):
        """
        """
        if order:
            basins = sorted(
                (node for node in self.graph.nodes()
                 if type(node) is SuperBasin), key=lambda n: n.energy)
        else:
            basins = [node for node in self.graph.nodes()
                      if type(node) is SuperBasin]
        return basins

    def SuperBasin(self, replicas, parent=None, **kwargs):
        """
        """
        basin = SuperBasin(replicas)
        self.repdict.update((rep, basin) for rep in replicas)
        self.graph.add_node(basin, energy=basin.energy, **kwargs)

        if parent is None:
            parent = self.findParentBasin(basin)
        if parent is not None:
            self.connectBasins(parent, basin)

        minset = set(self.genAllConnectedMinima(basin))
        for m in minset:
            minbasin = self.repdict[m]
            self.connectTreeBasins(basin, minbasin)

        return basin

    def newMinimum(self, m):
        """
        """
        if m not in self.repdict:
            newbasin = self.SuperBasin([m])
        else:
            newbasin = self.repdict[m]

        # Updating the tree
        predecessors = set()
        for r in self.repGraph.genReplica2Root(m):
            if self.repdict.has_key(r):
                predecessors.add(self.repdict[r])
        self.mergeBranches(predecessors)

        assert nx.is_branching(self.graph)

        self.checkBasin(newbasin)

        return newbasin
#        return None

    def connectBasins(self, parent, basin, calcruns=False):
        """
        """
        assert parent != basin

        if calcruns:
            runs = list(self.genConnectingRuns(parent, basin))
            parentrep = min(parent.replicas, key=lambda r: r.energy)
            childrep = min(basin.replicas, key=lambda r: r.energy)
            if runs:
                run = combineAllRuns(runs, parentrep, childrep)
                self.graph.add_edge(parent, basin, parent=parent,
                                    run=run, nruns=len(runs))
            else:
                paths = list(self.genConnectingPaths(parent, basin))
                self.graph.add_edge(parent, basin, parent=parent,
                                    paths=paths, nruns=0)
                run = None
            return run
        else:
            self.graph.add_edge(parent, basin, parent=parent)

    def connectTreeBasins(self, parent, child):

        parentbranch = (
            self.repdict[r2] for r1 in parent for r2 in
            self.repGraph.genReplica2Root(r1) if self.repdict.has_key(r2))
        childbranch = (
            self.repdict[r2] for r1 in child for r2 in
            self.repGraph.genReplica2Root(r1) if self.repdict.has_key(r2))
        newbranch = sorted(set(chain(parentbranch, childbranch)))
        for b1, b2 in izip(newbranch[1:], newbranch[:-1]):
            for pre in self.graph.predecessors(b2):
                if pre != b1:
                    self.graph.remove_edge(pre, b2)
            if not self.graph.has_edge(b1, b2):
                self.graph.add_edge(b1, b2, parent=b1)

    def removeBasin(self, basin):
        parent = self.getParent(basin)
        if parent is not None:
            for child in self.graph.successors_iter(basin):
                self.connectBasins(parent, child)
        self.graph.remove_node(basin)

    def updateBasinReplicas(self, basin):
        minrep = min(basin.replicas)
        toadd = [rep for rep in self.genSucceedingReplicas(basin)
                 if rep > minrep]
        if toadd:
            return self.addtoBasin(basin, toadd)
        else:
            return basin

    def mergeBranches(self, tomerge):
        """Ensures that all the basins and their predecessors in tomerge
        are connected in decreasing energy order
        """
        basins = sorted(set(
            b2 for b1 in tomerge
            for b2 in chain([b1], self.genPreceedingBasins(b1))))

        for i, b1 in enumerate(basins[:-1]):
            for b2 in self.graph.predecessors(b1):
                if b2 != basins[i+1]:
                    self.graph.remove_edge(b2, b1)

            if not self.graph.has_edge(basins[i+1], b1):
                self.connectBasins(basins[i+1], b1)

        assert all(self.checkBasin(b) for b in basins)
        #print 'mergeBranches check', all(self.checkBasin(b) for b in basins)

    def mergeBasins(self, basins):
        """
        """
        if len(basins) > 1:
            newbasin = reduce(lambda x,y: x+y, basins)
            self.graph.add_node(newbasin, energy=newbasin.energy)
            newbasin = self.updateBasinReplicas(newbasin)
            self.repdict.update((rep, newbasin)
                                for rep in newbasin.replicas)
            mapping = dict((basin, newbasin) for basin in basins)
            nx.relabel_nodes(self.graph, mapping, copy=False)
            return newbasin
        else:
            return basins[0]

    def addtoBasin(self, basin, replicas):
        """
        """
        newbasin = basin + SuperBasin(replicas)
        if basin != newbasin:
            mapping = {basin: newbasin}
            self.repdict.update(
                (rep, newbasin) for rep in newbasin.replicas)
            nx.relabel_nodes(self.graph, mapping, copy=False)
        return newbasin

    def getParent(self, basin):
        try:
            parent, = self.graph.predecessors_iter(basin)
        except ValueError:
            parent = None
        return parent

    def getSiblings(self, basin):
        """
        """
        parent = self.getParent(basin)
        siblings = [b for b in self.graph.successors_iter(parent)
                    if b != basin]
        return siblings

    def _genSuccessorReplicas(self, basin):
        """Generates the set of successor replicas to basin
        """
        replicas = SortedCollection(basin)
        minrep = min(basin)
        while replicas:
            r1 = replicas.pop() # get the highest energy replica
            for r2 in self.repGraph.tree.successors_iter(r1):
                if r2 < minrep:
                    yield r2
                else:
                    replicas.add(r2)

    def genSuccessorReplicas(self, basin):
        """Generates the set of successor replicas to basin
        """
        replicas = SortedCollection(self.genAllConnectedMinima(basin),
                                    key=lambda b: -b.energy)
        minrep = min(basin)
        while replicas:
            r1 = replicas.pop()
            successor = False
            for r2 in self.repGraph.graph.predecessors_iter(r1):
                if r2 >= minrep:
                    successor = True
                else:
                    replicas.add(r2)
            if successor:
                yield r1

    def genPredecessorReplicas(self, basin):
        """
        """
        for r1 in basin:
            for r2 in self.repGraph.tree.predecessors_iter(r1):
                if r2 not in basin:
                    yield r2

    def genPreceedingReplicas(self, basin, ordered=False):
        """
        """
        if ordered:
            # Create ordered list of  preceeding replicas so that
            # the last element is the lowest energy replica
            replicas = SortedCollection(
                self.genPredecessorReplicas(basin), key=lambda b: -b.energy)
            while replicas:
                replica = replicas.pop()
                for r in self.repGraph.tree.predecessors_iter(replica):
                    replicas.add(r)
                yield replica
        else:
            for r1 in self.genPredecessorReplicas(basin):
                for r2 in self.repGraph.genPreceedingReplicas(r1):
                    yield r2

    def genSucceedingReplicas(self, basin, runs=False, paths=False):
        for r1 in self.genSuccessorReplicas(basin):
            yield r1
            for r2 in self.repGraph.genSucceedingReplicas(r1, runs=runs,
                                                          paths=paths):
                yield r2

    def genPreceedingBasins(self, basin):
        assert basin not in self.graph.predecessors_iter(basin)
        for child in self.graph.predecessors_iter(basin):
            for child2 in self.genPreceedingBasins(child):
                yield child2
            yield child

    def genSucceedingBasins(self, basin):
        for child in self.graph.successors_iter(basin):
            for child2 in self.genSucceedingBasins(child):
                yield child2
            yield child

    def findParentBasin(self, basin):
        for r1 in self.genPreceedingReplicas(basin, ordered=True):
            if self.repdict.has_key(r1):
                return self.repdict[r1]
        else:
            return None

    def checkBasin(self, basin):
        """ Checks whether the basin has only
        """

        predecessors = self.graph.predecessors(basin)
        successors = self.graph.successors(basin)

        if len(predecessors) > 1:
            print "Too many parents"
            return False
        elif basin in predecessors:
            print "Self connected"
            return False
        if successors:
            minima = self.getMinimaSet(basin)
            childminima = reduce(
                lambda x, y: x.union(y),
                (self.getMinimaSet(b) for b in successors) )

            if minima.issuperset(childminima):
                return True
            else:
                minima = self.getMinimaSet(basin, recalc=True)
                childminima = set(m for b in successors
                                  for m in self.getMinimaSet(b, recalc=True))
                if not minima.issuperset(childminima):
                    print 'failure!!!!', basin.energy
                    print "superset", len(minima), len(childminima), len(minima.difference(childminima))
                return minima.issuperset(childminima)
        else:
            return True

    def getMinimaSet(self, basin, recalc=False):
        node = self.graph.node[basin]
        if recalc or not node.has_key('minima'):
            node['minima'] = set(self.genAllConnectedMinima(basin))
        return node['minima']

    def genAllConnectedMinima(self, basin):
        """
        """
        for r1 in basin:
            if isinstance(r1, Minimum):
                yield r1
            else:
                for r2 in self.repGraph.genSucceedingReplicas(r1):
                    if r2 in basin:
                        break
                    elif isinstance(r2, Minimum):
                        yield r2

    def updateGraph(self):
        """
        """
        self.updateMinima()
        self.updateAllMinimaSets()
        self.updateSubgraph(self.basins())

    def basinOutDegree(self, basin, runs=True, paths=False):
        return sum(iter_minlength(
            self.repGraph.genConnectedMinima(r, runs=runs, paths=paths), 1)
            for r in self.genSuccessorReplicas(basin))

    def basinDegree(self, basin):
        """ Returns the sum of the degree of the replicas in the basin
        """
        return sum(d for r, d in
                   self.repGraph.graph.out_degree_iter(basin.replicas))

    def genConnectedMinima(self, basin, runs=True, paths=False):
        """ Returns list of connected minima and weights
        if runs is True only returns minima connected by nested sampling runs
        if paths is True only returns minima connected by paths
        if runs and paths are False then returns all minima
        """
        successors = list(self.genSuccessorReplicas(basin,
                                                    runs=runs, paths=paths))
        if len(successors):
            f = 1./len(successors)
            if paths:
                return (
                    mf for r in successors
                    for mf in self.repGraph.genConnectedMinima(
                        r, f=f, paths=True))
            elif runs:
                return (
                    mf for r in successors
                    for mf in self.repGraph.genConnectedMinima(
                        r, f=f, runs=True))
            else:
                return (
                    mf for r in successors
                    for mf in self.repGraph.genConnectedMinima(
                        r, f=f, runs=False, paths=False))
        else:
            minima = [r for r in basin if isinstance(r, Minimum)]
            f = 1./len(successors)
            return ((m, f) for m in minima)

    def getMinimaCDF(self, basin, runs=True, paths=False):
        """Returns the CDF of the minima connected to basin

        Parameters
        ----------
        basin : SuperBasin

        Returns
        -------
        cdf : CDF
            CDF of minima
        """
        childminima = [
            [mf for mf in self.repGraph.genConnectedMinima(
                r, runs=runs, paths=paths)]
            for r in self.genSuccessorReplicas(basin)]

        for i, minfs in enumerate(childminima):
            tot = sum(f for m, f in minfs)
            childminima[i] = [(m, f/tot) for m, f in minfs]

        n = sum(1 for minima in childminima if minima)
        if n:
            ms, ws = zip(*[(m, f/n)
                         for minima in childminima for m, f in minima])
            return CDF([m.energy for m in ms], ws, n=n)
        else:
            return None

    def isMinimum(self, basin):
        for r in basin:
            if isinstance(r, Minimum):
                return True
        else:
            return False

    def genSamples(self, basin):
        return (x for r in basin for x in self.repGraph.genSamples(r))

    def genEnergySamples(self, basin):
        return (E for r in basin for E in self.repGraph.genEnergySamples(r))

    def getEnergyPercentile(self, basin, q):
        Es = list(self.genEnergySamples(basin))
        if Es:
            return np.percentile(Es, q)
        else:
            return np.nan * q

    def genConnectingReplicas(self, basin1, basin2):
        """
        Generates all the paths in repGraph.graph that 'connect'
        basin1 to basin2
        """
        if basin1 < basin2:
            basin1, basin2 = basin2, basin1

        target = min(basin1)
        stop = max(basin1)

        G = self.repGraph.graph
        cutoff = len(G)
        minima = self.genAllConnectedMinima(basin2)
        # Start from all the minima that basin2 is connected to.
        visitedstacks = [([m], [G.in_edges_iter(m)]) for m in minima]
        basinreps = set(basin2.replicas).copy()
        while visitedstacks:
            visited, stack = visitedstacks.pop()
            if stack:
                parent, _ = next(stack[-1], (None, None))
                basinreps.discard(parent)
                if parent is None:
                    stack.pop()
                    visited.pop()
                elif len(visited) < cutoff:
                    if parent >= target:
                        replicas = (visited + [parent])[::-1]
                        # Discard the replicas lower than basin2
                        for i, replica in enumerate(replicas):
                            if replica.energy < basin2.energy:
                                yield replicas[:i+1]
                                break
                        else:
                            yield replicas
                    elif parent not in visited and parent < stop:
                        visited.append(parent)
                        stack.append(G.in_edges_iter(parent))
                visitedstacks.insert(0, (visited, stack))
            if not visitedstacks:
                visitedstacks = [([r], [G.in_edges_iter(r)]) for r in basinreps]
                basinreps = set()

    def getConnectingPath(self, basin1, basin2):
        # Ensuring basins are in the right order
        if basin1 < basin2:
            basin1, basin2 = basin2, basin1

        replicas = next(self.genConnectingReplicas(basin1, basin2), None)
        startrep = min(basin1)
        endrep = min(basin2)

        if replicas is not None:
            path = self.repGraph.replicastoPath(replicas, startrep, endrep)
        else:
            path = None
        return path

    def genConnectingRuns(self, basin1, basin2):
        """ generates all runs that connect basin1 and basin2
        """
        # Ensuring basins are in the right order
        if basin1 < basin2:
            basin1, basin2 = basin2, basin1

        allreplicas = self.genConnectingReplicas(basin1, basin2)
        edgereplicas = (
            (self.repGraph.graph.edge[r1][r2], r1, r2) for replicas in
            allreplicas for r1, r2 in izip(replicas[:-1],replicas[1:]))
        runsreplicas = set((edge['run'], r1, r2) for edge, r1, r2 in
            edgereplicas if edge.has_key('run'))
        return (splitRun(run, r1, r2) for run, r1, r2 in runsreplicas)

    def getConnectingRun(self, basin1, basin2):
        """
        """
        # Ensuring basins are in the right order
        if basin1 < basin2:
            basin1, basin2 = basin2, basin1

        allreplicas = self.genConnectingReplicas(basin1, basin2)
        edgereplicas = (
            (self.repGraph.graph.edge[r1][r2], r1, r2) for replicas in
            allreplicas for r1, r2 in izip(replicas[:-1],replicas[1:]))
        # Need to make sure that each edge is unique
        runsreplicas = set((edge['run'], r1, r2) for edge, r1, r2 in
            edgereplicas if edge.has_key('run'))
        run = combineAllRuns(
            [splitRun(run, r1, r2) for run, r1, r2 in runsreplicas])
        connectingRun = splitRun(run, min(basin1), min(basin2))
        return connectingRun


    def splitBasins(self, basins, Ecut, sort=False):
        """
        """
        if sort:
            basins = sorted(basins)

        for i2, basin2 in enumerate(reversed(basins)):
            if basin2.energy < Ecut:
                break
            else:
                basin1 = basin2

        replicas = (
            path for basin2 in basins[-i2::-1] if basin2 != basin1
            for path in self.genConnectingReplicas(basin1, basin2)).next()

        rephi = replicas[0]
        for replo in replicas[1:]:
            if replo.energy < Ecut:
                break
            else:
                rephi = replo

        newrep = self.repGraph.splitReplicaPair(rephi, replo, Ecut)

        if self.repdict.has_key(newrep):
            return self.repdict[newrep], newrep
        else:
            newbasin = self.SuperBasin([newrep])
            return newbasin, newrep

    def compareBasinPair(self, basin1, basin2, runs=True, paths=False):
        """
        Compares the distribution of minima in basin1 and basin2
        returns the likelihood that that the distributions are the same

        parameters
        ----------
        basin1 : SuperBasin
        basin2 : SuperBasin
        runs : Bool, optional
            if true only matches minima connected by nested optimisation runs
        paths : Bool, optional
            if true only matches minima connected by paths

        returns
        -------
        sig : float
            the significance
        cdf1 : CDF
            the CDF of basin1
        cdf2 : CDF
            the CDF of basin2
        """

        allmin1 = set(self.genAllConnectedMinima(basin1))
        allmin2 = set(self.genAllConnectedMinima(basin2))

        maxmin1 = max(allmin1, key=lambda m: m.energy)
        maxmin2 = max(allmin2, key=lambda m: m.energy)

        if maxmin1.energy > basin1.energy or maxmin2.energy > basin2.energy:
            return 0., None, None

        cdf1 = self.getMinimaCDF(basin1, runs=runs, paths=paths)
        cdf2 = self.getMinimaCDF(basin2, runs=runs, paths=paths)

        if cdf1 is None or cdf2 is None:
            return None, cdf1, cdf2
        else:
            sig = AndersonDarling.compareDistributions((cdf1, cdf2))[0]
            return sig, cdf1, cdf2

    @property
    def disconnectivityGraph(self):
        """
        """
        minima = sorted(self.repGraph.minima())
        if self.Emax is not None:
            minima = filter(lambda m: m.energy < self.Emax, minima)

        if set(self._connectgraph.nodes()) != set(minima):
            self._connectgraph = self.calcConnectivityGraph(self.Emax)
            self._disconnectgraph = DisconnectivityGraph(self._connectgraph,
                                                         **self.disconnect_kw)
            self._disconnectgraph.calculate()

        return self._disconnectgraph

    def recalcDisconnectivityGraph(self):
        self._connectgraph = self.calcConnectivityGraph(self.Emax)
        self._disconnectgraph = DisconnectivityGraph(self._connectgraph,
                                                         **self.disconnect_kw)
        self._disconnectgraph.calculate()

    def calcConnectivityGraph(self, Emax=None):
        """ Calculates disconnectivity graph of the BasinGraph where
        the lowest energy basin that connects two minima is used as the
        Transition state
        """
        minima = sorted(self.repGraph.minima())
        if Emax is not None:
            minima = filter(lambda m: m.energy < Emax, minima)
        pairs = set((m1, m2) for i, m1 in enumerate(minima)
                    for m2 in minima[:i])

        g = nx.Graph()
        g.add_nodes_from(minima)

        basins = self.basins()
        for basin in basins:
            minlist = sorted(self.getMinimaSet(basin))
            if pairs:
                for i, m1 in enumerate(minlist):
                    for m2 in minlist[:i]:
                        if (m1, m2) in pairs:
                            ts = TransitionState(basin.energy, None, m1, m2)
                            g.add_edge(m1, m2, ts=ts)
                            pairs.remove((m1,m2))

        return g

    def plot(self, recalc=False, axes=None, **kwargs):
        """
        """
        if recalc:
            self.recalcDisconnectivityGraph()

        if axes is None:
            axes=plt.gca()
        kwargs.update(axes=axes)

        dg = self.disconnectivityGraph
        dg.plot(**kwargs)
        return dg

    def draw(self, energies=True, maxE=0., **kwargs):
        """
        """
        pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')
        if energies:
            pos = dict((r, (p[0], np.clip(r.energy, None, maxE)))
                        for r,p in pos.iteritems())
        else:
            basin = max(self.basins())
            pos[basin] = (pos[basin][0], 0)
            basins = set([basin])
            while basins:
                basin = basins.pop()
                for b in self.graph.successors_iter(basin):
                    pos[b] = (pos[b][0], pos[basin][1] - 1)
                    basins.add(b)
        nx.draw(self.graph, pos, **kwargs)












    def genConnectingReplicas2(self, basin1, basin2):
        G = self.repGraph.graph
        cutoff = len(G)
        maxE = max(basin1).energy
        visitedstacks = [([r], [G.in_edges_iter(basin2)]) for r in basin2]
        while visitedstacks:
            visited, stack = visitedstacks.pop()
            if stack:
                parents = stack[-1]
                parent, _ = next(parents, (None, None))
                if parent is None:
                    stack.pop()
                    visited.pop()
                elif len(visited) < cutoff:
                    if parent in basin1:
                        yield (visited + [parent])[::-1]
                    elif parent not in visited and parent.energy < maxE:
                        visited.append(parent)
                        stack.append(G.in_edges_iter(parent))
                    visitedstacks.insert(0, (visited, stack))

    def updateAllMinimaSets(self):
        for basin in self.basins():
            self.getMinimaSet(basin, recalc=True)

    def updateMinima(self):
        newminima = set(self.repGraph.minima()).difference(self.repdict)
        for m in newminima:
            self.SuperBasin([m])

    def updateNode(self, basin):
        """
        """
        minimaset = self.getMinimaSet(basin, recalc=True)
        basins = (basin for basin in self.basins()
                  if self.getMinimaSet(basin).intersection(minimaset))
        self.updateSubgraph(basins)

    def updateSubgraph(self, basins):
        """ This method makes sures that the connectivity of the basins
        satisfies the two constraints:
            1. all edges between basins must be downhill
            2. any basins that connect to the same minimum must be in
               the same super basin

        parameters
        ----------
        basins : iterable of SuperBasin
            all the basins that are going to be checked
        """
        basins = sorted(basins, reverse=True)
        minimadict = dict((m, basins[0]) for m in self.getMinimaSet(basins[0]))

        for newbasin in basins[1:]:

            parents = set(minimadict[m] for m in self.getMinimaSet(newbasin))
            if len(parents) == 1:
                parent, = parents
            elif len(parents) > 1:
                self.mergeBranches(parents)
                parent = min(b for b in parents if b != newbasin)
            if parents:
                predecessors = self.graph.predecessors(newbasin)
                if parent not in predecessors:
                    for b in predecessors:
                        self.graph.remove_edge(b, newbasin)
                if not self.graph.has_edge(parent, newbasin):
                    self.connectBasins(parent, newbasin)

            minimadict.update((m, newbasin)
                              for m in self.getMinimaSet(newbasin))

            assert self.checkBasin(newbasin)



if False:

    def newMinimum(self, m):

        if m not in self.repdict:
            print 'new min', m.energy
            newbasin = self.SuperBasin([m])
            predecessors = [self.repdict[r]
                            for r in self.repGraph.genPreceedingReplicas(m)
                            if self.repdict.has_key(r)]
            parent = min(predecessors)
            self.connectBasins(parent, newbasin)
            for basin in predecessors:
                self.getMinimaSet(basin).update([m])
        else:
            print 'old min', m.energy
            newbasin = self.repdict[m]
            predecessors = [self.repdict[r]
                            for r in self.repGraph.genPreceedingReplicas(m)
                            if self.repdict.has_key(r)]
            self.mergeBranches(predecessors)

        self.checkBasin(newbasin)

        return newbasin

    def getBasinReplicas(self, basin):
        """
        Returns the lowest energy replicas that are independently
        connected to the same minima that the replicas in basin
        are connected to which are higher in energy than the
        lowest energy replica in the basin.
        """
        minima = set(m for r1 in basin for m, f in
                     self.repGraph.genConnectedMinima(r1))
        replicas = set(r for m in minima
                       for r in self.repGraph.graph.predecessors_iter(m))
        basinreplica = min(basin)
        basinreplicas = set([basinreplica])
        while replicas:
            replica = replicas.pop()
            if replica > basinreplica:
                basinreplicas.add(replica)
            else:
                basinreplicas.update(
                    self.repGraph.graph.predecessors_iter(replica))
        return basinreplicas

    def _updateGraph(self):
        """
        """
        minima = self.repGraph.minima()
        activenodes = SortedCollection(
            key=lambda bm: (-max(bm[1]).energy, -len(bm[1])))

        for m in minima:
            if self.repdict.has_key(m):
                basin = self.repdict[m]
            else:
                basin = SuperBasin([m])
                self.repdict.update((rep, basin) for rep in [m])
                self.graph.add_node(basin, energy=basin.energy)

            activenodes.insert((basin, set([m])))

        for basin in self.graph.nodes_iter():
            self.getMinimaSet(basin, recalc=True)

        while activenodes:
            basin, mins = activenodes.pop()
            parent = self.getParent(basin)


            nextbasins = sorted(set(
                self.repdict[r] for r in self.genPreceedingReplicas(basin)
                if self.repdict.has_key(r)))

            if nextbasins:
                nextbasin = nextbasins[0]

                if parent is not None and nextbasin != parent:
                    self.graph.remove_edge(parent, basin)

                assert nextbasin.energy > basin.energy

                if parent is None or nextbasin != parent:
                    self.connectBasins(nextbasin, basin)

                for i, b1 in enumerate(nextbasins[:-1]):
                    for b2 in self.graph.predecessors(b1):
                        if b2 != nextbasins[i+1]:
                            self.graph.remove_edge(b2, b1)
                            self.connectBasins(b2, nextbasins[i+1])

                    if not self.graph.has_edge(nextbasins[i+1], b1):
                        self.connectBasins(nextbasins[i+1], b1)

                activenodes.insert((nextbasin, self.getMinimaSet(nextbasin)))

    def _updateMinimum(self, m):
        """
        DOESN'T WORK CURRENTLY!
        """
        replicas = self.repGraph.graph.predecessors(m)
        if m in self.repdict:
            basin = self.repdict[m]
            if not basin.replicas != set(replicas):
                self.addtoBasin(basin, replicas)
        else:
            basin = self.SuperBasin(replicas)
            self.repdict[m] = basin

        self.mergeBranches(basin)

    def _updateMinima(self):
        """
        DOESN'T WORK CURRENTLY!
        """
        minima = self.repGraph.minima()
        for m in minima:
            self.updateMinimum(m)

    def _mergeBranches(self, basin):
        """
        DOESN'T WORK CURRENTLY!
        """
        # Generating all the basins
        parentreplicas = set(chain(*(self.repGraph.genPreceedingReplicas(r)
                                     for r in basin.replicas)))
        splitreplicas = sorted((self.findEnergySplit(r), r)
                                for r in parentreplicas)
        basinreplicas = ((k,[r for _, r in g]) for k, g in
                         groupby(splitreplicas, lambda sr: sr[0]))
        newbasins = [(k, SuperBasin(rs)) for k, rs in basinreplicas]

        # Finding the current basins
        currentbasins = []
        parent = self.getParent(basin)
        while parent is not None:
            currentbasins.append((self.findEnergySplit(parent), parent))
            parent = self.getParent(parent)

        # grouping the new and old basins
        allbasins = currentbasins + newbasins
        allbasins.sort()

        # grouping basins into seperate energy splits
        groupedbasins = [[b for _, b in g]
                         for _, g in groupby(allbasins, lambda b:b[0])]

        # Merging basins in the same energy bin
        mergedbasins = [basin] + [self.mergeBasins(bs) for bs in groupedbasins]

        # Adding/removing edges
        for mbasin, pbasin in izip(mergedbasins[:-1], mergedbasins[1:]):
            parents = self.graph.predecessors(mbasin)
            if len(parents) > 0:
                for p in parents:
                    if p != pbasin:
                        self.graph.remove_edge(p, mbasin)
                    elif p is not pbasin:
                        nx.relabel_nodes(self.graph, {parents[0]: pbasin},
                                         copy=False)
            elif not self.graph.has_edge(pbasin, mbasin):
                self.connectBasins(pbasin, mbasin)

    def isDangling(self, basin, nsamples=2):
        """ Tests whether there is only less than nsamples replicas in a
        basin that is connected to nsamples or more replicas
        """
        if len(basin.replicas) < nsamples:
            r, = basin.replicas
            successors = chain(*(self.repGraph.graph.successors_iter(r)
                                 for r in basin.replicas))
            return not iter_minlength(successors, nsamples)
        else:
            return False

    def isSingleton(self, basin):
        """ Tests whether there is only one replica in basin which is
        connected to less than 2 other replicas
        """
        if self.isDangling(basin):
            r, = basin.replicas
            return not iter_minlength(self.repGraph.graph.successors_iter(r),2)
        else:
            return False


    def genConnectingRuns(self, basin, child):
        """
        """
        if basin!=child:
            for parentrep in basin.replicas:
                edges = self.repGraph.graph.edge[parentrep]
                for childrep, attr in edges.iteritems():
                    if childrep in child.replicas:
                        if attr.has_key('run'):
                            yield attr['run']

    def genConnectingPaths(self, basin, child):
        """
        """
        if basin!=child:
            for parentrep in basin.replicas:
                edges = self.repGraph.graph.edge[parentrep]
                for childrep, attr in edges.iteritems():
                    if childrep in child.replicas:
                        if attr.has_key('path'):
                            yield attr['path']

    def findEnergySplit(self, basin):
        Esplits = np.r_[self.repGraph.Esplits, np.inf]
        return Esplits.searchsorted(basin.energy, side='right')

    def _mergeBasins(self, basins):
        """
        """
        if len(basins) > 1:
            newbasin = reduce(lambda x,y: x+y, basins)
            self.graph.add_node(newbasin, energy=newbasin.energy)
            self.repdict.update((rep, newbasin)
                                for rep in newbasin.replicas)

            mapping = dict((basin, newbasin) for basin in basins)
            nx.relabel_nodes(self.graph, mapping, copy=False)
            return newbasin
        else:
            return basins[0]

    def getConnectingRuns2(self, parent, child):
        """
        """
        runs = []
        for rep1 in parent.replicas:
            for rep2 in child.replicas:
                runs.extend(self.repGraph.pathtoRun(p) for p in
                            nx.all_simple_paths(self.repGraph.graph,
                                                rep1, rep2))
        return runs

    def number_of_successors(self, basin):
        """
        """
        replicas = basin.replicas
        successors = set()
        for rep in replicas:
            successors.update(self.repGraph.graph.successors(rep))
        return len(successors.difference(replicas))

    def joinBasins(self, basins):
        """
        """
        newbasin = reduce(lambda x,y: x+y, basins)

        self.SuperBasin(newbasin.replicas)

        predecessors = set(sum((self.graph.predecessors(b)
                               for b in basins), [])).difference(basins)
        successors = set(sum((self.graph.successors(b)
                              for b in basins), [])).difference(basins)

        for parent in predecessors:
            self.add_edge(parent, newbasin)
        for child in successors:
            self.add_edge(newbasin, child)

        self.graph.remove_nodes_from(basins)

        return newbasin

    def get_lowest_basin(self):
        """
        """
        return min( (node for node in self.graph.nodes()
                     if type(node) is SuperBasin), key=lambda n: n.energy)

    def basins(self, order=True):
        """
        """
        if order:
            basins = sorted(
                (node for node in self.graph.nodes()
                 if type(node) is SuperBasin), key=lambda n: n.energy)
        else:
            basins = [node for node in self.graph.nodes()
                      if type(node) is SuperBasin]
        return basins

    def genNextReplicas(self, basin):
        return chain(*(self.repGraph.graph.successors_iter(r) for r in basin))

    def minima(self, order=True):
        """
        """
        if order:
            minima = sorted(
                (node for node in self.graph.nodes()
                 if type(node) is Minimum), key=lambda n: n.energy)
        else:
            minima = [node for node in self.graph.nodes()
                      if type(node) is Minimum]
        return minima

    def genBasinReplicas(self, basin, notchild=False, notparent=False):
        """
        """
        if notchild:
            for rep in basin.replicas:
                ischild = any(nx.has_path(self.repGraph.graph, _rep, rep)
                              for _rep in basin.replicas.difference([rep]))
                if not ischild:
                    yield rep
        elif notparent:
            for rep in basin.replicas:
                isparent = any(nx.has_path(self.repGraph.graph, rep, _rep)
                               for _rep in basin.replicas.difference([rep]))
                if not isparent:
                    yield rep
        else:
            for rep in basin.replicas:
                yield rep

    def genConnectedReplicas(self, basin, Esplit=-np.inf):
        """
        """
        for rep1 in self.genBasinReplicas(basin, notparent=False):
            if rep1.energy >= Esplit:
                for rep2 in self.repGraph.genConnectedReplicas(rep1):
                    if rep2.energy >= Esplit:
                        yield rep2




    def genConnectedRuns(self, basin, Efilter=None):
        """
        """
        replicas = sorted(basin.replicas, key=lambda r: -r.energy)
        startreps = replicas[:1]
        for rep in replicas[1:]:
            ischild = any(nx.has_path(self.repGraph.graph, srep, rep)
                            for srep in startreps)
            if not ischild:
                startreps.append(rep)
        # Generator to return runs
        for rep in startreps:
            for run in self.repGraph.genConnectedRuns(rep, Efilter):
                yield run

    def getBasinRuns(self, basin):
        """
        """
        replicas = sorted(basin.replicas, key=lambda r: -r.energy)
        startreps = replicas[:1]
        for rep in replicas[1:]:
            ischild = any(nx.has_path(self.repGraph.graph, srep, rep)
                            for srep in startreps)
            if not ischild:
                startreps.append(rep)
        runs = sum((self.repGraph.getConnectedRuns(rep) for rep in startreps),[])
        return runs


    def calcHarmonicConstraints(self, parent, basin, minimum, E, c, ndof,
                                numeric=False):
        """
        """
        edge = self.graph.edge[parent][basin]
        node = self.graph.node[parent]

        nlive = edge['run'].nlive
        Emax = edge['run'].Emax

        N = Emax.size - Emax[::-1].searchsorted(E,side='right') + 1

        Ec = Emax[N-1]
        logPhiCon = log(NumericIntegrator.harmonicEtoVol(Ec - minimum.energy,
                                                         c, ndof))
        Phi = node['Phi']
        logX = node['logX']
        logX2 = node['logX2']
        #X, X2 = exp(logX), exp(logX2)

        constraints = dict(E=E, logPhiCon=logPhiCon, Phi=Phi, Emax=Emax[:N],
                           logX=logX, logX2=logX2, nlive=nlive[:N])

        if numeric:
            aint = NumericIntegrator.HarmonicIntegrator(
                0, exp(logPhiCon), 1., ndof)
            constraints['NumericalIntegral'] = aint

        edge['constraints'] = constraints

        return edge

    def calcBranchVolume(self, parent, basin, Es=None, res=512):
        """
        """

        edge = self.graph.edge[parent][basin]
        edgesEs = [np.r_[parent.energy, edge['Es']]]
        edgesPhi = [edge['Phi'] * np.r_[1., edge['Xs']]]

        edges = {}
        edges.update(self.graph.edge[basin])
        while edges:
            for child in edges.keys():
                cedge = edges.pop(child)
                edgesEs.append(np.r_[cedge['parent'].energy, cedge['Es']])
                edgesPhi.append(cedge['Phi'] * np.r_[1., cedge['Xs']])
                edges.update(self.graph.edge[child])

        if Es is None:
            Emax = parent.energy
            Emin = min(_Es[-1] for _Es in edgesEs)
            Es = np.linspace(Emax, Emin, res)

        Phi = np.zeros_like(Es)
        for _Es, _Phi in izip(edgesEs, edgesPhi):
            Phi += np.interp(Es, _Es[::-1], _Phi[::-1], 0., 0.)

        return Es, Phi

    def plotBranchVolumes(self, parent, basin, Es=None, res=512,
                          ax=None, c=None, widthfunc=np.log10):
        """
        """

        Es, Phi = self.calcBranchVolume(parent, basin, Es=Es, res=res)

        width = widthfunc(Phi)
        width -= width.min()
        left = -width/2
        right = width/2

        if ax is None:
            ax = plt.gca()

        color = self.graph.node[basin].get('color', 'k') if c is None else c
        ax.plot(np.r_[left, right[-2::-1]], np.r_[Es, Es[-2::-1]], c=color)

        basinedge = {basin: left}
        basins = [basin]

        while basins:
            current = basins.pop()
            currentleft = basinedge[current]
            edgeVols = [(child,
                        self.calcBranchVolume(current, child, Es=Es, res=res))
                        for child in self.graph.successors(current)]
            edgeVols.sort(key=lambda x: x[1][0][x[1][1].nonzero()[0].max()])

            for child, (cEs, cPhi) in edgeVols[:-1]:
                cWidth = cPhi/Phi * width
                basinedge[child] = currentleft
                basins.append(child)

                currentleft = currentleft + cWidth
                nonzero = cWidth.nonzero()
                color = self.graph.node[child].get('color', 'k') if c is None else c
                ax.plot(currentleft[nonzero], Es[nonzero], c=color)

            if edgeVols:
                child, (cEs, cPhi) = edgeVols[-1]
                basinedge[child] = currentleft
                basins.append(child)

    def calcBasinVolume(self, basin, Es=None):
        """
        """
        node = self.graph.node[basin]
        edges = self.graph.edge[basin]

        edgesEs = []
        edgesPhi = []
        for child, edge in edges.iteritems():
            childEs, childPhi = self.calcBasinVolume(child, Es)

            if Es is not None:
                Xs = childPhi
                Xs += np.interp(Es[::-1],
                                edge['Es'][::-1], edge['Xs'][::-1])[::-1]
                E = Es
            else:
                E = np.r_[edge['Es'], childEs]
                Xs = np.r_[edge['Xs'], edge['Xs'][-1] * childPhi]

            phi = node['branchPi'][child] * Xs
            edgesEs.append(E)
            edgesPhi.append(phi)

        if edges:
            if Es is None:
                Es = min(edgesEs, key=lambda E: E[-1])
                Phi = np.zeros_like(Es)
                for _Es, _phi in izip(edgesEs, edgesPhi):
                    Phi += np.interp(Es[::-1], _Es[::-1], _phi[::-1])[::-1]
            else:
                Phi = sum(edgesPhi)
        else:
            Es, Phi = np.array([]), np.array([])

        return Es, Phi

    def plotBasinVolume(self, basin, basinshape=None, ax=None, c='k'):
        """
        """
        if ax is None:
            ax = plt.gca()

        if basinshape is None:
            Es, Phi = self.calcBasinVolume(basin)
            width = np.log10(Phi)
            width -= width.min()

            left = -width/2
            right = left + width
            ax.plot(np.r_[left,right[::-1]], np.r_[Es, Es[::-1]], c=c)
        else:
            Es, left, width = basinshape

        node = self.graph.node[basin]
        branchPi = node['branchPi']

        childrenVols = [(child, self.calcBasinVolume(child))
                        for child in self.graph.successors(basin)]
        try:
            childrenVols.sort(key=lambda x: x[1][0][-1])
        except IndexError:
            childrenVols = []

        totVol = np.zeros_like(width)
        childrenRelVols = []
        for child, (_Es, _Phi) in childrenVols:
            _Es = np.r_[_Es[::-1], basin.energy]
            _Phi = np.r_[_Phi[::-1], branchPi[child]]
            _nPhi = branchPi[child] * np.interp(Es, _Es, _Phi, 0.,0.)
            totVol += _nPhi
            childrenRelVols.append((child, _nPhi))

        nonzero = totVol.nonzero()[0][::-1]

        for child, _Phi in childrenRelVols:
            _Phi[nonzero] /= totVol[nonzero]

        currleft = left.copy()
        for child, relVol in childrenRelVols[:-1]:
            childwidth = relVol * width
            ax.plot(currleft[nonzero] + childwidth[nonzero], Es[nonzero], c=c)

            childshape = (Es, currleft, childwidth)
            self.plotBasinVolume(child, basinshape=childshape, ax=ax, c=c)
            currleft[nonzero] += childwidth[nonzero]

        if childrenRelVols:
            child, relVol = childrenRelVols[-1]
            childwidth = relVol * width
            childshape = (Es, currleft, childwidth)
            self.plotBasinVolume(child, basinshape=childshape, ax=ax, c=c)

    def calcBasins(self):
        """
        """
        basin0 = self.basins(order=True)[-1]
        tocalculate = [basin0]

        while tocalculate:
            basin = tocalculate.pop()
            self.calcBasinVolumeRatio(basin)
            self.calcBranchProbabilities(basin)
            successors = self.graph.successors(basin)
            tocalculate.extend(successors)

    def calcEdgeValues(self, edge):
        """
        """
        if edge.has_key('run'):
            run = edge['run']
            ns = run.nlive.astype(float)
            Emax = run.Emax.copy()
        else:
            ns = np.array([])
            Emax = np.array([])

        Xs = (ns/(ns+1.)).cumprod()
        n1n2 = ((ns+1.)/(ns+2.)).cumprod()
        X2s = Xs * n1n2
        n2n3 = ((ns+2.)/(ns+3.)).cumprod()
        n3n4 = ((ns+3.)/(ns+4.)).cumprod()

        X = Xs[-1] if Xs.size else 0.
        X2 = X2s[-1] if X2s.size else 0.

        dF = Xs / ns

        d2F2 = (n1n2/(ns+1.))
        dF2  = 2 * Xs / ns

        XdF = X * n1n2 / ns
        X2dF = X2 * n2n3 / ns

        Xd2F2 = n2n3/(ns+2.)
        XdF2 = 2 * n1n2/(ns+1.) * X

        X2d2F2 = n3n4/(ns+3.)
        X2dF2 = 2 * n2n3/(ns+2.) * X2


        dvarf = ( (2*(ns+1)/(ns+2) *
                  ((ns+2)**2/(ns+1)/(ns+3)).cumprod()).mean() -
                 (2*ns/(ns+1) * ((ns+1)**2/ns/(ns+2)).cumprod()).mean())
        dvarX = (np.exp((2*np.log(ns+2)-np.log(ns+1)-np.log(ns+3)).sum()) -
                 np.exp((2*np.log(ns+1)-np.log(ns+0)-np.log(ns+2)).sum()) )

        if dvarf < -1. or dvarf >= 0:
            dvarf = -1.
        if dvarX < -1. or dvarX >= 0:
            dvarX = -1.


        edge['Es'] = Emax
        edge['Xs'] = Xs
        edge['X2s'] = X2s
        edge['dF'] = dF
        edge['XdF'] = XdF
        edge['X2dF'] = X2dF
        edge['d2F2'] = d2F2
        edge['dF2'] = dF2
        edge['Xd2F2'] = Xd2F2
        edge['XdF2'] = XdF2
        edge['X2d2F2'] = X2d2F2
        edge['X2dF2'] = X2dF2

        edge['dvarf'] = dvarf
        edge['dvarX'] = dvarX

        return edge

    def calcConstrainedEdgeValues(self, edge):
        """
        """

        if edge.has_key('constraints'):

            constraints = edge['constraints']
            ns = constraints['nlive'].astype(float)
            Phi = constraints['Phi']
            logPhiCon = constraints['logPhiCon']
            logPhiCon2 = constraints.get('logPhiCon2', 2* logPhiCon)
            logX  = constraints['logX']
            logX2  = constraints['logX2']

            logPhi0 = log(Phi) + logX
            logPhi02 = 2*log(Phi) + logX2

            lognn1 = (np.log(ns) - np.log(ns-1.)).sum()
            lognn2 = (np.log(ns) - np.log(ns-2.)).sum()
            logphi1 = ((logPhi0 - logPhiCon - lognn1)/ns.size)
            logphi2 = ((logPhi02 - logPhiCon2 - lognn2)/ns.size)

            logn1np = (np.log(ns-1.) - np.log(ns) - logphi1).cumsum()
            logp1n2p2n1 = (np.log(ns-2.) - np.log(ns-1) +
                           logphi1 - logphi2).cumsum()

            dPhi = np.exp(logPhi0 + logn1np)/(ns-1.)
            d2Phi2 = np.exp( logn1np - np.log(ns-1.))
            dPhi2 = 2* np.exp( logPhi02 + logp1n2p2n1 ) / (ns-2.)

            constraints['dPhi'] = dPhi
            constraints['dPhi2'] = dPhi2
            constraints['d2Phi2'] = d2Phi2

            return edge
        else:
            return self.calcEdgeValues(edge)

    def calcBranchProbabilities(self, basin):
        """
        """
        node = self.graph.node[basin]
        successors = self.graph.successors(basin)
        if successors:
            nruns = dict((b, attr['nruns'])  for b, attr in
                         self.graph.edge[basin].iteritems())
            totruns = float(sum(nruns.itervalues()))
            branchP = dict((b, nrun/totruns) for b, nrun in
                            nruns.iteritems())
            branchPiPj = dict((bi,
                               dict((bj,0.) for bj in nruns))
                               for bi in nruns)
            MM1 = totruns*(totruns+1.)
            for bi, nruni in nruns.iteritems():
                for bj, nrunj in nruns.iteritems():
                    if bi is bj:
                        branchPiPj[bi][bi] = (nruni*(nruni+1.))/MM1
                    elif branchPiPj[bi][bj] == 0.:
                        branchPiPj[bi][bj] = nruni*nrunj/MM1
                        branchPiPj[bj][bi] = branchPiPj[bi][bj]

            node['branchPi'] = branchP
            node['branchPiPj'] = branchPiPj

            for child in successors:
                edge = self.graph.edge[basin][child]
                edge['Phi'] = node['Phi'] * node['X'] * branchP[child]
                self.calcEdgeValues(edge)

        else:
            node['branchPi'] = {}
            node['branchPiPj'] = {}

        return node

    def calcBasinVolumeRatio(self, basin, Phi=1.):
        """
        """
        node = self.graph.node[basin]
        predecessors = self.graph.predecessors(basin)
        if predecessors:
            assert len(predecessors) == 1
            parent = self.graph.node[predecessors[0]]
            edge = self.graph.edge[predecessors[0]][basin]

            logXp = parent['logX']
            logX2p = parent['logX2']
            branchP = parent['branchPi'][basin]
            branchP2 = parent['branchPiPj'][basin][basin]

            if edge.has_key('run'):
                run = edge['run']
                nj = run.nlive.astype(float)
            else:
                nj = np.array([])

            lognjsum = np.log(nj).sum()
            lognj1sum = np.log(nj + 1).sum()
            lognj2sum = np.log(nj + 2).sum()
            logXedge = lognjsum - lognj1sum
            logX2edge = lognjsum - lognj2sum

            node['logX'] = logXp + log(branchP) + logXedge
            node['logX2'] = logX2p + log(branchP2) + logX2edge
            node['X'] = exp(node['logX'])
            node['X2'] = exp(node['logX2'])
            node['Phi'] = parent['Phi']
        else:
            node['logX'] = 0.
            node['logX2'] = 0.
            node['X'] = 1.
            node['X2'] = 1.
            node['Phi'] = Phi

        return node
