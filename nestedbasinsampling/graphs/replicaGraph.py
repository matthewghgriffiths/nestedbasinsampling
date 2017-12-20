# -*- coding: utf-8 -*-

from itertools import chain, izip, groupby
import numpy as np

import networkx as nx

import matplotlib.pyplot as plt

from pele.utils.disconnectivity_graph import DisconnectivityGraph

from nestedbasinsampling.storage import (
    Minimum, Replica, Run, Path, TransitionState, Database)
from nestedbasinsampling.sampling.stats import AndersonDarling, CDF
from nestedbasinsampling.nestedsampling.combine import combineAllRuns
from nestedbasinsampling.utils import (
    dict_update_copy, iter_minlength, len_iter, GraphError)
from nestedbasinsampling.utils.sortedcollection import SortedCollection

class ReplicaGraph(object):
    """This object wraps around the database that the results of the
    nested basin-sampling are stored in and a directed networkx
    graph to allow more straightforward interaction with the results.

    It is recommended to interact with the database through this class

    Attributes
    ----------
    database : Database
        The sqlalchemy database where the Replicas, Runs and Minimum
        are stored
    graph : networkx.DiGraph
        The directed graph that stores the relationships between the
        replicas. The nodes of the graph are replicas or minima, and
        the edges are nested optimisation runs or associations between
        replicas and minima

    Methods
    -------
    Minimum(energy, coords)
        Creates a new minimum and adds it to the database and graph
    Replica(energy, coords, minimum=None, stepsize=None)
        Creates a new replica and adds it to the database and graph
        can optionally associate it with a minimum or a stepsize
    Run(Emax, nlive, parent, child,
        volume=1., configs=None, stepsizes=None)
        Creates a new nested run and adds it to the database and graph
    """

    def __init__(self, database=None, run_adder=None, rep_adder=None,
                 min_adder=None, path_adder=None, Etol=1e-5):

        self.Etol = Etol

        database = Database() if database is None else database
        # These methods will be used to create new objects
        self.NewRun = database.addRun     if run_adder is None else run_adder
        self.NewRep = database.addReplica if rep_adder is None else rep_adder
        self.NewMin = database.addMinimum if min_adder is None else min_adder
        self.NewPath = database.addPath if path_adder is None else path_adder

        self.on_minimum_added = []


        self.Emax = None
        self.disconnect_kw = {}

        # Create new graph from database
        self.loadFromDatabase(database, newGraph=True)

    def loadFromDatabase(self, database, newGraph=True):
        """
        NOT TESTED
        """
        if newGraph:
            self.graph = nx.DiGraph()
            self.tree = nx.DiGraph()
            self._connectgraph = nx.Graph()

        self.database = database
        self.runs = self.database.runs
        self.paths = self.database.paths

        [self.addMinimum(m) for m in self.database.minima()]
        [self.addReplica(rep) for rep in self.database.replicas()]
        [self.addRun(run) for run in self.database.runs()]
        [self.addPath(path) for path in self.database.paths()]

    def replicas(self, order=True, Esplit=None, noSuccessors=False):
        """
        Parameters
        ----------
        order : bool, optional
            If true orders the replicas by energy
        Esplit : float, optional
            If true then return the replicas within the energy band specified
            by Esplit and self.Esplits
        noSuccessors : bool, optional
            If true then only returns replicas with no other successor replicas
            in the returned list

        Returns
        -------
        replicas : list of Replica
            List of replicas satisfying the above options
        """

        if Esplit is not None:
            Esplits = self.Esplits
            if Esplits is None:
                maxE = np.inf
                minE = - np.inf
                if self.debug:
                    print "ReplicaGraph> Warning: no energy splits present"
            else:
                i = Esplits.searchsorted(Esplit, side='right')
                maxE = np.inf if i+1 > Esplits.size else Esplits[i]
                minE = - np.inf if i == 0 else Esplits[i-1]

            replicas = (n for n in self.graph.nodes() if
                       type(n) is Replica and minE <= n.energy < maxE)

        else:
            replicas = (n for n in self.graph.nodes() if type(n) is Replica)

        if noSuccessors:
            # Make a set of replicas within the energy bounds
            allreps = set(replicas)
            # Return only the replicas with no successors
            replicas = ( r for r in allreps if not
                         len(allreps.intersection(self.graph.successors(r))))

        if order:
            return sorted(replicas, key=lambda r: r.energy)
        else:
            return list(replicas)

    def minima(self, order=True):
        if order:
            return sorted((n for n in self.graph.nodes()
                           if type(n) is Minimum), key=lambda r: r.energy)
        else:
            return [n for n in self.graph.nodes() if type(n) is Minimum]

    def Minimum(self, energy, coords):
        m = self.NewMin(energy, coords)
        self.addMinimum(m)
        return m

    def Replica(self, energy, coords, stepsize=None):
        rep = self.NewRep(energy, coords, stepsize=stepsize)
        self.addReplica(rep)
        return rep

    def Run(self, Emax, nlive, parent, child, intermediates=[],
            stored=None, configs=None, stepsizes=None):
        """
        """
        run = self.NewRun(Emax, nlive, parent, child,
                          stored=stored, configs=configs,
                          stepsizes=stepsizes)
        self.addRun(run, intermediates)
        return run

    def Path(self, energy, parent, child, intermediates=[], energies=None,
             stored=None, configs=None, quench=False, minimum=False, **kwargs):
        """
        """
        path = self.NewPath(
            energy, parent, child, energies=energies,
            stored=stored, configs=configs, quench=quench, minimum=minimum)
        self.addPath(path, intermediates=intermediates, **kwargs)
        return path

    @property
    def Esplits(self):
        prop = self.database.get_property("Esplits")
        if prop is None:
            return np.array([])
        else:
            return prop.value

    @Esplits.setter
    def Esplits(self, Esplits):
        prop = self.database.get_property('Esplits')
        if prop is not None:
            prop.value = Esplits
            self.database.session.commit()
        else:
            self.database.add_property('Esplits', Esplits)


    def addMinimum(self, m):
        """
        """
        self.graph.add_node(m, energy=m.energy)
        self.tree.add_node(m, energy=m.energy)

        """
        replicas = sorted(set(self.genPreceedingReplicas(m)))
        for r1, r2 in izip(replicas[1:], replicas[:-1]):
            if not self.graph.has_edge(r1, r2):
                self.graph.add_edge(
                    r1, r2, energy=r1.energy, Emin=r2.energy)
        """
        for event in self.on_minimum_added:
            event(m)

    def addReplica(self, replica):
        """
        """
        self.graph.add_node(replica, energy=replica.energy)
        self.tree.add_node(replica, energy=replica.energy)

    def addPath(self, path, intermediates=[], **kwargs):
        """Adds path to graph will connect parent to child
        through intermediates"""
        middle = sorted(intermediates)
        tojoin = [path.parent] + middle + [path.child]
        if path.quench:
            kwargs['quench'] = True
        elif path.minimum:
            kwargs['minimum'] = True
        for r1, r2 in izip(tojoin[:-1], tojoin[1:]):
            self.graph.add_edge(
                r1, r2, path=path, energy=r1.energy, Emin=r2.energy, **kwargs)
            self.tree.add_edge(r1, r2)
            #self.connectTreeReplicas(r1, r2)

    def addRun(self, run, intermediates=[]):
        """Adds run to graph will connect parent to child
        through intermediates"""
        middle = sorted(intermediates)
        tojoin = [run.parent] + middle + [run.child]
        for r1, r2 in izip(tojoin[:-1], tojoin[1:]):
            self.graph.add_edge(r1, r2, run=run,
                                energy=run.parent.energy,
                                Emin=run.child.energy)
            self.tree.add_edge(r1, r2)
            #self.connectTreeReplicas(r1, r2)

    def addPartialRun(self, replica, res, saveConfigs=False, nsave=1):
        """
        """
        Es = res.Emax
        nlive = (res.nlive if hasattr(res, 'nlive') else
                 np.ones_like(Es, dtype=int))

        stepsizes, configs, stored = None, None, None
        if nsave:
            if hasattr(res, 'stepsize'):
                stepsizes = res.stepsize[::nsave]
                n = res.stepsize.size
            if saveConfigs and hasattr(res, 'coords_s'):
                configs = res.coords_s[::nsave]
                n = res.coords_s.size
            if n is not None:
                stored = np.arange(0,n,nsave)

        if Es.size:
            child = self.Replica(res.energy, res.coords, stepsize=res.stepsize)
            run = self.Run(Es, nlive, replica, child,
                           stored=stored, stepsizes=stepsizes, configs=configs)
            return child, run
        else:
            return None, None

    def addFullRun(self, replica, res, saveConfigs=False, nsave=1):
        Es = res.Emax
        nlive = (res.nlive if hasattr(res, 'nlive') else
                 np.ones_like(Es, dtype=int))

        stepsizes, configs, stored = None, None, None
        if nsave:
            if hasattr(res, 'stepsize'):
                stepsizes = res.stepsize[::nsave]
                n = res.stepsize.size
            if saveConfigs and hasattr(res, 'coords_s'):
                configs = res.coords_s[::nsave]
                n = res.coords_s.size
                childcoords = res.coords_s[-1]
            else:
                childcoords = None
            if n is not None:
                stored = np.arange(0,n,nsave)

        if Es.size:
            child = self.Replica(Es[-1], childcoords, res.stepsize)
            m = self.Minimum(res.energy, res.coords)
            run = self.Run(Es, nlive, replica, child,
                           stored=stored, stepsizes=stepsizes, configs=configs)
            path = self.Path(child.energy, child, m, minimum=True)
            # Updating tree structure
            self.tree.add_edge(replica, child)
            self.connectTreeReplicas(child, m)
            return child, m, run, path
        else:
            return None, None, None, None


    def addMinimization(self, replica, startrep, res,
                        useQuench=False, saveConfigs=False, nsave=1):
        """
        """
        Es = res.energy_s if hasattr(res, 'energy_s') else None
        argsort = np.array([], int) if Es is None else np.arange(Es.size)

        stepsizes = None
        configs = None
        stored = None
        n = None

        if not useQuench:
            Es = res.Emax
            nlive = (res.nlive if hasattr(res, 'nlive') else
                     np.ones_like(Es, dtype=int))
            argsort = np.arange(Es.size)
            if nsave:
                if hasattr(res, 'stepsize'):
                    stepsizes = res.stepsize[argsort[::nsave]]
                    n = res.stepsize.size
                if saveConfigs and hasattr(res, 'coords_s'):
                    configs = res.coords_s[argsort[::nsave]]
                    n = res.coords_s.size
                if n is not None:
                    stored = np.arange(0,n,nsave)
        else:
            if Es is not None:
                if not (Es[:-1] > Es[1:]).all():
                    argsort = Es.size - Es[::-1].argsort()[::-1] - 1
                    Es = Es[argsort]

        if nsave:
            if saveConfigs and hasattr(res, 'coords_s'):
                configs = res.coords_s[argsort[::nsave]]
                n = res.coords_s.size
            if n is not None:
                stored = np.arange(0,n,nsave)

        m = self.Minimum(res.energy, res.coords)

        if useQuench:
            run = self.Run([startrep.energy], [1], replica, startrep)
            path = self.Path(startrep.energy, startrep, m, quench=True,
                             stored=stored, energies=Es, configs=configs)
            self.tree.add_edge(replica, startrep)
            self.connectTreeReplicas(startrep, m)
        else:
            repcoords = res.coords if configs is None else configs[-1]
            child = self.Replica(Es[-1], repcoords)

            Es = np.r_[startrep.energy, Es]
            nlive = np.r_[1, nlive]
            if stored.size:
                stored += 1

            run = self.Run(Es, nlive, replica, child,
                           intermediates=[startrep], stored=stored,
                           stepsizes=stepsizes, configs=configs)
            path = self.Path(child.energy, child, m, minimum=True)

            # Updating tree structure
            self.tree.add_edge(replica, startrep)
            self.tree.add_edge(startrep, child)
            self.connectTreeReplicas(child, m)

        return startrep, m, run, path

    def connectTreeReplicas(self, parent, child, check=True):
        """ This updates the structure of self.tree such that parent and
        child are connected on the the tree but self.tree maintains
        its topology as a tree
        """
        if child > parent:
            parent, child = child, parent

        if self.tree.in_degree(child) == 0:
            self.tree.add_edge(parent, child)

        # Finding all predecessors
        toconnect = set()
        predecessors = SortedCollection(
            [parent, child], key=lambda r: -r.energy)
        while predecessors:
            r1 = predecessors.pop()
            toconnect.add(r1)
            for r2 in self.tree.predecessors_iter(r1):
                predecessors.add(r2)

        #The new order
        newbranch = sorted(toconnect)

        # Ensuring that each replica only has the single correct predecessor
        for r1, r2 in izip(newbranch[1:], newbranch[:-1]):
            for pre in self.tree.predecessors(r2):
                if pre != r1:
                    self.tree.remove_edge(pre, r2)
            if not self.tree.has_edge(r1, r2):
                self.tree.add_edge(r1, r2)

        assert nx.has_path(self.tree, parent, child)

        if check and any(self.tree.in_degree(r) > 1 for r in self.tree.nodes()):
            raise GraphError('ReplicaGraph.tree is no longer a tree')

    def fixTree(self):
        """
        """
        while not nx.is_forest(self.tree):
            for run in self.runs():
                self.connectTreeReplicas(run.parent, run.child, check=False)
            for path in self.paths():
                self.connectTreeReplicas(path.parent, path.child, check=False)

    def treeBranch(self, replica):
        try:
            while True:
                yield replica
                replica, = self.tree.predecessors_iter(replica)
        except ValueError:
            pass

    def genSamples(self, replica):
        runs = self.genConnectedRuns(replica)
        for run in runs:
            if run.parent == replica:
                if 0 in run.stored:
                    i = run.stored.searchsorted(0, side='left')
                    if run.configs.size:
                        yield run.configs[i]
                elif len(run.Emax) == 1:
                    yield run.child.coords
            else:
                Es = run.Emax[::-1]
                if run.stored.size:
                    i = Es.size - Es.searchsorted(replica.energy, side='left')
                    j = run.stored.searchsorted(i, side='left')
                    if run.stored[j] == i:
                        if run.configs.size:
                            yield run.configs[j]

    def genEnergySamples(self, replica):
        runs = self.genConnectedRuns(replica)
        for run in runs:
            Es = run.Emax[::-1]
            i = Es.searchsorted(replica.energy, side='left')
            i = 0 if i==0 else i-1
            yield Es[i-1]

    def replicastoPath(self, replicas, startrep=None, endrep=None):
        """
        """
        Es = []
        stored = []
        configs = []
        stepsizes = []
        for r1, r2 in izip(replicas[:-1], replicas[1:]):
            edge = self.graph.edge[r1][r2]
            if edge.has_key('run'):
                Es.append(edge['run'].Emax)
                if edge['run'].stepsizes.size:
                    stepsizes.append(edge['run'].stepsizes)
                if edge['run'].configs.size:
                    stored.append(edge['run'].stored)
                    configs.append(edge['run'].configs)
            elif edge.has_key('path'):
                Es.append(edge['path'].energies)
                if edge['path'].configs.size:
                    stored.append(edge['path'].stored)
                    configs.append(edge['path'].configs)

            if len(Es) > 1:
                if Es[-1].size:
                    i = Es[-2][::-1].searchsorted(Es[-1][0], side='right')
                    Es[-2] = Es[-2][:-i]
                    if len(stepsizes) == len(Es):
                        stepsizes[-2] = stepsizes[-2][:-i]
                    if len(configs) == len(Es):
                        stored[-2] = stored[-2]
                        configs[-2] = configs[-2]

        Es = np.concatenate(Es)

        # Shortening path to match the highest and lowest replica energy
        maxrep = max(replicas) if startrep is None else startrep
        minrep = min(replicas) if endrep is None else endrep
        i1 = Es[::-1].searchsorted(maxrep.energy, side='left')
        i2 = Es[::-1].searchsorted(minrep.energy, side='right')
        i2 = None if i2 < 2 else -i2 + 1

        Es = Es[-i1-1:i2]

        if stepsizes:
            stepsizes = np.concatenate(stepsizes)[-i1-1:i2]
        else:
            stepsizes = np.array([])

        stored = np.array([]) # TODO IMPLEMENT THIS!
        configs = np.array([]) # TODO IMPLEMENT THIS!

        path = Path(maxrep.energy, maxrep, minrep, energies=Es,
                    stored=stored, configs=configs, stepsizes=stepsizes)
        return path

    def connectingReplicas(self, source, target, runs=True, paths=False):
        """
        """
        if paths:
            key = 'path'
        elif runs:
            key = 'run'
        else:
            key = 'energy'

        G = self.graph
        cutoff = len(self.graph)
        visited = [source]
        stack = [iter(G[source])]
        while stack:
            children = stack[-1]
            child = next(children, None)

            while child is not None:
                if G[visited[-1]][child].has_key(key):
                    break
                elif child == target: # if target is a minimum
                    break
                child = next(children, None)

            if child is None:
                stack.pop()
                visited.pop()
            elif len(visited) < cutoff:
                if child == target:
                    yield visited + [target]
                elif child not in visited:
                    visited.append(child)
                    stack.append(iter(G[child]))
            else:
                if child == target or target in children:
                    yield visited + [target]
                stack.pop()
                visited.pop()

    genConnectingReplicas = connectingReplicas

    def connectingRuns(self, source, target):
        for replicas in self.connectingReplicas(source, target, runs=True):
            yield self.replicastoRun(replicas)

    genConnectingRuns = connectingRuns

    def replicastoRun(self, replicas):
        """
        """
        runs = []
        for u, v in izip(replicas[:-1], replicas[1:]):
            edge = self.graph.edge[u][v]
            if edge.has_key('run'):
                runs.append(edge['run'].split(u, v))
            else:
                break
        return combineAllRuns(runs)

    def findConnectedRuns(self, replica, Estart=None):
        """ returns an iterator of all the runs that are connected to
        replica that finish after Estart, can return the same run more than
        once
        """
        Estart = np.inf if Estart is None else Estart
        replicas = list(self.treeBranch(replica))
        return (edge['run'] for r1 in replicas
            for r2, edge in self.graph[r1].iteritems()
            if r2.energy <= Estart if edge.has_key('run'))

    def connectedRuns(self, replica, Efilter=None):
        """
        """
        for edge in self.graph.edge[replica].itervalues():
            if edge.has_key('run'):
                run = edge['run']
                Emax = run.Emax
                if Efilter is None:
                    yield run
                elif True:
                    if run.parent.energy > Efilter >= run.child.energy:
                        yield run
                elif len(Emax) > 1 and Emax[0] > Efilter >= Emax[-1]:
                    yield run

    genConnectedRuns = connectedRuns

    def preceedingReplicas(self, replica, runs=True, paths=False):
        """
        """
        for r1 in self.graph.predecessors_iter(replica):
            if paths:
                if self.graph[r1][replica].has_key('path'):
                    for r2 in self.preceedingReplicas(r1, paths=True):
                        yield r2
                    yield r1
            elif runs:
                if (self.graph[r1][replica].has_key('run') or
                    isinstance(replica, Minimum)):
                    for r2 in self.preceedingReplicas(r1, runs=True):
                        yield r2
                    yield r1
            else:
                for r2 in self.preceedingReplicas(r1, runs=False):
                    yield r2
                yield r1

    genPreeceedingReplicas = preceedingReplicas

    def succeedingReplicas(self, replica, runs=True, paths=False):
        """
        """
        for r1 in self.graph.successors_iter(replica):

            if paths:
                if not self.graph.edge[replica][r1].has_key('paths'):
                    continue
            elif runs:
                if not self.graph.edge[replica][r1].has_key('runs'):
                    continue

            yield r1
            for r2 in self.genSucceedingReplicas(r1, runs=runs, paths=paths):
                yield r2

    genSucceedingReplicas = succeedingReplicas

    def predecessors_iter(self, replica, runs=True, paths=False):
        """
        """

        if paths:
            key = 'path'
        elif runs:
            key = 'run'
        else:
            key = 'energy'

        for parent in self.graph.predecessors_iter(replica):
            if self.graph[parent][replica].has_key('energy'):
                yield parent

    def successors_iter(self, replica, runs=True, paths=False):
        """ Returns an iterator of successor replicas connected by runs,
        paths or both. Will return minima connected by paths if runs is True.
        """
        edges = self.graph[replica].iteritems()
        if paths:
            successors = (
                child for child, edge in edges if edge.has_key('path'))
        elif runs:
            # Checks whether the children and their children
            # are connected by runs or the child is a minimum
            successors = (
                c1 for c1, edge1 in edges
                if isinstance(c1, Minimum) or (edge1.has_key('run') and  any(
                    isinstance(c2, Minimum) or edge2.has_key('run')
                    for c2, edge2 in self.graph[c1].iteritems())))
        else:
            successors = (child for child, edge in edges)
        return successors

    def successors(self, replica, runs=True, paths=False):
        return list(self.successors_iter(replica, runs=runs, paths=paths))

    def genConnectedMinima(self, replica, f=1., runs=True, paths=False):
        """
        """
        successors = self.successors(replica, runs=runs, paths=paths)
        if successors:
            f /= len(successors)
            for child in successors:
                if isinstance(child, Minimum):
                    yield (child, f)
                else:
                    for m, f in self.genConnectedMinima(
                        child, f, runs=runs, paths=paths):
                        yield m, f
        else:
            return

    def genAllPreceedingReplicas(self, replica):
        """ generates all the replicas that are higher in energy to the
        replica given which are connected to the same minimum
        """
        try:
            replica, = self.tree.predecessors_iter(replica)
            yield replica
        except ValueError:
            pass

    def genAllSucceedingReplicas(self, replica):
        """ generates all the child replicas of the replica which
        connect to the same minima as replica
        """
        for r1 in self.tree.successors_iter(replica):
            yield r1
            for r2 in self.genAllSucceedingReplicas(r1):
                yield r2

    def allConnectedMinima(self, replica):
        """
        """
        for r in self.tree.successors_iter(replica):
            if isinstance(r, Minimum):
                yield r
            else:
                for m in self.genAllConnectedMinima(r):
                    yield m

    genAllConnectedMinima = allConnectedMinima

    def removeReplica(self, replica):
        parent, = self.graph.predecessors(replica)
        pedge = self.graph.edge[parent][replica]
        for child, edge in self.graph[replica].items():
            self.graph.remove_edge(replica, child)
            if edge.has_key('run'):
                run = edge['run']
                if run.parent != replica:
                    self.graph.add_edge(parent, child, energy=parent.energy,
                                        Emin=child.energy, run=run)
                else:
                    self.database.removeRun(run)
                    self.graph.add_edge(parent, child, energy=parent.energy,
                                        Emin=child.energy)
            elif edge.has_key('path'):
                path = edge['path']
                self.graph.add_edge(parent, child, energy=parent.energy,
                                    path=path, Emin=child.energy)
            else:
                self.graph.add_edge(parent, child, energy=parent.energy,
                                    Emin=child.energy)
        self.database.removeReplica(replica)

    def getReplicaMinimaCDFs(self, replicas):
        replicasets = self.connectReplicas(replicas)
        cdfs = []
        for reps, mset in replicasets:
            cdf = reduce(
                lambda x,y: x+y, map(self.getMinimaCDF, reps), CDF([]))
            cdfs.append(cdf)
        return replicasets, cdfs

    def genMinBasins(self, replica):
        """ generates the minima of the basins of attraction
        sampled below replica, the length of the iterator is
        the number of samples taken
        """
        edges = self.graph.edge[replica].iteritems()
        #Samples of the basin of replica will be connected by
        # runs of length 1
        samples = (
            r1 for r1, edge in edges
            if edge.has_key('run') and edge['run'].Emax.size==1)
        # These samples need to have paths
        samples = (
            r1 for r1 in samples if any(
                edge.has_key('quench')
                for edge in self.graph.edge[r1].itervalues()))
        minima = (
            [m for m, f in self.genConnectedMinima(r, runs=False, paths=True)]
            for r in samples)
        minima = (ms for ms in minima if ms)
        return minima

    def getMinBasinCDF(self, replica):
        """ Calculates the CDF of the energies of the basins
        of attraction in the local region of replica
        """
        cdfs = (CDF([m.energy for m in ms], n=1)
                for ms in self.genMinBasins(replica))
        # Calculating the CDF
        return reduce(lambda x, y: x+y, cdfs, CDF([]))

    def getMinimaCDF(self, replica):
        """
        """
        return CDF(*izip(*(
            (m.energy, f)
            for m,f in self.genConnectedMinima(replica, runs=True))
            ), n=len_iter(self.successors_iter(replica, runs=True)))

    def compareMinimaDistributions(self, replicas):
        cdfs = map(self.getMinimaCDF, replicas)
        return AndersonDarling.compareDistributions(cdfs)

    def findEnergySplit(self, replica):
        Esplits = np.r_[self.Esplits, np.inf]
        return Esplits.searchsorted(replica.energy, side='right')

    def addAllIntermediateReplicas(self):
        for r1, edges in self.graph.edge.items():
            for r2, edge in edges.iteritems():
                i1 = self.findEnergySplit(r1)
                i2 = self.findEnergySplit(r2)
                if i1 - i2 > 1:
                    if edge.has_key('run'):
                        run = edge['run']
                        self.addIntermediateReplicas(run, r1, r2)

    def splitReplicaPair(self, parent, child, Ecut):
        """
        """
        if parent.energy == Ecut:
            return parent
        elif child.energy == Ecut:
            return child

        assert parent.energy > Ecut > child.energy

        attr = {}
        if self.graph.has_edge(parent, child):
            edge = self.graph.edge[parent][child]
            if edge.has_key('run'):
                run = edge['run']
                Es = run.Emax[::-1]
                stored = - run.stored[::-1]
                configs = run.configs[::-1]
                stepsizes = run.stepsizes[::-1]
                attr['run'] = run
            elif edge.has_key('path'):
                path = edge['path']
                Es = path.energies[::-1]
                stored = - path.stored[::-1]
                configs = path.configs[::-1]
                stepsizes = np.array([])
                attr['path'] = path
            else:
                Es = np.array([])
                stored = np.array([], dtype=int)
            self.graph.remove_edge(parent, child)
            stored += stored.size - 1
        else:
            Es = np.array([])
            stored = np.array([], dtype=int)

        coords = child.coords
        stepsize = None
        Esplit = Ecut
        if Es.size:
            isplit = Es.searchsorted(Ecut, side='left')
            Esplit = Es[isplit] #if isplit < Es.size else Ecut
            if stored.size:
                jsplit = stored.searchsorted(isplit, side='right')
                if configs.size == stored.size:
                    coords = configs[jsplit]
                if stepsizes.size == stored.size:
                    stepsize = stepsizes[jsplit]
                elif stepsizes.size == Es.size:
                    stepsize = stepsizes[isplit]


        newrep = self.Replica(Esplit, coords, stepsize=stepsize)

        attr.update(energy=parent.energy, Emin=newrep.energy)
        self.graph.add_edge(parent, newrep, **attr)
        attr.update(energy=newrep.energy, Emin=child.energy)
        self.graph.add_edge(newrep, child, **attr)

        replica = child

        while True:
            parent, = self.tree.predecessors_iter(replica)
            if parent > newrep:
                self.tree.remove_edge(parent, replica)
                self.tree.add_edge(parent, newrep)
                self.tree.add_edge(newrep, replica)
                break
            else:
                replica = parent

        #assert nx.is_branching(self.tree)
        assert not any(self.tree.in_degree(r) > 1 for r in self.tree.nodes())

        return newrep

    def splitReplicas(self, replicas, Ecut, sort=True):
        """
        """
        if sort:
            replicas.sort(reverse=True)
        assert replicas[0].energy >= Ecut > replicas[-1].energy

        parent = replicas[0]
        for child in replicas[1:]:
            if child.energy < Ecut:
                break
            else:
                assert child < parent
                parent = child

        return self.splitReplicaPair(parent, child, Ecut)

    def calcConnectivityGraph(self, Emax=None):
        """
        """
        g = nx.Graph()

        minima = self.minima()
        setdict = dict((m, set([m])) for m in minima)
        replicas = SortedCollection(minima, key=lambda r: -r.energy)

        while replicas:
            replica = replicas.pop()
            successors = self.tree.successors(replica)

            if len(successors) > 1:
                minima = [list(self.genAllConnectedMinima(r))
                          for r in successors]
                for i, m1s in enumerate(minima):
                    for m2s in minima[:i]:
                        for m1 in m1s:
                            for m2 in m2s:
                                ts = TransitionState(replica.energy, None,
                                                     m1, m2)
                                g.add_edge(m1, m2, ts=ts)
                setdict[replica] = set(m for ms in minima for m in ms)
            elif len(successors) == 1:
                setdict[replica] = setdict[successors[0]]

            parents = self.tree.predecessors(replica)
            if parents:
                replicas.insert(parents[0])
        return g


    def calcDisconnectivityGraph(self, g=None, **kwargs):
        """
        """
        g = self.calcConnectivityGraph(self) if g is None else g
        dg = DisconnectivityGraph(g, **kwargs)
        dg.calculate()
        return dg

    @property
    def disconnectivityGraph(self):
        """
        """
        minima = self.minima()
        if self.Emax is not None:
            minima = filter(lambda m: m.energy < self.Emax, minima)

        if set(self._connectgraph.nodes()) != set(minima):
            self._connectgraph = self.calcConnectivityGraph(self.Emax)
            self._disconnectgraph = DisconnectivityGraph(self._connectgraph,
                                                         **self.disconnect_kw)
            self._disconnectgraph.calculate()

        return self._disconnectgraph

    def plot(self, recalc=False, axes=None, **kwargs):
        """
        """
        if axes is None:
            axes=plt.gca()
        kwargs.update(axes=axes)

        dg = self.calcDisconnectivityGraph() if recalc else self.disconnectivityGraph
        dg.plot(**kwargs)
        return dg

    def draw(self, tree=True, energies=True,
             maxE=0., arrows=False, node_size=5, **kwargs):
        """
        """
        if tree:
            pos = nx.nx_agraph.graphviz_layout(self.tree, prog='dot')
            if energies:
                pos = dict((r, (p[0], np.clip(r.energy, None, maxE)))
                            for r,p in pos.iteritems())
            else:
                replica = max(self.replicas())
                pos[replica] = (pos[replica][0], 0)
                basins = set([replica])
                while basins:
                    replica = basins.pop()
                    for r in self.tree.successors_iter(replica):
                        pos[r] = (pos[r][0], pos[replica][1] - 1)
                        basins.add(r)
            nx.draw(self.tree, pos, arrows=arrows,
                    node_size=node_size, **kwargs)
        else:
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')
            if energies:
                pos = dict((r, (p[0], np.clip(r.energy, None, maxE)))
                            for r,p in pos.iteritems())
            nx.draw(self.graph, pos, arrows=arrows,
                    node_size=node_size, **kwargs)














    # TODO MARKING END OF WORKING CODE

if 0:

    def getCombinedRun(self, replica, run=None):
        """
        Parameters
        ----------
        replica : Replica
            The starting point of the nested sampling runs
        run : Run, optional
            The run that connects to the starting replica, if not defined
            then a 0-length run is generated

        Returns
        -------
        combined run : Run
            The combined nested run of all the nested runs that start from the
            starting replica
        """
        runs = []
        for rep, edge in self.graph.edge[replica].iteritems():
            if edge.has_key('run'):
                newrun = joinRuns(run, edge['run'])
                runs.append(self.getCombinedRun(rep, newrun))
        if len(runs) == 0:
            return Run([], [], replica, replica)
        else:
            return combineAllRuns(runs)

    def _genConnectedMinima(self, replica, f=1.):
        """
        """
        if type(replica) is Minimum:
            yield (replica, f)
        else:
            connected = self.graph.successors(replica)
            newf = f / len(connected)
            for rep in connected:
                for mf in self.genConnectedMinima(rep, newf):
                    yield mf

    def genConnectedMinima2(self, replica, f=[]):
        """
        """
        if type(replica) is Minimum:
            yield (replica, f)
        else:
            connected = self.graph.successors(replica)
            newf = f + [(1, len(connected))]
            for rep in connected:
                for mf in self.genConnectedMinima2(rep, f=newf):
                    yield mf

    def checkSameBasin(self, replica1, replica2):
        min1 = set(mf[0] for mf in self.genConnectedMinima(replica1))
        return any(mf[0] in min1 for mf in self.genConnectedMinima(replica2))

    def connectReplicas(self, replicas):

        replicamins = [set(mf[0] for mf in self.genConnectedMinima(r))
                        for r in replicas]
        replicasets = [[{replicas[0]},replicamins[0]]]

        for rep, mins in izip(replicas[1:], replicamins[1:]):
            inset = False
            for reps, minset in replicasets:
                if rep in reps:
                    break
                else:
                    if len(minset.intersection(mins)):
                        reps.update([rep])
                        minset.update(mins)
                        inset = True
            if not inset:
                replicasets.append([{rep}, mins])

        return replicasets


    def splitReplicas(self, parent, child, splits):

        splits = sorted(splits, key=lambda rep: rep.energy)

        assert child.energy < splits[0].energy
        assert parent.energy > splits[-1].energy

        tojoin = [child] + splits + [parent]

        if self.graph.has_edge(parent, child):
            edge = self.graph.edge[parent][child]
            run = edge.get('run', None)
            self.graph.remove_edge(parent, child)
        else:
            run = None

        for r1, r2 in izip(tojoin[1:], tojoin[:-1]):
            attr = dict(energy=r1.energy,Emin=r2.energy)
            if run is not None: attr['run'] = run
            self.graph.add_edge(r1, r2, **attr)


    def addIntermediateReplicas(self, run, parent, child):
        """
        Creates a new replica and inserts into graph if there are
        any stored configurations between parent and child in run
        """
        stored = run.stored
        if stored.size:
            Emax = run.Emax
            configs = run.configs
            stepsizes = run.stepsizes
            Esplits = self.Esplits
            i1 = self.findEnergySplit(parent)
            i2 = self.findEnergySplit(child)
            isplits = Emax.size - Emax[::-1].searchsorted(
                Esplits[i2:i1][::-1], side='left') - 1
            js = np.unique(stored.searchsorted(isplits,
                                               side='left'))
            inds = stored[js]

            if inds.size:
                if self.graph.has_edge(parent, child):
                    self.graph.remove_edge(parent, child)

                lastrep = parent
                for i, j in izip(inds, js):
                    stepsize = stepsizes[j] if stepsizes.size else None
                    newrep = self.Replica(Emax[i], configs[j], stepsize=stepsize)
                    self.graph.add_edge(lastrep, newrep, run=run,
                                        energy=lastrep.energy, Emin=newrep.energy)
                    lastrep = newrep
                self.graph.add_edge(lastrep, child, run=run,
                                energy=lastrep.energy, Emin=child.energy)

    def insertEnergySplit(self, Esplit):

        Esplits = self.Esplits
        if len(Esplits):
            if np.isclose(Esplit, Esplits, atol=self.Etol).any():
                return [], [], []
            else:
                Esplits = np.sort(np.r_[Esplits, Esplit])
        else:
            Esplits = np.array([Esplit]).astype(float)
        self.Esplits = Esplits

        self.addAllIntermediateReplicas()

    def splitRun(self, run, Esplit, replace=False, above=True):
        """
        """
        configs = run.configs
        nlive = run.nlive
        stepsizes = run.stepsizes
        Emax = run.Emax
        stored = run.stored
        volume = run.volume
        parent = run.parent
        child = run.child

        isplit = Emax.size - np.searchsorted(Emax[::-1], Esplit, side='left')
        isplit = isplit - 1 if above else isplit
        # If we haven't stored all the configurations, find closest config
        j = stored.searchsorted(isplit, side='left')
        i = stored[j]

        if replace:
            self.database.removeRun(run)
            NewRep = self.NewRep
            NewRun = self.NewRun
        else:
            NewRep = Replica
            NewRun = Run



        if j == 0: ## Can't split the run
            newrep = parent
            run1 = Run([], [], newrep, newrep)
            run2 = run
        elif isplit == Emax.size: ## Can't split the run
            newrep = child
            run1 = run
            run2 = Run([], [], newrep, newrep)
        else:
            stored1 = stored[:j+1]
            stored2 = stored[j+1:] - i - 1
            if len(stepsizes) == len(configs):
                newrep = NewRep(Emax[i], configs[i], stepsize=stepsizes[j])
                run1 =   NewRun(Emax[:j+1], nlive[:j+1], parent, newrep,
                                volume=volume, stored=stored1,
                                configs=configs[:j+1],
                                stepsizes=stepsizes[:j+1])
                # recalculate vol?
                run2 =   NewRun(Emax[i+1:], nlive[i+1:], newrep, child,
                                configs=configs[j+1:], stored=stored2,
                                stepsizes=stepsizes[j+1:])
            else:
                newrep = NewRep(Emax[i], configs[j])
                run1 =   NewRun(Emax[:i+1], nlive[:i+1], parent, newrep,
                                volume=volume, stored=stored1,
                                configs=configs[:j+1])
                run2 =   NewRun(Emax[i+1:], nlive[i+1:], newrep, child,
                                stored=stored2, configs=configs[j+1:])
            if replace:
                # Making change to graph
                self.graph.remove_edge(parent, child)
                self.addReplica(newrep)
                self.addRun(run1)
                self.addRun(run2)

        return run1, run2, newrep


    def _Run(self, Emax, nlive, parent, child, volume=1.,
            stored=None, configs=None, stepsizes=None, above=True,
            save_configs=True):

        if save_configs:
            run = self.NewRun(Emax, nlive, parent, child, volume=volume,
                              stored=stored , configs=configs,
                              stepsizes=stepsizes)
        else:
            run = self.NewRun(Emax, nlive, parent, child, volume=volume)

        # Adding intermediate replicas if they exist
        if stored is not None:
            Esplits = self.Esplits
            isplits = Emax.size - Emax[::-1].searchsorted(
                Esplits[::-1], side='left')
            i1, i2 = Esplits.searchsorted([Emax[-1], Emax[0]], side='left')
            isplits = Emax.size - Emax[::-1].searchsorted(
                Esplits[i1:i2], side='left')
            if above:
                isplits -= 1
                isplits = isplits[1:] if 0 in isplits else isplits
            js = np.unique(stored.searchsorted(isplits, side='left'))
            inds = stored[js]
        else:
            inds = []
            js = []

        lastrep = parent
        for i, j in izip(inds, js):
            stepsize = None if stepsizes is None else stepsizes[j]
            newrep = self.Replica(Emax[i], configs[j], stepsize=stepsize)
            self.graph.add_edge(lastrep, newrep, run=run,
                                energy=lastrep.energy, Emin=newrep.energy)
            lastrep = newrep

        self.graph.add_edge(lastrep, child, run=run,
                            energy=lastrep.energy, Emin=child.energy)
        return run
    def mergeMinima(self, min1, min2):
        raise NotImplementedError
