
from itertools import chain, izip

import numpy as np

import networkx as nx
from networkx.algorithms.traversal import bfs_successors, bfs_edges

import matplotlib.pyplot as plt

from nestedbasinsampling.disconnectivitydatabase import \
    Minimum, Replica, Run, Database
    
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

    def __iter__(self):
        return iter(self.replicas)

    def __add__(self, new):
        new = [new] if type(new) is Replica else new
        replicas = self.replicas.union(new)
        return self.__class__(replicas=replicas)

    def __hash__(self):
        return hash(self.replicas)

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
                                       Erep=rep.energy, 
                                       Emin=rep.minimum.energy)

    def addRuns(self, runs):
        for run in runs:
            self.repGraph.add_edge(run.parent, run.child, run=run, type='run',
                                   Ecut=run.parent.energy, 
                                   Emin=run.child.energy)

    def replicas(self, order=True):
        if order:
            return sorted((n for n in self.repGraph.nodes() 
                           if type(n) is Replica), key=lambda r: r.energy)
        else:
            return [n for n in self.repGraph.nodes() if type(n) is Replica]

    def minima(self, order=True):
        if order:
            return sorted((n for n in self.repGraph.nodes() 
                           if type(n) is Minimum), key=lambda r: r.energy)
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
        return pred1.intersection(sum(bfs_edges(self.repGraph.reverse(), 
                                                node1),()))

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


class BasinGraph(object):
    """ This class joins replicas in the ReplicaClass together
    as a set of super basins.
    """

    def __init__(self, graph):
        self.repGraph = graph
        self.initialise()

    def initialise(self):

        self.graph = nx.DiGraph()

        replicas = self.repGraph.replicas()
        minima = self.repGraph.minima()

        self.repnodes = dict((rep, SuperBasin([rep])) for rep in replicas)
        self.repnodes.update((m,m) for m in minima)

        for node in self.repnodes.values():
            self.add_node(node)
        for parent, child in self.repGraph.repGraph.edges_iter():
            self.add_edge(self.repnodes[parent], self.repnodes[child])

    def add_node(self, node, **kwargs):
        self.graph.add_node(node, energy=node.energy, **kwargs)

    def add_edge(self, parent, child, **kwargs):
        self.graph.add_edge(parent, child, Eparent=parent.energy,
                                 Echild=child.energy, **kwargs)

    def update(self, newGraph=False):
        if newGraph:
            return self.initialise()

        replicas = self.repGraph.replicas()
        minima = self.repGraph.minima()

        newrepnodes = dict((rep, SuperBasin([rep])) for rep in replicas
                           if rep not in self.repnodes)
        newrepnodes.update((m,m) for m in minima if m not in self.repnodes)

        self.repnodes.update(newrepnodes)

        for node in newrepnodes.itervalues():
            self.add_node(node)

        for key in newrepnodes.iterkeys():
            edges = self.repGraph.repGraph.edge[key]
            for childkey in edges:
                self.add_edge(self.repnodes[key], self.repnodes[childkey])

    def joinBasins(self, basins):
        newbasin = sum(basins, [])

        self.graph.add_node(newbasin)

        predecessors = set(sum((self.graph.predecessors(b)
                               for b in basins), [])).difference(basins)
        successors = set(sum((self.graph.successors(b)
                              for b in basins), [])).difference(basins)

        for parent in predecessors:
            self.add_edge(parent, newbasin)
        for child in successors:
            self.add_edge(newbasin, child)
            
    def get_lowest_basin(self):
        return min( (node for node in self.graph.nodes() 
                     if type(node) is SuperBasin), key=lambda n: n.energy)
            
    def basins(self, order=True):
        if order:
            basins = sorted(
                (node for node in self.graph.nodes() 
                 if type(node) is SuperBasin), key=lambda n: n.energy)
        else:
            basins = [node for node in self.graph.nodes() 
                      if type(node) is SuperBasin]
        return basins
        
    def minima(self, order=True):
        if order:
            minima = sorted(
                (node for node in self.graph.nodes() 
                 if type(node) is Minimum), key=lambda n: n.energy)
        else:
            minima = [node for node in self.graph.nodes() 
                      if type(node) is Minimum]
        return minima
    
    def getBasinMinimaDistribution(self, basin, f=1.):
        if type(basin) is Minimum:
            return [(basin, f)]
        else:
            successors = self.graph.successors(basin)
            if successors:
                newf = f / len(successors)
                return sum( (self.getBasinMinimaDistribution(s, newf)
                             for s in successors), [])
            else:
                return []

    def draw(self, **kwargs):
        pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')
        nx.draw(self.graph, pos, **kwargs)
        
        
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
