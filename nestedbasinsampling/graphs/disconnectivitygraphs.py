
from itertools import chain, izip, groupby

from math import exp, log, sqrt

import numpy as np

from scipy.special import gammaln
from scipy.integrate import quad

import networkx as nx

import matplotlib.pyplot as plt

from nestedbasinsampling.disconnectivitydatabase import \
    Minimum, Replica, Run, Database
from nestedbasinsampling.random.stats import AndersonDarling, CDF

from nestedbasinsampling.nestedsampling import findRunSplit, joinRuns, combineAllRuns
from nestedbasinsampling.utils import dict_update_copy, iter_minlength

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

        # Create new graph from database
        self.loadFromDatabase(database, newGraph=True)

    def loadFromDatabase(self, database, newGraph=True):
        if newGraph:
            self.graph = nx.DiGraph()

        self.database = database

        [self.addMinimum(m) for m in self.database.minima()]
        [self.addReplica(rep) for rep in self.database.replicas()]
        [self.addRun(run) for run in self.database.runs()]
        [self.addPath(path) for path in self.database.paths()]

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

    def Minimum(self, energy, coords):
        m = self.NewMin(energy, coords)
        self.addMinimum(m)
        return m

    def Replica(self, energy, coords, stepsize=None):
        rep = self.NewRep(energy, coords, stepsize=stepsize)
        self.addReplica(rep)
        return rep

    def Run(self, Emax, nlive, parent, child, volume=1.,
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

    def Path(self, energy, parent, child, energies=None, configs=None):
        path = self.NewPath(energy, parent, child, energies, configs)
        self.addPath(path)
        return path

    def addMinimum(self, m):
        self.graph.add_node(m, energy=m.energy)

    def addReplica(self, replica):
        self.graph.add_node(replica, energy=replica.energy)

    def addPath(self, path):
        child = path.child
        self.graph.add_edge(path.parent, child, path=path,
                            energy=path.energy,Emin=child.energy)
    def addRun(self, run):
        self.graph.add_edge(run.parent, run.child, run=run,
                               energy=run.parent.energy,
                               Emin=run.child.energy)

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

    def nsuccessors(self, replica):
        return sum(1 for _ in self.graph.successors_iter(replica))

    def minima(self, order=True):
        if order:
            return sorted((n for n in self.graph.nodes()
                           if type(n) is Minimum), key=lambda r: r.energy)
        else:
            return [n for n in self.graph.nodes() if type(n) is Minimum]

    def runs(self, Esplit=None):
        edges = self.graph.edge
        runs = []
        for edges2 in edges.itervalues():
            for attr in edges2.itervalues():
                if 'run' in attr:
                    run = attr['run']
                    if Esplit is None:
                        runs.append(attr['run'])
                    else:
                        Emax = run.Emax
                        if len(Emax) > 1:
                            if Emax[0] > Esplit and Emax[-1] <= Esplit:
                                runs.append(run)
        return runs

    def paths(self):
        edges = self.graph.edge
        paths = []
        for edges2 in edges.itervalues():
            for attr in edges2.itervalues():
                if 'path' in attr:
                    paths.append(attr['path'])
        return paths

    def findSamples(self, replica):
        return list(self.genSamples(replica))

    def genSamples(self, replica):
        runs = self.genConnectedRuns(replica)
        for run in runs:
            if run.parent == replica:
                if 0 in run.stored:
                    i = run.stored.searchsorted(0, side='left')
                    yield run.configs[i]
                elif len(run.Emax) == 1:
                    yield run.child.coords

    def pathtoRun(self, path):
        """
        """
        edges = [self.graph.edge[u][v] for u, v in izip(path[:-1], path[1:])]
        run = reduce(joinRuns,
                     (edge['run'] for edge in edges if edge.has_key('run')))
        return run

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

    def getConnectedRuns(self, replica, Efilter=None):
        """
        """
        return list(self.genConnectedRuns(replica, Efilter))

    def genConnectedRuns(self, replica, Efilter=None):
        """
        """
        for rep, edge in self.graph.edge[replica].iteritems():
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
            for _run in self.getConnectedRuns(rep, Efilter):
                yield _run

    def genPreceedingReplicas(self, replica):
        for r1 in self.graph.predecessors_iter(replica):
            for r2 in self.genPreceedingReplicas(r1):
                yield r2
            yield r1

    def genConnectedReplicas(self, replica):
        """
        """
        successors = [node for node in self.graph.successors(replica)
                        if isinstance(node, Replica)]
        for rep1 in successors:
            for rep2 in self.genConnectedReplicas(rep1):
                yield rep2
            yield rep1

    def genConnectedMinima(self, replica, f=1.):
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

    def getReplicaMinimaCDFs(self, replicas):
        replicasets = self.connectReplicas(replicas)
        cdfs = []
        for reps, mset in replicasets:
            cdf = reduce(lambda x,y: x+y, map(self.getMinimaCDF, reps))
            cdfs.append(cdf)
        return replicasets, cdfs

    def getMinimaCDF(self, replica):
        return CDF(*zip(*( (m.energy, f)
                            for m,f in self.genConnectedMinima(replica))
                        ), n=self.graph.out_degree(replica))

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


    def plot(self, energies=True, maxE=0., arrows=False, node_size=5, **kwargs):
        pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')
        if energies:
            pos = dict((r, (p[0], np.clip(r.energy, None, maxE)))
                        for r,p in pos.iteritems())
        nx.draw(self.graph, pos, arrows=arrows, node_size=node_size, **kwargs)

    def mergeMinima(self, min1, min2):
        raise NotImplementedError

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

    def __hash__(self):
        return hash(self.replicas)

class BasinGraph(object):
    """ This class joins replicas in the ReplicaClass together
    as a set of super basins.
    """
    def __init__(self, replicaGraph, target_sig=1e-2):
        """
        """
        self.repGraph = replicaGraph
        self.target_sig = 1e-2
        self.initialize()

    def initialize(self):
        """
        """
        self.graph = nx.DiGraph()
        self.repdict = {}

    def connectBasins(self, parent, basin):
        """
        """
        runs = list(self.genConnectingRuns(parent, basin))
        parentrep = min(parent.replicas, key=lambda r: r.energy)
        childrep = min(basin.replicas, key=lambda r: r.energy)
        if runs:
            run = combineAllRuns(runs, parentrep, childrep)
            self.graph.add_edge(parent, basin, parent=parent,
                                run=run, nruns=len(runs))
        else:
            if isinstance(childrep, Minimum):
                children = set(
                    chain(*chain(self.repGraph.graph.successors_iter(r)
                                 for r in parent)) )
                if children == basin.replicas:
                    self.graph.add_edge(parent, basin, parent=parent,
                                        minimum=childrep, nruns=len(parent))
            run = None
        return run

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

    def getParent(self, basin):
        parents = self.graph.predecessors(basin)
        parent, = parents if parents else (None,)
        return parent

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

    def SuperBasin(self, replicas, parent=None, **kwargs):
        """
        """

        if parent is None:
            parentreplicas = set(
                chain(*(self.repGraph.genPreceedingReplicas(r)
                        for r in replicas)) )
            parentbasins = set(self.repdict[r] for r in parentreplicas
                               if self.repdict.has_key(r))
            parent = [basin for basin in parentbasins
                      if not any(nx.has_path(self.graph, basin, b)
                                 for b in parentbasins if not b is basin)]
        else:
            parent = [parent]

        basin = SuperBasin(replicas)
        self.repdict.update((rep, basin) for rep in replicas)
        self.graph.add_node(basin, energy=basin.energy, **kwargs)

        if len(parent):
            assert len(parent) == 1
            parent = parent.pop()
            self.connectBasins(parent, basin)

        return basin

    def updateMinimum(self, m):
        """
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

    def updateMinima(self):
        """
        """
        minima = self.repGraph.minima()
        for m in minima:
            self.updateMinimum(m)

    def findEnergySplit(self, basin):
        Esplits = np.r_[self.repGraph.Esplits, np.inf]
        return Esplits.searchsorted(basin.energy, side='right')

    def getSiblings(self, parent, basin):
        """
        """
        Esplits = np.r_[-np.inf, self.repGraph.Esplits, np.inf]

        i = self.findEnergySplit(basin)
        siblings = filter(lambda b: self.findEnergySplit(basin) == i,
                          self.genAllSuccessors(parent))

        return siblings


    def mergeBranches(self, basin):
        """
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
        print len(mergedbasins), 'len mergedbasins'
        for mbasin, pbasin in izip(mergedbasins[:-1], mergedbasins[1:]):
            parents = self.graph.predecessors(mbasin)
            print len(parents), 'len parents'
            if len(parents) > 0:
                for p in parents:
                    if p != pbasin:
                        print 'remove edge', p.energy, mbasin.energy
                        self.graph.remove_edge(p, mbasin)
                    elif p is not pbasin:
                        print 'relabel', p.energy, pbasin.energy
                        print p
                        print pbasin
                        nx.relabel_nodes(self.graph, {parents[0]: pbasin},
                                         copy=False)
            elif not self.graph.has_edge(pbasin, mbasin):
                print 'connect basins', pbasin.energy, mbasin.energy
                self.connectBasins(pbasin, mbasin)

    def mergeBasins(self, basins):
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

    def addtoBasin(self, basin, replicas):
        """
        """
        newbasin = basin + SuperBasin(replicas)
        if basin != newbasin:
            mapping = {basin: newbasin}
            self.repdict.update((rep, newbasin) for rep in newbasin.replicas)
            nx.relabel_nodes(self.graph, mapping)
        return newbasin

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

    def genConnectedMinima(self, basin):
        """
        """
        replicas = []
        for rep in basin.replicas:
            isparent = any(nx.has_path(self.repGraph.graph, rep, _rep)
                           for _rep in basin.replicas if rep!=_rep)
            if not isparent:
                replicas.append(rep)
        f = 1./len(replicas)
        for rep in replicas:
            for minf in self.repGraph.genConnectedMinima(rep, f):
                yield minf

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

    def getConnectedMinima(self, basin, f=1.):
        """
        """
        if type(basin) is Minimum:
            return [[basin], [f]]
        else:
            successors = self.graph.successors(basin)
            if successors:
                newf = f / len(successors)
                return reduce(
                    lambda x,y: [x[0]+y[0],x[1]+y[1]],
                    (self.getConnectedMinima(s, newf) for s in successors))
            else:
                return []

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

    def genAllSuccessors(self, basin):
        for child in self.graph.successors_iter(basin):
            for child2 in self.genAllSuccessors(child):
                yield child2
            yield child

    def genPaths(self, basin, target):
        """
        """
        paths = []
        for rep in basin.replicas:
            paths.append(nx.all_simple_paths(self.repGraph.graph, rep, target))
        return chain(*paths)

    def getBasinBranchReplicas(self, basin):
        """Find Replicas connected to a basin within the closest energy cutoff
        """
        try:
            # Finding the highest energy replica connected to the basin
            Esplits = self.repGraph.Esplits
            replica = max(self.genConnectedReplicas(basin),
                          key=lambda rep: rep.energy)
            iRep = Esplits.searchsorted(replica.energy)

            # Finding Esplit
            if iRep > 0:
                Esplit = Esplits[iRep-1]
            else:
                Esplit = None

            replicas = list(self.genConnectedReplicas(basin, Esplit=Esplit))
        except ValueError:
            # No replicas found
            Esplit = None
            replicas = []

        return replicas, Esplit

    def getMinimaCDF(self, basin):
        """Returns the CDF of the minima connected to basin

        Parameters
        ----------
        basin : SuperBasin

        Returns
        -------
        cdf : CDF
            CDF of minima
        """
        repCDFs = (self.repGraph.getMinimaCDF(r) for r in basin.replicas)
        return reduce(lambda x,y:x+y, repCDFs)

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
        X, X2 = exp(logX), exp(logX2)

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

    def plot(self, energies=True, maxE=0., **kwargs):
        """
        """
        pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')
        if energies:
            pos = dict((r, (p[0], np.clip(r.energy, None, maxE)))
                        for r,p in pos.iteritems())
        nx.draw(self.graph, pos, **kwargs)

class NumericIntegrator(object):
    """Class for performing numerical integrals of a function
    when the configuration volume is known.
    """

    def __init__(self, voltoE, a, b, args=(), quad_kw={}):
        self.voltoE = voltoE
        self.args = args
        self.a = a
        self.b = b
        self.quad_kw = quad_kw

    def calcIntegral(self, func, **kwargs):
        """
        """
        kw = dict_update_copy(kwargs, self.quad_kw)

        try:
            intfunc = lambda E: func(self.voltoE(E, *self.args))
            nint, nstd = quad(intfunc, self.a, self.b, **kw)
        except:
            # If func returns more than one value
            # Not particularly efficient, but hopefully not bottleneck...
            intfunc = lambda E, i: func(self.voltoE(E, *self.args))[i]
            nint, nstd = [], []
            i = 0
            while True:
                try:
                    res = quad(intfunc, self.a, self.b, args=(i,), **kw)
                    nint.append(res[0])
                    nstd.append(res[1])
                    i += 1
                except IndexError:
                    break
        return nint, nstd

    __call__ = calcIntegral

    @staticmethod
    def harmonicVoltoE(vol, c, ndof):
        """
        """
        return vol**(1./ndof) / c

    @staticmethod
    def harmonicEtoVol(E, c, ndof):
        """
        """
        return c * E**(ndof)

    @classmethod
    def HarmonicIntegrator(cls, a, b, c, ndof, quad_kw={}):
        """
        """
        args = (c, ndof)
        return cls(cls.harmonicVoltoE, a, b, args=args, quad_kw=quad_kw)


class FunctionGraph(object):
    """ Class which calculates and stores the result of calculating the
    integral of a function over a basinGraph.


    """

    def __init__(self, basinGraph, func):

        self.basinGraph = basinGraph
        self.func = func

        self.basinGraph.calcBasins()
        self.graph = self.copyGraph(self.basinGraph.graph)

        self.calculated = False

    def copyGraph(self, graph):
        """
        Performs shallow copy of graph
        """
        newgraph = nx.DiGraph()

        for node, attr in graph.node.iteritems():
            newgraph.add_node(node, **attr)

        for node in graph.nodes_iter():
            for child, edge in graph.edge[node].iteritems():
                newgraph.add_edge(node, child, **edge)

        return newgraph

    def calcConstrainedIntegral(self, edge, func, std=True):
        """
        """
        constraints = edge['constraints']
        Emax = constraints['Emax']
        dPhi = constraints['dPhi']
        dPhi2 = constraints['dPhi2']
        d2Phi2 = constraints['d2Phi2']

        fj = np.atleast_2d(self.func(Emax))
        f = fj.dot(dPhi)

        if constraints.has_key('NumericalIntegral'):
            analytic = True
            af, afstd = constraints['NumericalIntegral'].calcIntegral(func)
        else:
            analytic = False

        if std:
            f2 = np.einsum("ij,j,ij->i", fj, d2Phi2, (fj*dPhi2).cumsum(1))
            if analytic:
                f2 = f2 + 2*f*af + afstd**2 + af**2
                f += af
            return f, f2
        else:
            return f

    def calcEdgeIntegral(self, edge, func, std=True):
        """
        """
        if edge.has_key('run'):
            run = edge['run']
            Emax = run.Emax
        else:
            Emax = np.array([])

        dF = edge['dF']
        XdF = edge['XdF']
        X2dF = edge['X2dF']
        d2F2 = edge['d2F2']
        dF2 = edge['dF2']
        Xd2F2 = edge['Xd2F2']
        XdF2 = edge['XdF2']
        X2d2F2 = edge['X2d2F2']
        X2dF2 = edge['X2dF2']

        fj = np.atleast_2d(self.func(Emax))

        f = fj.dot(dF)
        Xf = fj.dot(XdF)

        if std:
            X2f = fj.dot(X2dF)
            f2 = np.einsum("ij,j,ij->i", fj, d2F2, (fj*dF2).cumsum(1))
            Xf2 = np.einsum("ij,j,ij->i", fj, Xd2F2, (fj*XdF2).cumsum(1))
            X2f2 = np.einsum("ij,j,ij->i", fj, X2d2F2, (fj*X2dF2).cumsum(1))
            return f, Xf, X2f, f2, Xf2, X2f2
        else:
            return f, Xf

    def calcNodeIntegral(self, parent, basin, std):

        if parent is not None:
            parentnode = self.graph.node[parent]
            edge = self.graph.edge[parent][basin]
            plogX = parentnode['logX']
            pbranch = parentnode['branchPi'][basin]
            pbranch2 = parentnode['branchPiPj'][basin][basin]
            if std:
                plogX2 = parentnode['logX2']
                pX2X2 = exp(plogX2 - 2*plogX) # To avoid underflow
                f, Xf, X2f, f2, Xf2, X2f2 = \
                    self.calcEdgeIntegral(edge, self.func, std)
            else:
                f, Xf = self.calcEdgeIntegral(edge, self.func, std)
            dvarf = edge['dvarf']
            dvarX = edge['dvarX']
        else:
            # If parent does not exist then set appropriate values
            plogX, plogX2 = 0., 0.
            pX2X2 = 1.
            pbranch = 1.
            pbranch2 = 1.
            f, Xf, X2f, f2, Xf2, X2f2 = np.zeros((6,1))
            dvarf, dvarX = 0., 0.

        node = self.graph.node[basin]
        phi = node['Phi']
        phi2 = node['Phi2'] if node.has_key('Phi2') else phi**2
        logX = node['logX']
        branchPi = node['branchPi']
        if std:
            logX2 = node['logX2']
            X2X2 = exp(logX2 - 2*logX) # To avoid underflow
            branchPiPj = node['branchPiPj']

        children = self.graph.edge[basin].keys()

        # Obtaining values of child branch integrals, calculates them
        # when necessary
        childrenints = {}
        for child in children:
            if std:
                if self.graph.node[child].has_key('f2'):
                    fi = self.graph.node[child]['f']
                    fi2 = self.graph.node[child]['f2']
                    dvarfi = self.graph.node[child]['dvarf']
                else:
                    fi, fi2, fistd, dvarfi = self.calcNodeIntegral(basin, child, std)
                    self.graph.node[child]['f'] = fi
                    self.graph.node[child]['f2'] = fi2
                    self.graph.node[child]['fstd'] = fistd
                    self.graph.node[child]['dvarf'] = dvarfi
                childrenints[child] = (fi, fi2, dvarfi)
            else:
                if self.graph.node[child].has_key('f'):
                    fi = self.graph.node[child]['f']
                else:
                    fi = self.calcNodeIntegral(basin, child, std)
                childrenints[child] = fi

        # Calculating 1st moment of edge
        funcint = np.exp(plogX + np.log(f - Xf)) * phi * pbranch

        if std:
            # Adding first moments of child branches
            childint = sum(fi for fi, f2i, dvarfi in childrenints.itervalues())

            # Calculating 2nd moment of edge
            funcint2 = np.exp(plogX2 + np.log(f2 - 2*Xf2 + X2f2)) * phi2 * pbranch2

            dvarfuncint = np.zeros_like(f)
            funcchild = np.zeros_like(f)
            childint2 = np.zeros_like(f)

            # Adding 2nd moments of child branches
            for i, childi in enumerate(children):
                fi, f2i, dvarfi = childrenints[childi]

                pi = branchPi[childi]

                # Variance reduction due to sampling child branch
                dvarfuncint = dvarfuncint + pi * dvarfi
                # Moment of edge x childi
                funcchild = funcchild + 2 * np.exp(np.log((Xf - X2f) * fi)+# / pi) +
                                                   plogX2 - logX) * phi2 / phi

                for j, childj in enumerate(children[:i+1]):
                    if i==j:
                        # Moment of childi**2
                        childint2 = childint2 + f2i
                    else:
                        # Moment of childi x childj
                        fj, f2j, dvarfj = childrenints[childj]
                        pj = branchPi[childj]
                        pipj = branchPiPj[childi][childj]
                        childint2 = ( childint2 +
                                      2*(X2X2 * pipj / pi / pj * fi * fj) *
                                      phi2 / phi**2 )

            childvar =  (childint2 - childint**2)
            # Variance reduction from calculating
            dvarfuncint = dvarfuncint + dvarX * childvar

            # Ensuring values are finite and sensible
            notfinite = np.isfinite(dvarfuncint) == False
            dvarfuncint[notfinite] = - childvar[notfinite]**2
            # The variance reduction if you sample from this basin.
            node['dsvarf'] = dvarfuncint

            # The variance reduction due to sampling the edge above the basin
            dvarfuncint = ( dvarfuncint -
                            sqrt(dvarf*dvarX) *
                            (funcchild - 2*funcint*childint) )
            dvarfuncint += dvarf * (funcint2-funcint**2)

            # Aggregating values
            fint = funcint + childint
            fint2 = funcint2 + childint2 + funcchild
            fintstd = np.sqrt(fint2 - fint**2)

            # Ensuring values are finite and sensible
            notfinite = np.isfinite(dvarfuncint) == False
            dvarfuncint[notfinite] = - fintstd[notfinite]**2

            node['f'] = fint
            node['f2'] = fint2
            node['fstd'] = fintstd
            node['dvarf'] = dvarfuncint

            return fint, fint2, fintstd, dvarfuncint
        else:
            # Adding first moments of child branches
            funcint += sum(childrenints.itervalues())
            return funcint

    def calcIntegral(self, std=True):

        parent = None
        basin = max(self.basinGraph.basins(), key=lambda b: b.energy)

        res =  self.calcNodeIntegral(parent, basin, std)
        self.calculated = True

        if std:
            self.f = res[0]
            self.f2 = res[1]
            self.fstd = res[2]
        else:
            self.f = res
        return res

    __call__ = calcIntegral

    @property
    def integral(self):
        if not self.calculated:
            self.calcIntegral(std=False)

        return self.f

    @property
    def error(self):
        if not hasattr(self, 'fstd'):
            self.calcIntegral(std=True)
        return self.fstd

    def calcNodeVarianceLoss(self, parent, basin):

        if parent is not None:
            parentnode = self.graph.node[parent]
            edge = self.graph.edge[parent][basin]
            plogX = parentnode['logX']
            pbranch = parentnode['branchPi'][basin]
            pbranch2 = parentnode['branchPiPj'][basin][basin]
            plogX2 = parentnode['logX2']
            #pX2X2 = exp(plogX2 - 2*plogX) # To avoid underflow
            f, Xf, X2f, f2, Xf2, X2f2 = \
                self.calcEdgeIntegral(edge, self.func, True)
            dvarf = edge['dvarf']
            dvarX = edge['dvarX']
        else:
            # If parent does not exist then set appropriate values
            plogX, plogX2 = 0., 0.
            #pX2X2 = 1.
            pbranch = 1.
            pbranch2 = 1.
            f, Xf, X2f, f2, Xf2, X2f2 = 0., 0., 0., 0., 0., 0.
            dvarf = 0.
            dvarX = 0.

        node = self.graph.node[basin]
        phi = node['Phi']
        phi2 = node['Phi2'] if node.has_key('Phi2') else phi**2
        logX = node['logX']
        branchPi = node['branchPi']
        logX2 = node['logX2']
        X2X2 = exp(logX2 - 2*logX) # To avoid underflow
        branchPiPj = node['branchPiPj']

        if node.has_key('f2'):
            f = node['f']
            f2 = node['f2']
            fstd = node['fstd']
        else:
            f, f2, fstd = self.calcNodeIntegral(parent, basin, True)

        varf = fstd**2

        children = self.graph.edge[basin].keys()
        # Obtaining values of child branch integrals, calculates them
        # when necessary
        childrenints = {}
        for child in children:
            if self.graph.node[child].has_key('f2'):
                fi = self.graph.node[child]['f']
                fi2 = self.graph.node[child]['f2']
            else:
                fi, fi2, fistd = self.calcNodeIntegral(basin, child, True)
                self.graph.node[child]['f'] = fi
                self.graph.node[child]['f2'] = fi2
                self.graph.node[child]['fstd'] = fistd
            childrenints[child] = (fi, fi2)

        return 0
