# -*- coding: utf-8 -*-

from itertools import chain, izip, groupby

from math import exp, log, sqrt

import numpy as np

from scipy.special import gammaln
from scipy.integrate import quad

import networkx as nx

import matplotlib.pyplot as plt

from nestedbasinsampling.storage import \
    Minimum, Replica, Run, Database
from nestedbasinsampling.random.stats import AndersonDarling, CDF

from nestedbasinsampling.nestedsampling import combineAllRuns
from nestedbasinsampling.utils import iter_minlength

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