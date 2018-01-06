# -*- coding: utf-8 -*-

from itertools import chain, izip, groupby
from math import exp, log, sqrt
import numpy as np
from scipy.integrate import quad
import networkx as nx

from nestedbasinsampling.storage import Run
from nestedbasinsampling.nestedsampling import calcRunAverageValue
from nestedbasinsampling.utils import dict_update_copy

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

        self.graph = self.copyGraph(self.basinGraph.graph)
        self.calcBasins()

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
        if edge.has_key('weights'):
            weights = edge['weights']
            return calcRunAverageValue(weights, func, std)
        elif edge.has_key('run'):
            run = edge['run']
            return run.calcAverageValue(func, std=std)
        else:
            if std:
                return np.zeros(6,1)
            else:
                return np.zeros(2,1)

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

            if edge.has_key('weights'):
                weights = edge['weights']
                dvarf = weights['dvarf']
                dvarX = weights['dvarX']
            else:
                dvarf, dvarX = 0., 0.
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
                        if pi > 0 or pj > 0:
                            childint2 = (
                                childint2 + 2*(X2X2 * pipj / pi / pj * fi * fj)
                                * phi2 / phi**2 )

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
            funcint = funcint + sum(childrenints.itervalues())
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

    def calcBasins(self):
        """
        """
        # Aggregating and calculating the number and associated weights
        # of the runs associated with each edge
        for b1, edges in self.graph.adjacency_iter():
            for b2, edge in edges.iteritems():
                runs = self.basinGraph.genConnectingRuns(b1, b2)
                nruns = sum(r.parent.energy >= b1.energy for r in runs)
                run = self.basinGraph.getConnectingRun(b1, b2)
                weights = run.calcWeights()

                edge['run'] = run
                edge['weights'] = weights
                edge['nruns'] = nruns


        basin0 = self.basinGraph.basins(order=True)[-1]
        tocalculate = [basin0]

        while tocalculate:
            basin = tocalculate.pop()
            self.calcBasinVolumeRatio(basin)
            self.calcBranchProbabilities(basin)
            successors = self.graph.successors(basin)
            tocalculate.extend(successors)

    def calcBranchProbabilities(self, basin):
        """
        """
        node = self.graph.node[basin]
        successors = self.graph.successors(basin)
        if successors:
            nruns = dict((b, attr['nruns'])  for b, attr in
                         self.graph.edge[basin].iteritems())
            totruns = float(sum(nruns.itervalues()))
            if totruns > 0:
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
                    #self.calcEdgeWeights(edge)
            else:
                branchP = dict((b, 0.) for b, nrun in
                                nruns.iteritems())
                branchPiPj = dict((bi,
                                   dict((bj,0.) for bj in nruns))
                                   for bi in nruns)
                node['branchPi'] = branchP
                node['branchPiPj'] = branchPiPj

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

            if branchP > 0:
                node['logX'] = logXp + log(branchP) + logXedge
                node['logX2'] = logX2p + log(branchP2) + logX2edge
            else:
                node['logX'] = -np.inf
                node['logX2'] = -np.inf
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
