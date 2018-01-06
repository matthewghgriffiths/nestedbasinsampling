# -*- coding: utf-8 -*-

from collections import defaultdict, namedtuple
from itertools import chain, izip, groupby
from functools import total_ordering
from math import exp, log, sqrt

import numpy as np
from scipy.special import polygamma, betaln, gammaln
from scipy.stats import chi2
from scipy.linalg import cho_factor, cho_solve

import networkx as nx
import matplotlib.pyplot as plt

from pele.utils.disconnectivity_graph import DisconnectivityGraph

from nestedbasinsampling.storage import (
    Minimum, Replica, Run, Database, TransitionState)
from nestedbasinsampling.sampling.stats import AndersonDarling, CDF
from nestedbasinsampling.nestedsampling.combine import combineAllRuns
from nestedbasinsampling.nestedsampling.integration import logsumexp, logtrapz
from nestedbasinsampling.utils import (
    len_iter, iter_minlength, SortedCollection, GraphError)

def calc_thermodynamics(Es, log_vol, Ts, Emin=None, log_f=0., ntrap=5000):
    """
    """
    if Emin is None:
        Emin = Es[-1]

    stride = max(Es.size/ntrap, 1)
    Es = Es[::-stride] - Emin
    log_vol = log_vol[::-stride]
    assert Es[0] >= 0

    ET = -Es[None,:]/Ts[:,None]
    logEs = np.log(Es)[None,:]

    logZ = logtrapz(ET, log_vol,axis=1) + log_f
    ET += logEs
    logE1 = logtrapz(ET, log_vol,axis=1) + log_f
    ET += logEs
    logE2s = logtrapz(ET, log_vol,axis=1) + log_f

    return logZ, logE1, logE2s, Emin

def calcCv(lZ, lE1, lE2, Emin=0.):
    U = np.exp(lE1 - lZ) + Emin
    U2 = np.exp(lE2 - lZ) + 2*Emin*U - Emin**2
    V = U - 0.5*k * Ts
    V2 = U2 - U**2 + V**2

    Cv = 0.5 * k + (V2 - V ** 2) * Ts ** -2

    Ret = namedtuple("CvReturn", "lZ U U2 Cv")
    return Ret(lZ=lZ, U=U, U2=U2, Cv=Cv)

def logmoment2Beta(logm1, logm2):
    """ Calculates the two parameter beta distribution parameters
    that best fits the logarithm of the first and second moments

    Parameters
    ----------
    logm1 : array_like
        the logarithm of the first moments
    logm2 : array_like
        the logarithm of the second moments

    Returns
    -------
    a : array_like
        the first beta distribution parameter
    b : array_like
        the second beta distribution parameter
    """
    logvar = logm2 + np.log(1 - np.exp(2*logm1 - logm2))
    logm3 = logm1 + np.log(1 - np.exp(logm2 - logm1))
    a = np.exp(logm1 + logm3 - logvar)
    b = np.exp(logm3 - logvar + np.log(1. - np.exp(logm1)))
    return a, b

def beta_log_bayes_factor(a, b, pa=0.5, pb=0.5):
    """ Calculates the log of the Bayes factor (BF)
    between all pairs of values of a and b

    Parameters
    ----------
    a : array_like
        the first shape parameters of beta distributions to compare
    b : array_like
        the second shape parameters of the beta distributions to compare
    pa : float, optional
        the prior pseudocount of a
    pb : float, optional
        the prior pseudocount of b

    Returns
    -------
    logBF : ndarray
        the logarithm of the BF between all of the as and bs
    comp : tuple, shape(2) of ndarrays, shape(n)
        the index array of the comparisons
    """
    a = np.array(a, copy=False, ndmin=2)
    b = np.array(b, copy=False, ndmin=2)

    comp = np.triu_indices(len(a), 1)
    logBF = (
        ( betaln(a[comp[0]]+a[comp[1]] - 2*pa, b[comp[0]]+b[comp[1]] - 2*pb)
        - betaln(2*pa,2*pb) )
        -
        ( betaln(a[comp[0]] - pa, b[comp[0]] - pb)
        + betaln(a[comp[1]] - pa, b[comp[1]] - pb)
        - 2*betaln(pa, pb) ))
    return logBF, comp

def find_run_harmonic_energy(r, m, k, res=1000, sig=1e-6, minlive=100,fac=2./3):
    """Finds the maximum energy of a nested sampling run that looks
    harmonic

    Parameters
    ----------
    r : Run
        the nested sampling run
    m : Minimum
        the minimum associated with the nested sampling run
    k :
        the number of degrees of freedom

    Returns
    -------
    Eharm : float
        the maximum harmonic energy
    i : int
        the index of the dead point which is at the
        maximum harmonic energy
    """
    Es = r.Emax[::-1]-m.energy
    ns = r.nlive[::-1].astype(float)

    l = (np.log(ns+1) - np.log(ns)).cumsum()
    varl = (polygamma(1,ns) - polygamma(1,ns+1))
    cvarl = varl[::-1].cumsum()[::-1]
    lH = 0.5*k*np.log(Es)

    chi_sig = []
    for j in xrange(1,l.size,l.size/res):
        chi_sig.append(
            chi2.cdf(
                ((l[:j] - lH[:j] - l[j] + lH[j])**2
                /(cvarl[:j] - cvarl[j])).sum(), j)
            )
    chimin = np.minimum.accumulate(chi_sig[::-1])[::-1]
    im = ns.searchsorted(min(minlive, r.nlive[0]))
    ix = 1 + int(chimin.searchsorted(sig)*l.size/res*fac)
    i = l.size - max(im, ix) - 1
    return r.Emax[i], i

@total_ordering
class SuperBasin(object):
    """ SuperBasin object.

    A super basin is defined by a single replica. All the states
    accessible via downhill paths from this replica are part of
    the superbasin. The objects of this class are hashable
    so that they can included in networkx graphs.

    Attributes
    ----------
    energy : float
        the energy of the basin
    log_vol : float
        the logarithm of the expected volume of the basin
    log_vol2 : float
        the logarithm of the expected volume squared
    """
    def __init__(self, energy, minima={}, log_vol=None, log_vol2=None):
        self.energy = energy
        self.log_vol = log_vol
        if log_vol2 is not None:
            self.log_vol2 = log_vol2
        elif self.log_vol is not None:
            self.log_vol2 = 2*self.log_vol
        else:
            self.log_vol2 = None
        self.minima = frozenset(minima)

    def __eq__(self, basin):
        return self.energy == basin.energy

    def __gt__(self, basin):
        return self.energy > basin.energy

    def __hash__(self):
        return hash((self.energy, self.minima))

    def __str__(self):
        return (
            "SuperBasin(energy={:10.5g},log_vol={:10.5g},log_vol2={:10.5g})".
            format(self.energy, self.log_vol, self.log_vol2))

class BasinGraph(object):
    """ This class joins replicas in the ReplicaClass together
    as a set of super basins.
    """
    def __init__(self, replicaGraph, dof=None, max_energy=np.inf, debug=True):
        """
        """
        self.repGraph = replicaGraph
        self.dof = dof
        self._connectgraph = nx.Graph()
        self._disconnectgraph = None
        self.debug=debug
        self.initialize(max_energy)

    def SuperBasin(self, energy, minima={}, log_vol=None, log_vol2=None):
        newbasin = SuperBasin(
            energy, minima=minima, log_vol=log_vol, log_vol2=log_vol2)
        self.graph.add_node(newbasin)
        self.basin[newbasin.minima] = newbasin
        return newbasin

    def initialize(self, max_energy=np.inf):
        """
        """
        self.graph = nx.DiGraph()
        self.basin = {}

        self.basins = self.graph.nodes

        minima = self.repGraph.minima()

        gbasin = self.SuperBasin(
            max_energy, minima=minima, log_vol=0., log_vol2=0.)

        minima = self.repGraph.minima()

        if self.debug:
            print ("BasinGraph > initialising with {:d} minima"
                   ).format(len(minima))

        for m in self.repGraph.minima():
            newbasin = self.SuperBasin(m.energy, [m])
            self.basin[m] = newbasin
            newrun = self.repGraph.nested_run(m).split(max_energy, None)
            self.graph.add_edge(gbasin, newbasin, run=newrun)

    @property
    def global_basin(self):
        return max(self.graph)

    def set_harmonic_basins(
        self, res=1000, sig=1e-6, minlive=500, nlowest_minima=2):
        """
        Calculates the harmonic correction to the minbasin volumes
        """
        gbasin = self.global_basin

        assert self.dof is not None

        # Calculate the log volumes top down
        for child in self.graph.successors(gbasin):
            self.calc_edge_moments(gbasin, child)

        # Calculate difference between top down volume and harmonic volume
        diffs = {}
        weights = {}
        log_vols = {}
        log_harms = {}
        log_counts = {}
        for parent, child in izip(self.harmbasins, self.minbasins):
            m, = child.minima
            lh = (0.5*self.dof*np.log(parent.energy - m.energy)
                  - 0.5*m.fvib - np.log(m.pgorder))

            (_,_, edge), = self.graph.in_edges_iter(parent, True)
            run = edge['run']
            ns = run.nlive
            weight = 1./(polygamma(1,ns) - polygamma(1,ns+1)).sum()

            log_count = np.log(self.count_runs(child))
            log_vol = edge['log_vol'][-1]

            parent.log_vol = log_vol - log_count

            diffs[parent] = parent.log_vol - lh
            log_vols[parent] = log_vol
            log_harms[parent] = lh
            log_counts[parent] = log_count
            weights[parent] = weight
            if self.debug:
                print (
                    "BasinGraph > harmonic interpolation, basinE="+
                    "{:10.5g}, log vol diff"+
                    "{:10.5g}").format(parent.energy, diffs[parent])

        # Weighted average of the shift
        self.harm_shift = np.average(diffs.values(), weights=weights.values())
        if self.debug:
            print (
                "BasinGraph > harmonic interpolation, "+
                "average log vol diff = {:10.5g}").format(self.harm_shift)

        # Apply shift to harmonic basins
        for parent, child in izip(self.harmbasins, self.minbasins):
            parent.log_vol -= diffs[parent] - self.harm_shift
            parent.log_vol2 = parent.log_vol * 2
            if self.debug:
                print (
                    "BasinGraph > harmonic interpolation, basinE="+
                    "{:10.5g}, logShift={:8.4g}").format(
                        parent.energy, diffs[parent] - self.harm_shift)

        # Calculate full volumes including harmonic wells
        for child in self.graph.successors(gbasin):
            self.calc_edge_moments(gbasin, child)

    def calc_energy_spacings(self, gbasin, log_step=5.):
        """
        """
        Es, log_vol = map(
            np.concatenate,izip(*(
                (edge['Es'], edge['log_vol'])
                for edge in self.graph.edge[gbasin].itervalues())))

        Es[::-1].sort()
        log_vol[::-1].sort()

        log_space = np.arange(0., log_vol[-1], -log_step)
        Ematch = Es[-log_vol[::-1].searchsorted(log_space)]

    def regroup_minima(self, res=1000, sig=1e-6, minlive=500, lowest=2):
        """
        """
        minima = self.repGraph.minima()
        edges = chain(*(
            self.graph.in_edges_iter(self.basin[m], data=True)
            for m in minima[lowest:]))

        tokeep = [
            edge for m in minima[:lowest]
            for edge in self.graph.in_edges_iter(self.basin[m], data=True)]
        tomerge = []
        for edge in edges:
            if edge[2].has_key('run'):
                (tomerge.append(edge) if edge[2]['run'].nlive[0] < minlive else
                 tokeep.append(edge))

        if self.debug:
            print ("BasinGraph > merging {:d} basins, "+
                   "keeping {:d} basins").format(len(tomerge), len(tokeep))

        self.mergedbasin = self.merge_basins([edge[1] for edge in tomerge])
        for edge in tomerge:
            self.graph.remove_node(edge[1])

        # Basins to keep
        self.minbasins = [edge[1] for edge in tokeep]
        # Finding harmonic energy of minbasins
        if self.dof is not None:
            self.harmbasins = [
                self.find_harmonic_basin(b, res=res, sig=sig, minlive=minlive)
                for b in self.minbasins]

        return self.mergedbasin, self.minbasins, self.harmbasins

    def parent(self, basin):
        try:
            parent, = self.graph.predecessors_iter(basin)
        except ValueError:
            parent = None
        return parent

    def merge_basins(self, basins, energy=None, log_vol=None, log_vol2=None):
        """
        """
        # There should only be one parent
        parent, = set(self.parent(b) for b in basins)

        minima = reduce(lambda x, y: x.union(y), (b.minima for b in basins))
        newbasin = self.SuperBasin(
            energy, minima=minima, log_vol=log_vol, log_vol2=log_vol2)

        runs = []
        for b in basins:
            edge = self.graph[parent][b]
            self.graph.remove_edge(parent, b)
            self.graph.add_edge(newbasin, b, **edge)
            runs.append(edge['run'])

        newrun =  combineAllRuns(runs).split(parent.energy, newbasin.energy)
        self.graph.add_edge(parent, newbasin, run=newrun)

        return newbasin

    def find_harmonic_basin(self, basin, res=1000, sig=1e-6, minlive=100):
        """
        """
        # there should be only one minima in this basin
        m, = basin.minima
        assert len(self.graph.successors(basin)) == 0
        parent = self.parent(basin)
        r = self.graph[parent][basin]['run']

        Eharm = find_run_harmonic_energy(
            r, m, self.dof, res=res, sig=sig, minlive=minlive)[0]

        newbasin = self.SuperBasin(Eharm, minima=[m])
        newrun = r.split(r.parent, Eharm)

        self.graph.remove_edge(parent, basin)
        self.graph.add_edge(parent, newbasin, run=newrun)
        self.graph.add_edge(newbasin, basin,
                            minimum=m, fvib=m.fvib, pgorder=m.pgorder)

        if self.debug:
            print (
                "BasinGraph > finding harmonic energy range, minimumE =" +
                "{:10.5g}, harmonicE ={:10.5g}").format(m.energy, Eharm)

        return newbasin

    def calc_edge_moments(self, parent, child):
        """
        """
        edge = self.graph[parent][child]

        l0 = parent.log_vol ## probably need to factor whether l0 is None
        l02 = parent.log_vol2
        lf = child.log_vol
        lf2 = child.log_vol2

        run = edge['run'].split(parent.energy, child.energy)

        ld = run.log_frac
        ld2 = run.log_frac2

        ld += l0
        ld2 += l02
        if lf is None:
            l = ld
            l2 = ld2
        else:
            dshift = np.linspace(0, lf - ld[-1], ld.size+1)[1:]
            ld += dshift
            ld2 += 2*dshift

            lu = run.log_frac_up
            lu2 = run.log_frac2_up
            lu += lf
            lu2 += lf2
            ushift = np.linspace(l0-lu[0], 0, lu.size+1)[1:]
            lu += ushift
            lu2 += 2*ushift

            varld = ld2 + np.log1p(-np.exp(2*ld - ld2))
            varlu = lu2 + np.log1p(-np.exp(2*lu - lu2))
            var = - np.logaddexp(-varlu,-varld)

            l = np.logaddexp(ld-varld,lu-varlu) + var
            l2 = 2*l + np.log1p(np.exp(var - 2*l))

        edge['log_vol'] = l
        edge['log_vol2'] = l2
        edge['Es'] = run.Emax

        return edge

    def interpolate_connected_runs(self, basin, log_step=5.):
        """
        """
        connected_runs = (
            (b, edge['run'].split(basin.energy, None))
            for b, edge in self.graph.edge[basin].iteritems())
        children, basin_Es, basin_vol, basin_vol2 = izip(*(
            (b, run.Emax,
             run.log_frac, run.log_frac2)
            for b, run in connected_runs))

        # To get a smooth interpolation
        # Probably can do this better...
        Es = np.concatenate(basin_Es)
        log_vol = np.concatenate(basin_vol)
        Es[::-1].sort()
        log_vol[::-1].sort()
        log_space = np.arange(0., log_vol[-1], -log_step)
        Ematch = Es[-log_vol[::-1].searchsorted(log_space)]

        log_vols = np.array(
            [np.interp(Ematch, _Es[::-1], _l[::-1], left=-np.inf)
            + (basin.log_vol or 0.)
             for _Es, _l in izip(basin_Es, basin_vol)])
        log_vol2s = np.array(
            [np.interp(Ematch, _Es[::-1], _l[::-1], left=-np.inf)
            + (basin.log_vol2 or 0.)
             for _Es, _l in izip(basin_Es, basin_vol2)])

        log_ratio = np.diff(np.c_[[0.]*len(log_vols), log_vols], axis=1)
        log_ratio2 = np.diff(np.c_[[0.]*len(log_vols), log_vol2s], axis=1)

        a_vol, b_vol = logmoment2Beta(log_vols , log_vol2s)
        a_rat, b_rat = logmoment2Beta(log_ratio , log_ratio2)

        return Ematch, a_vol, b_vol, a_rat, b_rat, children

    def count_runs(self, basin):
        """"""
        parent = self.parent(basin)
        if parent is None:
            parent = basin
        nested_runs = (
            r for m in basin.minima for r in self.repGraph.nested_runs(m)
            if r.parent.energy >= parent.energy)
        return len_iter(nested_runs)

    def find_top_basin(self, log_step=5., pa=0.5, pb=0.5):
        """"""
        Ematch, a_vol, b_vol, a_rat, b_rat, basins = (
            self.interpolate_connected_runs(
                self.global_basin, log_step=log_step))
                
        logBF_vol, comp = beta_log_bayes_factor(a_vol, b_vol, pa=pa, pb=pb)
        logBF_rat, comp = beta_log_bayes_factor(a_rat, b_rat, pa=pa, pb=pb)
        logBF_vol[np.isnan(logBF_vol)] = -np.inf
        logBF_rat[np.isnan(logBF_rat)] = -np.inf
        logBF_rm = logBF_rat.min(0)
        logBF_rc = np.minimum.accumulate(logBF_rm)
        targetBF = -np.log(comp[0].size-1)
        i = -logBF_rc[::-1].searchsorted(targetBF)
        top_basin = self.merge_basins(basins, Ematch[i-1])
        
        if self.debug:
            print ("BasinGraph > Merging all basins at energy ={:10.5g}"
                   ).format(top_basin.energy)
        
        return top_basin

    def merge_runs(self, basin, log_step=1., target_BF=0., pa=0.5, pb=0.5):
        """
        """
        degree = -1
        while self.graph.out_degree(basin) != degree:
            degree = self.graph.out_degree(basin)

            Ematch, a_vol, b_vol, a_rat, b_rat, basins = (
                self.interpolate_connected_runs(
                    basin, log_step=log_step))

            logBF_vol, comp = beta_log_bayes_factor(a_vol, b_vol, pa=pa, pb=pb)
            logBF_rat, comp = beta_log_bayes_factor(a_rat, b_rat, pa=pa, pb=pb)

            logBFc = np.minimum.accumulate(logBF_rat,1)
            maxBF = np.nanmax(logBFc, axis=0)
            i_merge = -maxBF[::-1].searchsorted(target_BF) - 1
            
            if i_merge==-1:
                break
            print i_merge, -_merge==-1
            
            Emerge = Ematch[i_merge]
            good = logBFc[:,i_merge]>target_BF
            goodBF = logBFc[good,i_merge]
            goodpairs = np.c_[comp[0][good], comp[1][good]]
            BFpairs = sorted(izip(goodBF, goodpairs))
            pairset = set(tuple(pair) for pair in goodpairs)
            groups = defaultdict(set)
            while BFpairs:
                bf, pair = BFpairs.pop()
                group = groups[pair[0]].union(groups[pair[1]])
                group.update(pair)
                if all((i,j) in pairset for i in group for j in group if i < j):
                    for i in group:
                        groups[i] = group

            tomerge = []
            for group in groups.itervalues():
                if group and group not in tomerge:
                    tomerge.append(group)

            for group in tomerge:
                merge_a = sum(a_vol[j,i_merge] for j in group)
                merge_b = sum(b_vol[j,i_merge] for j in group)
                log_vol = -np.log1p(merge_b/merge_a)
                merge = [basins[j] for j in group]

                newbasin = self.merge_basins(merge, energy=Emerge)

                if self.debug:
                    print (
                        "BasinGraph > merging {:d} runs at energy {:10.5g}, "+
                        "log vol {:10.5}").format(len(group), Emerge, log_vol)

            if self.debug:
                print (
                    "BasinGraph > after merge primary basin "+
                    "has {:d} children").format(self.graph.out_degree(basin))
        
    def calc_log_volume_down(self, basin, force_calc=True):
        """"""

        c_node = self.graph.node[basin]

        try:
            (parent, child, edge), = self.graph.in_edges_iter(basin, data=True)
            assert basin is child
        except ValueError:
            # no parent
            return c_node, None

        p_node = self.graph.node[parent]
        # Calculate if parent top down volume not calculated
        if force_calc or not p_node.has_key('log_vol_d'):
            p_node, _ = self.calc_log_volume_down(parent)
        if not p_node.has_key('nruns'):
            p_node['nruns'] = self.count_runs(child)

        # Count number of runs
        c_nruns = self.count_runs(child)
        c_node['nruns'] = c_nruns

        # Calculate the fraction of runs that go down the child branch
        log_f = np.log(c_nruns) - np.log(p_node['nruns'])
        log_f2 = log_f + np.log(c_nruns + 1) - np.log(p_node['nruns']+1)
        c_node['frac'] = log_f
        c_node['frac2'] = log_f2

        # Calculate the fraction volumes by nested sampling
        if edge.has_key('run'):
            run = edge['run'].split(parent.energy, child.energy)

            # the log fractional volume of the edge
            edge['log_frac_d'] = run.log_frac
            edge['log_frac2_d'] = run.log_frac2

            # the log volume of the basin
            c_node['log_vol_d'] = (
                edge['log_frac_d'][-1] + p_node['log_vol_d'] + log_f)
            c_node['log_vol2_d'] = (
                edge['log_frac2_d'][-1] + p_node['log_vol2_d'] + log_f2)

        return c_node, edge

    def calc_log_volume_up(self, basin, force_calc=False):
        """"""
        node = self.graph.node[basin]

        try:
            children, edges = izip(*self.graph.edge[basin].iteritems())
        except ValueError:
            return node

        log_vols = dict()
        log_vol2s = dict()
        counts = dict()
        for i, (childi, edgei) in enumerate(izip(children, edges)):
            node_i = self.graph.node[childi]
            counts[childi] = self.count_runs(childi)

            if edgei.has_key('run'):
                # Calculating volume using bottom up nested sampling
                run = edgei['run'].split(basin.energy, childi.energy)

                # Calculate bottom up volumes if not already calculated
                if force_calc or not node_i.has_key('log_vol_u'):
                    node_i = self.calc_log_volume_up(childi, 
                                                     force_calc=force_calc)

                if node_i.has_key('log_vol_u'):
                    log_vol_i = node_i['log_vol_u']
                    log_vol2_i = node_i['log_vol2_u']

                    log_F = run.log_frac_up
                    log_F2 = run.log_frac2_up

                    edgei['log_frac_u'] = log_F
                    edgei['log_frac2_u'] = log_F2

                    log_vols[childi] = edgei['log_frac_u'][0] + log_vol_i
                    log_vol2s[childi] = edgei['log_frac2_u'][0] + log_vol2_i
                else:
                    log_vols[childi] = None
                    log_vol2s[childi] = None
            else:
                # Calculating volume using harmonic approximation
                m, = childi.minima
                log_vols[childi] = (
                    0.5 * self.dof *np.log(basin.energy - m.energy)
                    - 0.5*m.fvib - np.log(m.pgorder))
                log_vol2s[childi] = 2*log_vols[childi]

        tot_count = sum(counts.itervalues())

        # Filter out children that don't have bottom up volumes
        childrenb, edgesb = zip(*(
            (child, edge) for child, edge in izip(children, edges)
            if log_vols[child] is None)) or ([],[])
        # Filter out children that have bottom up volumes
        childrenu, edgesu = zip(*(
            (child, edge) for child, edge in izip(children, edges)
            if log_vols[child] is not None)) or ([],[])

        # Calculate Moments
        log_X = np.array([log_vols[childi] for childi in childrenu])
        log_X2 = np.array([log_vol2s[childi] for childi in childrenu])
        # Adding variances together
        log_var = logsumexp(log_X2 - 2*log_X)

        log_vol_u = logsumexp(log_X)
        log_vol2_u = log_var + 2*log_vol_u

        tot_count = sum(counts.itervalues())
        count_u = sum(counts[childi] for childi in childrenu)

        #Including the volume of the branch with out bottom up volumes
        log_tot = np.log(tot_count-1)
        log_tot2 = np.log(tot_count-2)
        log_cu = np.log(count_u-1)
        log_c2u = np.log(count_u-2)
        log_p = log_tot - log_cu
        log_p2 = log_p + log_tot2 - log_c2u

        node['log_vol_u'] = log_vol_u + log_p
        node['log_vol2_u'] = log_vol2_u + log_p2

        return node

    def calc_log_volume(self, basin, force_calc=True, ntrap=2000):
        """
        """
        p_node = self.graph.node[basin]

        log_vols = {}
        log_vol2s = {}
        for child, edge in self.graph[basin].iteritems():
            c_node = self.graph.node[child]
            if edge.has_key('log_frac_d'):
                Es = edge['run'].split(basin.energy, child.energy).Emax
                log_vol_d = (
                    edge['log_frac_d'] + p_node['log_vol_d'] + c_node['frac'])
                log_vol2_d = (
                    edge['log_frac2_d'] + p_node['log_vol2_d'] + c_node['frac2'])

                if edge.has_key('log_frac_u'):
                    if force_calc or not c_node.has_key('log_vol_up'):
                        self.calc_log_volume(
                            child, force_calc=force_calc, ntrap=ntrap)
                    # Getting bottom up volumes
                    log_vol_u = (
                        edge['log_frac_u'] + c_node['log_vol_up'])
                    log_vol2_u = (
                        edge['log_frac2_u'] + c_node['log_vol2_up'])

                    # Combining bottom up and top down volumes
                    log_rel_u = log_vol2_u - 2*log_vol_u
                    log_rel_d = log_vol2_d - 2*log_vol_d
                    log_rel = (log_rel_u**-1 + log_rel_d**-1)**-1
                    # Weighted sum
                    log_vol = ((log_vol_u/log_rel_u + log_vol_d/log_rel_d)
                               *log_rel)
                    log_vol2 = 2*log_vol + log_rel
                else:
                    log_vol = log_vol_d
                    log_vol2 = log_vol2_d

            elif edge.has_key('minimum'):
                # Calculate Harmonic volumes
                m = edge['minimum']
                Es = np.linspace(basin.energy - m.energy, 0., ntrap)
                log_vol = (0.5 * self.dof * np.log(Es)
                           -0.5*m.fvib - np.log(m.pgorder) + self.harm_shift)
                log_vol2 = 2*log_vol

            assert Es.size == log_vol.size
            edge['Es'] = Es
            edge['log_vol'] = log_vol
            edge['log_vol2'] = log_vol2
            log_vols[child] = log_vol[0]
            log_vol2s[child] = log_vol2[0]

        log_vars = dict((child, log_vol2s[child] - 2*log_vols[child])
                        for child in self.graph[basin])
        log_vol = logsumexp(log_vols.values())
        log_var = sum(log_vars.values())
        log_vol2 = log_var + log_vol*2
        p_node['log_vol'] = log_vol
        p_node['log_vol2'] = log_vol2

        # Calculating log_vol_u for smooth interpolation
        log_var_d = p_node['log_vol2_d'] - 2*p_node['log_vol_d']
        if log_var and log_var_d:
            log_var_u = p_node['log_vol2_u'] - 2*p_node['log_vol_u']
            log_vol_d = p_node['log_vol_d']
            log_var = (log_var_u**-1 + log_var_d**-1)**-1
            p_node['log_vol_up'] = log_var_u * (log_vol/log_var - log_vol_d/log_var_d)
            p_node['log_vol2_up'] = log_var_u + 2*p_node['log_vol_up']
        elif log_var_d:
            p_node['log_vol_up'] = log_vol
            p_node['log_vol2_up'] = log_vol2

        return p_node

    def calc_all_volumes(self, force_calc=True, ntrap=2000):
        """
        """
        gbasin = self.global_basin
        self.graph.node[gbasin]['log_vol_d'] = 0.
        self.graph.node[gbasin]['log_vol2_d'] = 0.

        self.calc_log_volume_up(self.global_basin, force_calc=force_calc)

        for basin in self.harmbasins + [self.mergedbasin]:
            self.calc_log_volume_down(basin, force_calc=force_calc)
                
        ##### Calculating average harmonic shift
        diffs = []
        weights = []
        basins = []
        for basin in self.basins():
            node = self.graph.node[basin]
            ld = node.get('log_vol_d',None)
            lu = node.get('log_vol_u',None)
            if ld and lu:
                l2d = node.get('log_vol2_d',None)
                l2u = node.get('log_vol2_u',None)
                lreld = l2d - 2 * ld
                lrelu = l2u - 2 * lu
                basins.append(basin)
                diffs.append(ld-lu)
                weights.append(1./(lreld + lrelu))
                if self.debug:
                    print (
                        "BasinGraph > Basin Energy ={:10.5g}, log vols :"+
                        "top down ={:10.5g}, bottom up ={:10.5g}").format(
                            basin.energy, ld, lu)

        self.harm_shift = np.average(diffs, weights=weights)
        
        if self.debug:
            print ("BasinGraph > Matching top down and bottom up volumes, "+
                   "harmonic shift ={:10.5g}").format(self.harm_shift)
        
        self.calc_log_volume(gbasin, force_calc=force_calc, ntrap=ntrap)

    def calc_thermodynamics(self, Ts, force_calc=False, ntrap=2000):
        """
        """
        self.calc_all_volumes(force_calc=force_calc, ntrap=ntrap)
        self.Emin = self.repGraph.database.get_lowest_energy_minimum().energy
        
        for parent, edges in self.graph.adjacency_iter():
            for child, edge in edges.iteritems():
                if edge.has_key('Es'):
                    log_therm = calc_thermodynamics(
                        edge['Es'], edge['log_vol'], Ts, 
                        Emin=self.Emin, ntrap=1000)
                    edge['log_therm'] = log_therm
                    
        lZs, lE1s, lE2s, _ = izip(*(edge['log_therm']
            for parent, edges in self.graph.adjacency_iter()
            for child, edge in edges.iteritems()
            if edge.has_key('log_therm')))
            
        lZ = logsumexp(lZs, axis=0)
        lE1 = logsumexp(lE1s, axis=0)
        lE2 = logsumexp(lE2s, axis=0)
        res = calcCv(lZ, lE1, lE2, self.Emin)
        
        return res
        
    def disconnectivity_graph(self, Emax=None, **kwargs):
        """"""
        if Emax is None:
            Emax = max(
                b.energy for b in self.basins() if np.isfinite(b.energy))

        minima = list(m for b in self.minbasins for m in b.minima)
        pairs = set((m1, m2) for i, m1 in enumerate(minima)
                            for m2 in minima[:i])
        g = nx.Graph()
        g.add_nodes_from(minima)

        # Finding lowest connected basins
        basins = self.basins()
        for basin in sorted(basins):
            if pairs:
                minlist = sorted(basin.minima)
                for i, m1 in enumerate(minlist):
                    for m2 in minlist[:i]:
                        if (m1, m2) in pairs:
                            ts = TransitionState(basin.energy, None, m1, m2)
                            g.add_edge(m1, m2, ts=ts, basin=basin)
                            pairs.remove((m1,m2))

        dg = DisconnectivityGraph(g, Emax=Emax)
        dg.calculate()
        return dg        

















