# -*- coding: utf-8 -*-

from itertools import izip, groupby
from collections import defaultdict

import numpy as np
from scipy.special import gammaln, betaln, digamma

from nestedbasinsampling.nestedsampling.bayesian import (
    BayesianBetaMixture, _estimate_beta_parameters)
from nestedbasinsampling.utils import SortedCollection

def matchRuns(runs):
    """
    """
    nruns = len(runs)
    rEs = [r.Emax for r in runs]
    rnlive = [r.nlive for r in runs]
    Emin = max(Es.min() for Es in rEs)
    iend = [Es.size - Es[::-1].searchsorted(Emin) - 1 for Es in rEs]
    imin = min(iend)
    imax = max(iend)

    allEs = np.ones((len(runs), imax)) * -np.inf
    allns = np.ones((len(runs), imax))

    for i, (Es, ns, ie) in enumerate(izip(rEs, rnlive, iend)):
        allEs[i,:ie] = Es[:ie]
        allns[i,:ie] = ns[:ie]

    alllogXs = (np.log(allns) - np.log(allns+1)).cumsum(1)
    alllogX2s = (np.log(allns) - np.log(allns+2)).cumsum(1)

    Ematch = np.zeros(imin)
    icurr = np.zeros(nruns, int)
    i = 0
    ir = np.arange(nruns)

    try:
        while True:
            Ematch[i] = allEs[ir, icurr].min()
            icurr += 1

            gt = allEs[ir, icurr] > Ematch[i]
            while gt.any():
                icurr[gt] += 1
                gt = allEs[ir, icurr] > Ematch[i]
            i += 1
    except IndexError:
        pass

    Ematch = Ematch[:i]

    rlogXs = np.empty((nruns, Ematch.size))
    rlogX2s = np.empty((nruns, Ematch.size))

    for i, (logX, logX2) in enumerate(izip(alllogXs, alllogX2s)):
        rlogXs[i] = np.interp(Ematch, allEs[i,::-1], logX[::-1])
        rlogX2s[i] = np.interp(Ematch, allEs[i,::-1], logX2[::-1])

    dlogXs = np.diff(np.c_[[0.]*nruns, rlogXs])
    dlogX2s = np.diff(np.c_[[0.]*nruns, rlogX2s])

    rns, rnr = logmoment2Beta(rlogXs, rlogX2s)
    dns, dnr = logmoment2Beta(dlogXs, dlogX2s)

    return Ematch, rns, rnr, dns, dnr

def moment2Beta(m1, m2):
    """
    """
    var = m2 - m1**2
    m3 = m1 - m2
    ns = m1 * m3 / var
    nr = (1- m1) *m3 / var
    return ns, nr

def logmoment2Beta(logm1, logm2):
    """
    """
    logvar = logm2 + np.log(1 - np.exp(2*logm1 - logm2))
    logm3 = logm1 + np.log(1 - np.exp(logm2 - logm1))
    ns = np.exp(logm1 + logm3 - logvar)
    nr = np.exp(logm3 - logvar + np.log(1. - np.exp(logm1)))
    return ns, nr

def logBetaEvidence(a1, b1, a2, b2, pa=0.5, pb=0.5):
    logSame = (betaln(a1+a2+2*pa, b1+ b2 +2*pb)
               - betaln(2*pa,2*pb))
    logDiff = (betaln(a1 + pa, b1 + pb) +
               betaln(a2 + pa, b2 + pb) - 2*betaln(pa,pb) )
    return logSame, logDiff

def compareRuns(runs, prior=0.5, weight=1.):
    Ematch, newNs, newNr, _, _ = matchRuns(runs)
    comp = np.triu_indices(len(runs), 1)

    notfinite = np.isfinite(newNs) == False
    invalid = newNs < 1.
    lim = Ematch.size
    if invalid.any():
        lim = min(invalid.argmax(1)[invalid.any(1)].min(), lim)
    if notfinite.any():
        lim = min(notfinite.argmax(1)[notfinite.any(1)].min(), lim)

    logSame, logDiff = logBetaEvidence(
        newNs[comp[0],:lim], newNr[comp[0],:lim],
        newNs[comp[1],:lim], newNr[comp[1],:lim], pa=prior, pb=prior)

    BF = np.exp(logSame-logDiff)
    psame = 1. - 1./(1. + weight*BF)

    return Ematch[:lim], psame, comp

class Cluster(frozenset):

    def __new__(cls, iterable, loglike=-np.inf, parameters=None):
        obj = super(Cluster, cls).__new__(Cluster, iterable)
        obj.loglike = loglike
        obj.parameters = parameters
        return obj

    def __repr__(self):
        return "Cluster({:s},loglike={:8.4g})".format(
            list(self), self.loglike)

    __str__ = __repr__

frozenset

class ClusterRuns(object):

    def __init__(self, runs, replicas=None, Efinish=None, ntries=10,
                 n_components=None, init_params='random', beta_prior=1.,
                 **kwargs):

        self.init_params = init_params
        self.beta_prior = beta_prior
        self.n_components = len(runs) if n_components is None else n_components
        self.ntries = ntries
        self._dpbmm_kw = kwargs

        self._initialize()
        self.addRuns(runs, replicas, Efinish=Efinish)

    def _initialize(self):
        self.runs = []
        self.replicas = []
        self.allclusters = set()

        self.dpbmm = BayesianBetaMixture(
            n_components=self.n_components, beta_prior=self.beta_prior,
            init_params=self.init_params, **self._dpbmm_kw)

    @property
    def clusters(self):
        if not self.allclusters:
            self.fit()
        return max(self.allclusters, key=lambda c: c.loglike)

    def fit_random(self):
        """"""
        cluster = self._do_fit('random')
        return cluster

    def _add_cluster(self, cluster):
        """"""
        if cluster in self.allclusters:
            new = set([cluster])
            c2, = new.intersection(self.allclusters)
            if cluster.loglike > c2.loglike:
                self.allclusters.remove(c2)
                self.allclusters.add(cluster)
        else:
            self.allclusters.add(cluster)

    def do_fit(self, init_params=None):
        """"""
        if init_params is not None:
            save_params = self.dpbmm.init_params
            self.dpbmm.init_params = init_params

        self.dpbmm.fit(self.AB)
        clusters = self._classify(self.AB, self.replicas)
        cluster = Cluster(
            (frozenset(l) for l in clusters.itervalues()),
            loglike=self.dpbmm.lower_bound_,
            parameters=self.dpbmm._get_parameters())

        if init_params is not None:
            self.dpbmm.init_params = save_params

        self._add_cluster(cluster)
        return cluster

    def fit(self):
        """"""
        for i in xrange(self.ntries):
            self.do_fit('random')
        self.do_fit([0]*len(self.AB))

        return self.clusters

    def _classify(self, AB, replicas):
        """"""
        labels = self.dpbmm.predict(AB)
        clusters = defaultdict(list)
        [clusters[i].append(r) for i, r in izip(labels, replicas)]
        return clusters

    def addRuns(self, runs, replicas=None, Efinish=None):
        """"""
        self.runs += runs
        self.nruns = len(self.runs)
        self.replicas += (range(self.nruns - len(runs), self.nruns)
                         if replicas is None else replicas)
        assert len(self.replicas) == len(self.runs)

        (self.Ematch_, self.Ns_, self.Nr_,
             self.dNs_, self.dNr_) = matchRuns(self.runs)
        self.Efinish = self.Ematch_[-1] if Efinish is None else Efinish
        i = (- self.Ematch_[::-1].searchsorted(self.Efinish)) or None
        self.Ematch = self.Ematch_[:i]
        self.Ns, self.Nr = self.Ns_[:,:i], self.Nr_[:,:i]
        self.dNs, self.dNr = self.dNs_[:,:i], self.dNr_[:,:i]
        self.AB = np.concatenate(
            (self.Ns[...,None], self.Nr[...,None]), axis=2)
        self.dAB = np.concatenate(
            (self.dNs[...,None], self.dNr[...,None]), axis=2)

    def testRuns(self, runs, replicas=None):
        """"""
        nruns = len(runs)
        replicas = range(nruns) if replicas is None else replicas
        assert len(replicas) == len(runs)

        # Matching the runs to the runs used in the classification
        rEs = [r.Emax for r in runs]
        rnlive = [r.nlive for r in runs]
        iend = [Es.size - Es[::-1].searchsorted(self.Efinish,side='right')
                for Es in rEs]
        imax = max(iend)
        allEs = np.ones((len(runs), imax)) * -np.inf
        allns = np.zeros((len(runs), imax))
        for i, (Es, ns, ie) in enumerate(izip(rEs, rnlive, iend)):
            allEs[i,:ie] = Es[:ie]
            allns[i,:ie] = ns[:ie]
        alllogXs = (np.log(allns) - np.log(allns+1)).cumsum(1)
        alllogX2s = (np.log(allns) - np.log(allns+2)).cumsum(1)

        rlogXs = np.empty((nruns, self.Ematch.size))
        rlogX2s = np.empty((nruns, self.Ematch.size))
        for i, (logX, logX2) in enumerate(izip(alllogXs, alllogX2s)):
            rlogXs[i] = np.interp(self.Ematch, allEs[i,::-1], logX[::-1])
            rlogX2s[i] = np.interp(self.Ematch, allEs[i,::-1], logX2[::-1])
        Ns, Nr = logmoment2Beta(rlogXs, rlogX2s)

        # Classifying the new runs
        AB = np.concatenate((Ns[...,None],Nr[...,None]), axis=2)
        clusters = self._classify(AB, replicas)

        return clusters

    def plot(self, f=None, axes=None, colors=None, cm='viridis',
             alphafill=0.8, alphaline=0.3):
        """
        """
        import matplotlib.pyplot as plt

        clusters = self.clusters
        self.dpbmm._set_parameters(clusters.parameters)
        log_prob_norm, log_resp = self.dpbmm._e_step(self.AB)
        _, _, ratios = _estimate_beta_parameters(self.AB, np.exp(log_resp))
        clusters = self._classify(self.AB, range(len(self.AB)))

        if colors is None:
            colors = getattr(plt.cm, cm)(np.linspace(0,1,len(clusters)))
        elif len(colors) <= len(clusters):
            colors = getattr(plt.cm, cm)(np.linspace(0,1,len(clusters)))

        if axes is None:
            f, axes = plt.subplots(len(clusters) + 1, sharex=True)

        for i, j in enumerate(clusters):
            logX = (np.log(ratios[j,:,0])
                    - np.log(ratios[j].sum(1)))
            logX2 = (logX + np.log(ratios[j,:,0] + 1)
                    - np.log(ratios[j].sum(1) + 1))
            stdlogX = 0.5 * np.log(np.exp(logX2/2/logX) - 1.)
            plt.sca(axes[i])
            plt.fill_between(self.Ematch, logX+1.96*stdlogX, logX-1.96*stdlogX,
                             alpha=alphafill, color=colors[i])
            for j1 in clusters[j]:
                ns = self.runs[j1].nlive
                rlogX = (np.log(ns) - np.log(ns+1)).cumsum()
                plt.plot(self.runs[j1].Emax, rlogX, c='k', alpha=alphaline)
            #plt.legend(loc='upper right')
            plt.sca(axes[-1])
            plt.fill_between(self.Ematch, logX+1.96*stdlogX, logX-1.96*stdlogX,
                             alpha=alphafill, color=colors[i], label=j)

        for j1, r in enumerate(self.runs):
            ns = r.nlive
            logX = (np.log(ns) - np.log(ns+1)).cumsum()
            plt.plot(r.Emax, logX, c='k', alpha=alphaline)

        plt.legend(loc='upper left')

        return f, axes









