# -*- coding: utf-8 -*-

from itertools import izip, groupby
from collections import defaultdict

import numpy as np

from nestedbasinsampling.nestedsampling.combine import combineAllRuns
from nestedbasinsampling.nestedsampling.bayesian import (
    BayesianBetaMixture, _estimate_beta_parameters)

def logsumexp(arr, axis=0):
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out

def matchRuns(runs, log_step=1., log_Xmin=None, Emin=None):
    """
    """
    run = combineAllRuns(runs)
    logX = run.log_frac
    rlogX = [r.log_frac for r in runs]
    rlogX2 = [r.log_frac2 for r in runs]

    if log_Xmin is None:
        Emin = max((r.Emax[-1]) for r in runs) if Emin is None else Emin
        imin = -run.Emax[::-1].searchsorted(Emin)
        log_Xmin = logX[imin]

    Lsplits = np.arange(-log_step, log_Xmin, -log_step)
    isplits = logX[::-1].searchsorted(Lsplits)
    Esplits = run.Emax[-isplits]

    risplits = [-r.Emax[::-1].searchsorted(Esplits) for r in runs]
    risplits0 = [np.r_[0,i[:-1]] for i in risplits]

    dlogX = np.array(
        [lX[i1] - lX[i0] for i1, i0, lX in izip(risplits, risplits0, rlogX)])
    dlogX2 = np.array(
        [lX[i1] - lX[i0] for i1, i0, lX in izip(risplits, risplits0, rlogX2)])

    a, b = logmoment2Beta(dlogX, dlogX2)
    return Esplits, np.concatenate((a[:,:,None], b[:,:,None]), axis=2)

def logmoment2Beta(logm1, logm2):
    """
    """
    logvar = logm2 + np.log(1 - np.exp(2*logm1 - logm2))
    logm3 = logm1 + np.log(1 - np.exp(logm2 - logm1))
    ns = np.exp(logm1 + logm3 - logvar)
    nr = np.exp(logm3 - logvar + np.log(1. - np.exp(logm1)))
    return ns, nr

class Cluster(frozenset):

    def __new__(cls, iterable, loglike=-np.inf, parameters=None):
        obj = super(Cluster, cls).__new__(Cluster, iterable)
        obj.loglike = loglike
        obj.parameters = parameters
        return obj

    def __repr__(self):
        return "Cluster({:s},loglike={:8.4g})".format(
            [list(i) for i in self], self.loglike)

    __str__ = __repr__

class ClusterRuns(object):

    def __init__(self, runs, replicas=None, Efinish=None,
                 ntries=50, log_step=1.,
                 n_components=None, init_params='random', beta_prior=1.,
                 **kwargs):

        self.init_params = init_params
        self.log_step = log_step
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

    def addRuns(self, runs, replicas=None, Efinish=None):
        """"""
        self.runs += runs
        self.nruns = len(self.runs)
        self.replicas += (range(self.nruns - len(runs), self.nruns)
                         if replicas is None else replicas)
        assert len(self.replicas) == len(self.runs)

        self.Ematch, self.AB = matchRuns(
            self.runs, log_step=self.log_step, Emin=Efinish)

    @property
    def cluster(self):
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

    @property
    def p_mat(self):
        if not self.allclusters:
            self.fit()

        clusters = self.allclusters

        likes = np.array([c.loglike for c in clusters])
        likes -= likes.max()
        ps = np.exp(likes)
        ps /= ps.sum()

        connected = np.zeros((len(self.AB),)*2)
        for p, c in izip(ps, clusters):
            self.dpbmm._set_parameters(c.parameters)
            log_prob = self.dpbmm._estimate_weighted_log_prob(self.AB)
            log_prob -= logsumexp(log_prob, 1)[:,None]
            connected += np.exp(
                log_prob[:,None,:] + log_prob[None,:,:]).sum(2) * p

        d = connected.diagonal()
        connected /= np.sqrt(d[:,None]*d[None,:])
        return connected

    def fit(self):
        """"""
        for i in xrange(self.ntries):
            self.do_fit('random')
        self.do_fit([0]*len(self.AB))

        return self.cluster

    def _classify(self, AB, replicas):
        """"""
        labels = self.dpbmm.predict(AB)
        clusters = defaultdict(list)
        [clusters[i].append(r) for i, r in izip(labels, replicas)]
        return clusters

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









