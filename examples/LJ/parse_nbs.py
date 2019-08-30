#! /usr/bin/python2
# flake8: noqa
from collections import namedtuple
import os
import cPickle
from operator import attrgetter
import itertools

from tqdm import tqdm
import numpy as np
from scipy.special import polygamma, betaln, gammaln, logsumexp
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import \
    inset_axes, zoomed_inset_axes, InsetPosition
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import seaborn as sns
import networkx as nx

from pele.systems import LJCluster
from pele.utils.disconnectivity_graph import DisconnectivityGraph
from pele.thermodynamics import minima_to_cv

from nestedbasinsampling.storage import \
    Database, Minimum, Run, Replica, Path, TransitionState
from nestedbasinsampling.nestedsampling.combine import combineAllRuns
from nestedbasinsampling import calc_CV
from nestedbasinsampling.thermodynamics import heatcapacity


def plot_run(r, Emax=None, Emin=None, plot_kws=None, fill_kws=None, ax=None,
             **kwargs):
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    fill_kws = {} if fill_kws is None else fill_kws.copy()
    for key, val in kwargs.items():
        plot_kws.setdefault(key, val)
        fill_kws.setdefault(key, val)

    Es = r.Emax
    n = Es.size
    imin = n - Es[::-1].searchsorted(Emax) if Emax is not None else 0
    imax = n - Es[::-1].searchsorted(Emin) if Emin is not None else -1
    Es = Es[imin: imax]
    log_vol = r.log_frac[imin: imax]
    log_err = r.log_rel_error[imin: imax]

    if ax is None:
        ax = plt.gca()
    ax.plot(Es, log_vol, **plot_kws)
    ax.fill_between(Es, log_vol - log_err, log_vol + log_err, **fill_kws)
    return ax


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
    a0, a1 = a[comp[0]], a[comp[1]]
    b0, b1 = b[comp[0]], b[comp[1]]
    logBF = (
        (betaln(a0 + a1 - 2 * pa, b0 + b1 - 2 * pb) - betaln(2 * pa, 2 * pb)) -
        (betaln(a0 - pa, b0 - pb) + betaln(a1 - pa, b1 - pb) -
              2 * betaln(pa, pb)))
    return logBF, comp


def log_beta_evidence(a, b, pa, pb):
    return betaln(a + pa, b + pb) - betaln(pa, pb)


def group_beta(a, b, pa=0.5, pb=0.5):
    sub_groups_values = dict(
        (frozenset([i]), (ai, bi, log_beta_evidence(ai, bi, pa, pb)))
        for i, (ai, bi) in enumerate(zip(a, b)) if np.isfinite([ai, bi]).all())
    current_grouping = frozenset(sub_groups_values)
    current_evidence = sum(sub_groups_values[g][2] for g in current_grouping)
    groupings_evidence = {current_grouping: current_evidence}

    def merge_groups(g1, g2):
        gnew = g1.union(g2)
        if gnew in sub_groups_values:
            _, _, knew = sub_groups_values[gnew]
        else:
            a1, b1, k1 = sub_groups_values[g1]
            a2, b2, k2 = sub_groups_values[g2]
            anew, bnew = a1 + a2, b1 + b2
            knew = log_beta_evidence(anew, bnew, pa, pb)
            sub_groups_values[gnew] = anew, bnew, knew
        return gnew

    while len(current_grouping) > 1:
        current_evidence = -np.inf
        for g1, g2 in itertools.combinations(current_grouping, 2):
            gnew = merge_groups(g1, g2)
            new_grouping, ignore = [gnew], set([g1, g2])
            for g in current_grouping:
                if g not in ignore:
                    new_grouping.append(g)

            new_evidence = sum(sub_groups_values[g][2] for g in new_grouping)

            grouping = frozenset(new_grouping)
            groupings_evidence[grouping] = new_evidence
            if new_evidence > current_evidence:
                current_evidence = new_evidence
                current_grouping = grouping

    return dict((g, k) for g, k in groupings_evidence.items())


def interpolate_run(int_Es, run):
    Es = run.Emax[::-1]
    return (np.interp(int_Es, Es, run.log_frac[::-1], left=-np.inf),
            np.interp(int_Es, Es, run.log_frac2[::-1], left=-np.inf))


def interpolate_run_up(int_Es, run):
    Es = run.Emax[::-1]
    return (np.interp(int_Es, Es, run.log_frac_up[::-1], left=-np.inf),
            np.interp(int_Es, Es, run.log_frac2_up[::-1], left=-np.inf))


def match_runs(runs, log_step=np.log(2), minlive=300):
    agg_runs = combineAllRuns(runs)
    nlive = agg_runs.nlive
    imax = nlive.size - nlive[::-1].searchsorted(minlive)
    Es = agg_runs.Emax[:imax]
    log_vol = agg_runs.log_frac[:imax]
    int_log_vol = np.arange(-log_step, log_vol[-1], - log_step)
    int_Es = np.interp(int_log_vol, log_vol[::-1], Es[::-1])
    r_Es, r_Vs, r_V2s = zip(*(
        (r.Emax, r.log_frac, r.log_frac2) for r in runs))
    int_vols = np.array([np.interp(int_Es, _Es[::-1], _V[::-1], left=-np.inf)
                         for _Es, _V in zip(r_Es, r_Vs)])
    int_vol2s = np.array([np.interp(int_Es, _Es[::-1], _V[::-1], left=-np.inf)
                          for _Es, _V in zip(r_Es, r_V2s)])
    log_ratio = np.diff(np.c_[[0.]*len(int_vols), int_vols], axis=1)
    log_ratio2 = np.diff(np.c_[[0.]*len(int_vol2s), int_vol2s], axis=1)
    a_rat, b_rat = logmoment2Beta(log_ratio, log_ratio2)
    return int_Es, int_vols, int_vol2s, a_rat, b_rat


def group_runs(runs, log_step=np.log(2), minlive=300, pa=0.5, pb=0.5,
               progress_bar=False):
    int_Es, _, _, a_rat, b_rat = match_runs(
        runs, log_step=log_step, minlive=minlive)
    _i = tqdm(zip(a_rat.T, b_rat.T)) if progress_bar else zip(a_rat.T, b_rat.T)
    groups = [group_beta(*ab, pa=pa, pb=pb) for ab in _i]
    groupings = set(grouping for group in groups for grouping in group)
    groupings_evidence = dict(
        (g, np.array([gs.get(g, -np.inf) for gs in groups])) for g in groupings)
    return int_Es, groupings_evidence


def _build_graph(runs, log_step=np.log(2), minlive=300, pa=0.5, pb=0.5,
                progress_bar=False):
    int_Es, _, _, a_rat, b_rat = match_runs(
        runs, log_step=log_step, minlive=minlive)
    G = nx.DiGraph()
    for i, (ai, bi) in enumerate(zip(a_rat, b_rat)):
        G.add_node(
            frozenset([i]), a=ai, b=bi, logZ=log_beta_evidence(ai, bi, pa, pb))

    merge_indexes = dict()
    current_grouping = set(G.node)

    def merge_groups(g1, g2):
        gnew = g1.union(g2)
        pair = frozenset((g1, g2))
        if pair in merge_indexes:
            merge_i = merge_indexes[pair]
        else:
            a1, b1, Z1 = (G.node[g1][k] for k in ['a', 'b', 'logZ'])
            a2, b2, Z2 = (G.node[g2][k] for k in ['a', 'b', 'logZ'])

            if gnew in G.node:
                Znew = G.node[gnew]['logZ']
            else:
                anew, bnew = a1 + a2, b1 + b2
                Znew = log_beta_evidence(anew, bnew, pa, pb)
                G.add_node(gnew, a=anew, b=bnew, logZ=Znew)

            logBF = np.nan_to_num(Znew - Z1 - Z2)
            merge_logp = logBF.cumsum() - logBF[::-1].cumsum()[::-1]
            merge_i = merge_logp.argmax()
            merge_indexes[pair] = merge_i

        return gnew, merge_i

    while len(current_grouping) > 1:
        merge_i = 0
        for g1, g2 in itertools.combinations(current_grouping, 2):
            gnew, new_i = merge_groups(g1, g2)
            if new_i > merge_i:
                update = (g1, g2, gnew)
                new_group = gnew
                merge_i = new_i

        G.add_edge(new_group, update[0])
        G.add_edge(new_group, update[1])
        G.node[new_group]['merge'] = int_Es[merge_i]
        current_grouping.symmetric_difference_update(update)

    return int_Es, G


def _calc_disconnectivity_graph(graph, **kwargs):
    basin_Es = [(node, attr['merge']) for node, attr in graph.nodes(True)
                if 'merge' in attr]

    dg = nx.Graph()
    dg.add_nodes_from(lone_mins)
    for i, m1 in enumerate(lone_mins):
        for j, m2 in enumerate(lone_mins[:i]):
            pair = set([i, j])
            tE = min(E for g, E in basin_Es if pair.issubset(g))
            dg.add_edge(m1, m2, ts=TransitionState(tE, None, m1, m2))

    dg_p = DisconnectivityGraph(dg, **kwargs)
    dg_p.calculate()

    return dg_p

def build_disconnectivity_graph(G, linkage, **kwargs):
    dg = nx.Graph()
    n = len(linkage)
    link = np.c_[linkage, range(n + 1, 2*n + 1)]
    Ecuts = np.unique(link[:, 2])
    for Ecut in Ecuts:
        l = link[(link[:, 2] == Ecut).nonzero()]
        new = dict()
        for n1, n2, E, n, n3 in l:
            new[n3] = [n1] + (new.pop(n2) if n2 in new else [n2])

        for p, nodes in new.items():
            splitR = Replica(Ecut + minE, None)
            new_run = combineAllRuns(
                [G.node[n]['run'] for n in nodes]).split(None, splitR)
            children = [G.node[n]['children'] for n in nodes]
            G.add_node(
                int(p), energy=Ecut + minE, run=new_run,
                children=reduce(set.union, children),
                nruns=sum(G.node[n]['nruns'] for n in nodes))
            for n in nodes:
                G.add_edge(int(p), int(n))
                G.node[n]['run'] = G.node[n]['run'].split(splitR)

            for pair in itertools.combinations(children, 2):
                for m1, m2 in itertools.product(*pair):
                    if isinstance(m1, Minimum) and isinstance(m2, Minimum):
                        dg.add_edge(m1, m2, ts=TransitionState(
                            Ecut + minE, None, m1, m2))

    kwargs.setdefault('Emax', linkage[:,3].max())
    dg_p = DisconnectivityGraph(dg, **kwargs)
    dg_p.calculate()
    return dg_p


def find_run_merge(merge_runs, progress_bar=True):
    Es, groupings = group_runs(merge_runs, progress_bar=progress_bar)
    norm = logsumexp(groupings.values(), axis=0)
    inds = np.triu_indices(len(merge_runs), k=1)
    mergeEs = np.zeros_like(inds[0], dtype=float)
    for i, pair in enumerate(zip(*inds)):
        pair = set(pair)
        ev = logsumexp([val for group, val in groupings.items() if
                        any(pair.issubset(g) for g in group)], axis=0)
        logBF = ev - np.log1p(-np.exp(ev - norm)) - norm
        logBF[~np.isfinite(logBF)] = 0.
        log_p = logBF.cumsum() - logBF[::-1].cumsum()[::-1]
        log_p -= logsumexp(log_p)
        mergeEs[i] = Es[log_p.argmax()]

    return Es, groupings, mergeEs


def find_run_harmonic_energy(min_run, min_energy, k, p=0.5):
    min_run = min_combined_runs[m]
    Emin = min_run.Emax[- min_run.nlive[::-1].searchsorted(minlive)]
    Emax = min_run.Emax[0]
    harm_vols = np.arange(
        0.5 * k * np.log(Emin - min_energy),
        0.5 * k * np.log(Emax - min_energy), -np.log(p))[::-1]
    Es = np.exp(harm_vols / k * 2) + min_energy
    log_vol, log_vol2 = interpolate_run(Es, min_run)
    log_ratio = np.diff(np.r_[0., log_vol])
    log_ratio2 = np.diff(np.r_[0., log_vol2])
    a, b = logmoment2Beta(log_ratio, log_ratio2)
    logZ = log_beta_evidence(a, b, pa, pb)
    logH = a * np.log(p) + b * np.log(1. - p)
    logBF = logH - logZ
    log_p = - logBF.cumsum() + logBF[::-1].cumsum()[::-1]
    log_p -= logsumexp(log_p)
    acs, bcs = a[::-1].cumsum()[::-1], b[::-1].cumsum()[::-1]
    logZc = log_beta_evidence(acs, bcs, pa, pb)
    logHc = acs * np.log(p) + bcs * np.log(1. - p)
    logBFc = logHc - logZc
    mlogBFc = np.minimum.accumulate(logBFc[::-1])[::-1]
    Eharm = Es[mlogBFc.searchsorted(0.)]
    return Eharm, (Es, log_vol, log_vol2, logBF, mlogBFc)

def calc_node_volume_up(i, G, Eres=1e-4):
    node = G.node[i]
    Es = node['Es']
    if 'harm_energy' in node:
        m = node['minimum']
        Eharm = min(node['harm_energy'], Es.max())
        i_harm = Es.size - Es[::-1].searchsorted(Eharm, side='right')
        Eharm = Es[i_harm]
        Eharms = np.arange(Eharm, m.energy, -Eres)
        log_harm_vols = (0.5 * k * np.log(Eharms - m.energy) - 0.5 * m.fvib
                         - np.log(m.pgorder))
        Es = np.r_[Es[:i_harm], Eharms]
        r = node['run']
        run_up = r.split(None, Replica(Eharm, None))
        run_vol_u = run_up.log_frac_up
        run_vol2_u = run_up.log_frac2_up
        harm_vols = log_harm_vols + harm_avg_diff
        run_vol_d = np.interp(
            Eharms, node['run'].Emax[::-1], node['run'].log_frac[::-1])
        run_var_d = (np.interp(
            Eharms, r.Emax[::-1], r.log_frac2[::-1], left=-np.inf) -
                     run_vol_d * 2)**2
        run_vol_d += harm_vols[0] - run_vol_d[0]
        weight = np.interp(Eharms, [m.energy, Eharm], [1, 0])
        harm_weight = weight / harm_avg_var
        run_weight = (1 - weight) / run_var_d
        harm_vols_u = (
            harm_weight * harm_vols + run_weight * run_vol_d) / (
                harm_weight + run_weight)
        harm_vol2s_u = 2*harm_vols_u + harm_avg_var**0.5
        run_vol_u += harm_vols_u[0]
        run_vol2_u += 2*harm_vols_u[0] + harm_avg_var**0.5
        log_vol_u = np.r_[run_vol_u[1:], harm_vols_u]
        log_vol2_u = np.r_[run_vol2_u[1:], harm_vol2s_u]
        top_vol_u, top_vol2_u = run_vol_u[0], run_vol2_u[0]

        if 'log_vol_d' in node:
            Es_d = node['Es']
            node['log_vol_d'] = np.interp(
                Es, Es_d[::-1], node['log_vol_d'][::-1])
            node['log_vol2_d'] = np.interp(
                Es, Es_d[::-1], node['log_vol2_d'][::-1])

        node['Es'] = Es
    else:
        children = [
            calc_node_volume_up(j, G, Eres) for j in G.successors(i)]
        if children:
            bottom_vol_us = [child['top_vol_u'] for child in children]
            bottom_vol2_us = [child['top_vol2_u'] for child in children]
            bottom_vol_u = logsumexp(bottom_vol_us)
            bottom_vol2_u = logsumexp(bottom_vol2_us)
            run_vol_u = node['run'].log_frac_up + bottom_vol_u
            run_vol2_u = node['run'].log_frac2_up + bottom_vol2_u
            top_vol_u, top_vol2_u = run_vol_u[0], run_vol2_u[0]
            log_vol_u = np.r_[run_vol_u[1:], bottom_vol_u]
            log_vol2_u = np.r_[run_vol2_u[1:], bottom_vol2_u]
            if all('log_frac2' in child for child in children):
                log_f2 = logsumexp(
                    [child['log_frac2'] for child in children])
                log_vol2_u -= log_f2
        else:
            log_vol_u, log_vol2_u = node['log_vol_d'], node['log_vol2_d']
            top_vol_u, top_vol2_u = node['top_vol_d'], node['top_vol2_d']

    # Saving results
    node['log_vol_u'] = log_vol_u
    node['log_vol2_u'] = log_vol2_u
    node['top_vol_u'] = top_vol_u
    node['top_vol2_u'] = top_vol2_u

    return node

AggRuns = namedtuple("AggRuns", "min_runs min_local_runs all_runs")

if __name__ == '__main__':
    sns.set(style="ticks", color_codes=True, font_scale=1.)
    plt.ion()

    natoms = 31
    k = 3*natoms - 6
    system = LJCluster(natoms)
    pot = system.get_potential()
    opt = system.get_minimizer()
    radius = 2.5
    filename = 'nbs-noguts/lj31_08.sqlite'
    minRes = 1e-1
    pkl_file = filename + '.agg.pickle'
    name = filename.split('.')[0]

    db = Database(filename)
    properties = dict((p.name, p.value) for p in db.properties())
    print(name)
    print("\n".join("%s = %r" % item for item in properties.items()))

    minima = db.minima()
    runs = db.runs()
    paths = db.paths()
    rep_mins = dict((p.parent, p.child) for p in paths)
    path_mins = dict((p, p.child) for p in paths)
    parents = dict()
    for r in runs:
        parents.setdefault(r.parent, []).append(r)

    greplica = next((p for p in parents if p.coords is None), None)
    mreplicas = [p for p in parents if p.coords is not None]
    min_runs = dict()
    min_local_runs = dict()
    for r in runs:
        m = rep_mins[r.child]
        if r.Emax[-1] < m.energy + minRes:
            if r.parent == greplica:
                min_runs.setdefault(m, []).append(r)
            else:
                min_local_runs.setdefault(m, []).append(r)

    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            agg_runs = cPickle.load(f)

        all_runs = agg_runs.all_runs
        min_agg_run = dict(
            (db.session.query(Minimum).get(m.id()), r)
            for m, r in agg_runs.min_runs.items())
        min_local_run = dict(
            (db.session.query(Minimum).get(m.id()), r)
            for m, r in agg_runs.min_local_runs.items())
    else:
        min_agg_run = dict((m, combineAllRuns(rs))
                           for m, rs in tqdm(min_runs.items()))
        min_local_run = dict(
            (m, combineAllRuns(rs))
            for m, rs in tqdm(min_local_runs.items()))
        all_runs = combineAllRuns(min_agg_run.values())

        agg_runs = AggRuns(
            min_runs=min_agg_run, min_local_runs=min_local_run, all_runs=all_runs)

        with open(pkl_file, 'wb') as f:
            cPickle.dump(agg_runs, f)

    p = pa = pb = 0.5
    log_step = np.log(2)
    min_combined_runs = min_agg_run.copy()
    for m, local_run in min_local_run.items():
        agg_run = min_agg_run[m]
        Es, vols, vol2s, a, b = match_runs(
            [agg_run, local_run], log_step=log_step)
        logBF = np.nan_to_num(
            log_beta_evidence(a.sum(0), b.sum(0), pa, pb) -
            log_beta_evidence(a, b, pa, pb).sum(0))
        merge_logp = logBF[::-1].cumsum()[::-1] - logBF.cumsum()
        split = merge_logp.argmax()
        if split:
            local_run = local_run.split(Replica(Es[split], None))

        min_combined_runs[m] = combineAllRuns([agg_run, local_run])

    minE = db.get_lowest_energy_minimum().energy
    minlive = 300
    sep_mins = ([], [])
    agg_mins, lone_mins = sep_mins
    for m, r in min_combined_runs.items():
        sep_mins[r.nlive[0] > minlive].append(m)

    for m in lone_mins:
        m.pgorder = system.get_pgorder(m.coords)
        m.fvib = system.get_log_product_normalmode_freq(m.coords)

    lone_mins.sort(key=attrgetter('energy'))
    agg_run = combineAllRuns([min_agg_run[m] for m in sep_mins[0]])
    merge_runs = [min_combined_runs[m] for m in lone_mins] + [agg_run]
    #merge_runs = [min_agg_run[m] for m in lone_mins] + [agg_run]
    nruns = [min_agg_run[m].nlive[0] for m in lone_mins] + [agg_run.nlive[0]]

    if 0:
        log_step = np.log(2)
        int_Es, _, _, a_rat, b_rat = match_runs(
            merge_runs, log_step=log_step, minlive=minlive)
        a_cum = a_rat.cumsum(1)
        b_cum = b_rat.cumsum(1)
        plt.plot(int_Es, (a_cum/(a_cum + b_cum)).T)
        plt.xlim(-140, -110)
        plt.semilogx(int_Es - minE, (a_cum/(a_cum + b_cum)).T)

    Es, groupings, mergeEs = find_run_merge(merge_runs)
    dist = mergeEs - minE
    linkage = hierarchy.linkage(dist, method='complete')
    den = hierarchy.dendrogram(linkage)

    ##########################################################################
    ## Building Graph
    ##########################################################################

    G = nx.DiGraph()
    for i, m in enumerate(lone_mins):
        r = merge_runs[i]
        G.add_node(
            i, energy=m.energy, run=r, children=set([m]), nruns=nruns[i])

    i += 1
    G.add_node(i, energy=agg_run.Emax[-1], run=agg_run,
               children=set([agg_run]), nruns=nruns[i])

    dg_mat = linkage.copy()
    dg_mat[:, 3] += minE
    dg = build_disconnectivity_graph(G, dg_mat, Emax=dg_mat[:, 3].max() + 10.)
    dg.plot()

    node = G.node[max(G)]
    nodes = G.successors(max(G))
    r = node['run']
    node['Es'] = r.Emax
    node['log_vol_d'] = r.log_frac
    node['log_vol2_d'] = r.log_frac2
    node['top_vol_d'] = 0.
    node['top_vol2_d'] = 0.
    while nodes:
        i = nodes.pop()
        parent = G.node[G.predecessors(i)[0]]
        node = G.node[i]
        nodes.extend(G.successors(i))
        p_nruns, c_nruns = parent['nruns'], node['nruns']
        log_f = np.log(c_nruns) - np.log(p_nruns)
        log_f2 = (
            log_f + np.log(c_nruns + 1) - np.log(p_nruns+1))
        node['log_frac'] = log_f
        node['log_frac2'] = log_f2
        parent_vol = parent['log_vol_d'][-1]
        parent_vol2 = parent['log_vol2_d'][-1]
        r = node['run']
        i_max = r.nlive.size - r.nlive[::-1].searchsorted(minlive)
        node['Es'] = r.Emax
        node['top_vol_d'] = parent_vol + log_f
        node['top_vol2_d'] = parent_vol2 + log_f2
        node['log_vol_d'] = parent_vol + log_f + r.log_frac
        node['log_vol2_d'] = parent_vol2 + log_f2 + r.log_frac2

    p = 0.5
    k = 3*natoms - 6
    log_harm_vols = []
    log_harm_diffs = []
    log_harm_vol2s = []
    for i, m in enumerate(lone_mins):
        min_run = min_combined_runs[m]
        Eharm, (Es, log_vol, log_vol2, logBF, mlogBFc) = \
            find_run_harmonic_energy(min_run, m.energy, k)
        logp = mlogBFc - logsumexp(mlogBFc)
        p_harm = np.exp(logp).cumsum()
        node = G.node[i]
        node['harm_energy'] = Eharm
        node['p_harm'] = np.interp(node['Es'], Es[::-1], p_harm[::-1])
        ######
        node['minimum'] = m
        Es = node['Es']
        Eharm = min(Eharm, Es.max())
        harm_vol = (0.5 * k * np.log(Eharm - m.energy) - 0.5 * m.fvib -
                    np.log(m.pgorder))
        log_vol = node['log_vol_d'][-Es[::-1].searchsorted(Eharm)]
        log_vol2 = node['log_vol2_d'][-Es[::-1].searchsorted(Eharm)]
        log_harm_vols.append(log_vol)
        log_harm_diffs.append(log_vol - harm_vol)
        log_harm_vol2s.append(log_vol2)

    min_Es = [m.energy for m in lone_mins]
    min_Es, log_harm_diffs, log_harm_vols, log_harm_vol2s = map(
        np.array, [min_Es, log_harm_diffs, log_harm_vols, log_harm_vol2s])
    log_harm_err = log_harm_vol2s - 2*log_harm_vols
    harm_avg_diff = np.average(log_harm_diffs, weights=log_harm_err**-2)
    harm_avg_var = (log_harm_err**-2).sum()**-1

    calc_node_volume_up(max(G), G)
    for i, node in G.nodes(True):
        log_vol_u = node.get('log_vol_u')
        log_vol2_u = node.get('log_vol2_u')
        if log_vol_u is not None:
            log_vol_d = node['log_vol_d']
            log_vol2_d = node['log_vol2_d']

            log_rel_u = (log_vol2_u - 2*log_vol_u)**2
            log_rel_d = (log_vol2_d - 2*log_vol_d)**2
            zero_u = log_rel_u == 0.
            log_rel_u[zero_u] = 1.
            log_rel_d[zero_u] = np.inf
            log_rel = (log_rel_u**-1 + log_rel_d**-1)**-1

            # Weighted sum
            log_vol = (log_vol_u/log_rel_u + log_vol_d/log_rel_d) * log_rel
            log_vol2 = 2*log_vol + log_rel
            node['log_vol'] = log_vol
            node['log_vol2'] = log_vol2
        else:
            node['log_vol'] = node['log_vol_d']
            node['log_vol2'] = node['log_vol2_d']


    BSfile = 'rephd/Cv.out.D'
    PTfile = 'rephd/Cv.nores.reference'
    BS_Ts, _, _, _, _, BS_Cv = np.loadtxt(BSfile).T
    PT_Ts, _, _, _, _, PT_Cv = np.loadtxt(PTfile).T
    logZ, logE1, logE2, _ = (
        logsumexp(out, axis=0) for out in
        zip(*(heatcapacity._calc_thermodynamics(
            attr['Es'], attr['log_vol'], PT_Ts, minE, ntrap=2000)
              for attr in G.node.values())))
    ret = heatcapacity._calcCv(logZ, logE1, logE2, PT_Ts, k, minE)


    logZ, logE1, logE2, _ = (
        logsumexp(out, axis=0) for out in
        zip(*(heatcapacity._calc_thermodynamics(
            attr['Es'], attr['log_vol_u'], PT_Ts, minE, ntrap=2000)
              for attr in G.node.values())))
    ret_up = heatcapacity._calcCv(logZ, logE1, logE2, PT_Ts, k, minE)
    logZ, logE1, logE2, _ = (
        logsumexp(out, axis=0) for out in
        zip(*(heatcapacity._calc_thermodynamics(
            attr['Es'], attr['log_vol_d'], PT_Ts, minE, ntrap=2000)
              for attr in G.node.values())))
    ret_down = heatcapacity._calcCv(logZ, logE1, logE2, PT_Ts, k, minE)

    nlive = 20000
    ns_file = 'LJ31-K{:d}/lj31.energies'.format(nlive)
    ns_pickle = 'LJ31-K{:d}/lj31.ns'.format(nlive)
    with open(ns_pickle, 'r') as f:
        ns_data = cPickle.load(f)

    Es = np.loadtxt(ns_file)[:-1]
    ns_run = Run(Es, np.ones_like(Es, int) * nlive)
    ns_res = calc_CV(ns_run.Emax, ns_run.log_frac, k=k, Ts=PT_Ts)

    opt = system.get_minimizer()
    for m in tqdm(minima):
        res = opt(m.coords)
        m.energy = res.energy
        m.pgorder = system.get_pgorder(res.coords)
        m.fvib = system.get_log_product_normalmode_freq(res.coords)

    hsa = minima_to_cv(minima, PT_Ts, k=k)

    sns.set(style="ticks", color_codes=True, font_scale=1.)
    plt.rc('font',**{'family':'serif'})
    plt.rc('text', usetex=True)
    plt.rcParams.update({"text.latex.preamble":[r'\usepackage{palatino}']})
    width = 483.7/72
    plt.rcParams["mathtext.fontset"] = "custom"

    width = 483.7/72
    fig, ax = plt.subplots(figsize=(width,0.7*width))
    ax.plot(PT_Ts, PT_Cv, label='PT')
    ax.plot(BS_Ts, BS_Cv, label='BSPT')
    # ax.plot(PT_Ts, hsa.Cv, label='HSA')
    ax.plot(PT_Ts, ns_res.Cv, label='NS')
    ax.plot(PT_Ts, ret.Cv, label='NBS')

    ax.set_xlim(0,0.6)
    ax.set_ylim(80, 180)
    ax.set_ylabel(r"C${}_\textrm{V}$(T) / k${}_\textrm{\footnotesize B}$")
    ax.set_xlabel(r"k${}_\textrm{B}$T/$\epsilon_\textrm{\footnotesize LJ}$")

    axins = plt.axes([0.018,80,0.036,130])
    ip = InsetPosition(ax, [0.05,.64,0.3,0.4])
    axins.set_axes_locator(ip)

    axins.set_xlim(0.0125,0.045)
    axins.set_ylim(82,130)
    axins.set_yticks(range(90,121,10))
    axins.plot(PT_Ts, PT_Cv, label='PT')
    axins.plot(BS_Ts, BS_Cv, label='BSPT')
    axins.plot(PT_Ts, ns_res.Cv, label='NS')
    axins.plot(PT_Ts, ret.Cv, label='NBS')
    axins.yaxis.tick_right()
    axins.xaxis.tick_top()
    axins.xaxis.set_tick_params(pad=2)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.subplots_adjust(top=0.87,hspace=0.1)

    fig.savefig('LJ31_CV.pdf')
    fig.savefig('LJ31_CV.png', dpi=600)


    fig, ax = plt.subplots(figsize=(width,0.45*width))
    ax.plot(PT_Ts, PT_Cv, label='PT')
    ax.plot(BS_Ts, BS_Cv, label='BSPT')
    # ax.plot(PT_Ts, hsa.Cv, label='HSA')
    ax.plot(PT_Ts, ns_res.Cv, label='NS')
    ax.plot(PT_Ts, ret.Cv, label='NBS')

    ax.set_xlim(0,0.6)
    ax.set_ylim(80, 180)
    ax.set_ylabel(r"C${}_\textrm{V}$(T) / k${}_\textrm{\footnotesize B}$")
    ax.set_xlabel(r"k${}_\textrm{B}$T/$\epsilon_\textrm{\footnotesize LJ}$")

    axins = plt.axes([0.0518,80,0.036,130])
    ip = InsetPosition(ax, [0.05,.64,0.3,0.4])
    axins.set_axes_locator(ip)

    axins.set_xlim(0.015,0.045)
    axins.set_ylim(82,130)
    axins.set_yticks(range(90,121,10))
    axins.plot(PT_Ts, PT_Cv, label='PT')
    axins.plot(BS_Ts, BS_Cv, label='BSPT')
    axins.plot(PT_Ts, ns_res.Cv, label='NS')
    axins.plot(PT_Ts, ret.Cv, label='NBS')
    axins.yaxis.tick_right()
    axins.xaxis.tick_top()
    axins.xaxis.set_tick_params(pad=2)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.subplots_adjust(top=0.87,hspace=0.1)

    fig.savefig('LJ31_CV_s.pdf')
    fig.savefig('LJ31_CV_s.png', dpi=600)
    import subprocess
    subprocess.check_output(['pdfcrop', 'LJ31_CV_s.pdf', 'LJ31_CV_s.pdf'])

    f, ax = plt.subplots(figsize=(width/2, width/2.))
    dg.plot(axes=ax)
    ax.set_ylabel(r"Energy /$\epsilon_{\textrm{\footnotesize LJ}}$")
    ax.set_ylim(-134, -118)
    ax.set_yticks(np.arange(-133,-118,2))
    f.tight_layout()
    f.savefig("LJ31_DG.pdf")

    import subprocess
    subprocess.check_output(['pdfcrop', 'LJ31_DG.pdf', 'LJ31_DG.pdf'])

    min_Es = np.array([m.energy for m in min_runs])
    min_freq = np.array(map(len, min_runs.values()))
    min_prob = min_freq * 1./ min_freq.sum()
    ind = min_prob.argsort()[::-1]
    nbar = np.arange(ind.size + 1)
    pbar = np.r_[0., min_prob[ind].cumsum()]

    ##########################################################################

    fig, ax = plt.subplots(figsize=[width*0.5, width*0.4])
    ax.step(nbar, pbar)
    axins = plt.axes([0,0,1,1])
    ip = InsetPosition(ax, [0.3,.07,0.49,0.7])
    axins.set_axes_locator(ip)

    ax.set_xlim(-10, ind.size)
    ax.set_ylim(0, 1)
    axins.set_xlim(-2, 20)
    axins.set_ylim(0, 0.8)
    axins.step(nbar, pbar)
    axins.yaxis.tick_right()
    axins.xaxis.tick_top()

    axins.xaxis.set_tick_params(pad=1)

    axins.set_xticks([0,10,20])
    ax.set_xticks([0, 200, 400, 600, 800])

    ax.set_ylabel(r"Cumulative probability")
    ax.set_xlabel(r"Number of minima")
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    fig.tight_layout()
    fig.savefig("LJ31_CDF.pdf")

    ##########################################################################

    fig, ax = plt.subplots(figsize=[width*0.5, width*0.4])
    ax.scatter(min_Es, min_prob, marker='+')
    ax.set_yscale('log')
    ax.set_ylim(3e-5,1.)
    ax.set_xlim(-134,-127)
    ax.set_xticks(np.arange(-133,-126,2))
    ax.set_xlabel(r"Minimum energy /$\epsilon_\textrm{\footnotesize LJ}$")
    ax.set_ylabel('Probability')
    fig.tight_layout()
    fig.savefig("LJ31_PDF.pdf")

    colors = sns.color_palette(n_colors=6)
    from matplotlib import ticker

    fig, ax = plt.subplots(figsize=(0.5 * width,0.4*width))
    ax.plot(PT_Ts, PT_Cv, label='PT', color=colors[0])
    ax.semilogx(PT_Ts, ret.Cv, label="Interpolated", color=colors[3])
    ax.plot(PT_Ts, ret_down.Cv, label="Top-down", color=colors[5])
    ax.plot(PT_Ts, ret_up.Cv, label="Bottom-up", color=colors[4])
    ax.set_xlim(0.0125, 0.25)
    ax.set_ylim(80, 160)
    formatter = ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    ax.xaxis.set_major_formatter(formatter)
    ax.set_ylabel(r"C${}_\textrm{V}$(T) / k${}_\textrm{\footnotesize B}$")
    ax.set_xlabel(r"k${}_\textrm{B}$T/$\epsilon_\textrm{\footnotesize LJ}$")
    ax.legend()
    fig.subplots_adjust(top=0.955,
        bottom=0.198,
        left=0.22,
        right=0.973,
        hspace=0.2,
        wspace=0.2)
    fig.savefig('LJ31_BU_TD.pdf')
    fig.savefig('LJ31_BU_TD.png', dpi=600)
