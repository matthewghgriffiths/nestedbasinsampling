# -*- coding: utf-8 -*-

from itertools import izip, izip_longest, repeat, chain
import numpy as np

try:
    from .fortran import combineruns as combine
    has_fortran = True
except ImportError:
    has_fortran = False

def findRunSplit(run, splitf=0.5):
    logt = np.log(run.nlive) - np.log(run.nlive+1)
    logF = logt.cumsum()
    i = np.searchsorted(-logF, -logF[-1]*splitf)
    return run.Emax[i], i

def py_combineAllRuns_old(runs, parent=None, child=None):
    """ Combines all the runs passed into a single combined run, starting and
    finishing at the parent and child replica (if specified).

    If the parent or child are not specified, then this function will pick
    the highest energy parent replica or lowest energy child replica from runs

    Parameters
    ----------
    runs : iterable of Run
        list of nested sampling runs to be combined
    parent : Replica, optional
        the replica the nested sampling run starts from
    child : Replica, optional
        the replica the nested sampling run finishes at

    Returns
    -------
    newrun : Run
        the combined nested sampling run
    """
    runs = list(runs)
    if not runs:
        return Run([], [])
    # Need to sort arrays so that consecutive runs are joined appropriately
    runs.sort(key=lambda r: -r.parent.energy)
    # We append the parent energy to Emax so we know if a run is active or not
    Emaxs = [np.r_[run.parent.energy, run.Emax] for run in runs]
    nEmaxs = [-Emax for Emax in Emaxs]
    ns = np.array(map(len, Emaxs))
    nlives = [np.r_[0,run.nlive] for run in runs]

    store = any(len(run.stored) for run in runs)
    # Need to add 1 to the stored indexes to compensate for adding the parent
    # replica
    storeds = [run.stored + 1 if run.stored.size else np.array([])
                for run in runs]
    configs = [run.configs for run in runs]
    stepsizes = [run.stepsizes for run in runs]
    store_configs = all(len(stored)==len(configs)
                         for stored, configs in izip(storeds, configs))
    store_stepsize = all(len(stored)==len(stepsize)
                         for stored, stepsize in izip(storeds, stepsizes))

    Emaxnew = []
    nlivenew = []
    storednew = []
    configsnew = []
    stepsizesnew = []

    # Choosing the parent and child replicas
    if parent is None:
        parent = max((run.parent for run in runs), key=lambda r: r.energy)
    if child is None:
        child = min((run.child for run in runs), key=lambda r: r.energy)

    Ecut = parent.energy
    Efinish = max(min(Emax[-1] for Emax in Emaxs), child.energy)

    #Setting up state
    Ecurr = Ecut
    currind = np.array([nEmax.searchsorted(-Ecut, side='right')
                        for nEmax in nEmaxs])
    currstored = np.array([stored.searchsorted(i, side='left')
                           for stored, i in izip(storeds, currind)])
    Emaxcurr = np.array([Emax[_i] for _i, Emax in izip(currind, Emaxs)])
    nactive = np.array([nlive[_i] for _i, nlive in izip(currind, nlives)])

    nlive = nactive.sum()
    assert nlive > 0 # If not the parent replica is too high energy
    i = Emaxcurr.argmax()
    Ecurr = Emaxcurr[i]

    while(Ecurr >= Efinish):
        # Adding highest energy live point to run

        Emaxnew.append(Ecurr)
        nlivenew.append(nlive)
        if store:
            # If this configuration has been saved, save it
            if (storeds[i].size and
                currstored[i] < storeds[i].size and
                storeds[i][currstored[i]] == currind[i]):
                storednew.append(len(Emaxnew)-1)
                if store_configs:
                    configsnew.append(configs[i][currstored[i]])
                if store_stepsize:
                    stepsizesnew.append(stepsizes[i][currstored[i]])
                currstored[i] += 1

        # Updating the state
        currind[i] += 1
        nlive -= nactive[i]

        if currind[i] == ns[i]:
            Emaxcurr[i] = - np.inf
            nactive[i] = 0
            # If combining consecutive nested sampling runs then we need
            # to skip the first value of the next nested sampling run
            # this is why we need to sort the runs
            i = Emaxcurr.argmax()
            while Emaxcurr[i] == Ecurr and currind[i] == 0:
                currind[i] = 1
                Emaxcurr[i] = Emaxs[i][1]
                nactive[i] = nlives[i][1]
                nlive += nactive[i]
                i = Emaxcurr.argmax()
        else:
            Emaxcurr[i] = Emaxs[i][currind[i]]
            nactive[i] = nlives[i][currind[i]]
            nlive += nactive[i]

        if (nactive==0).all():
            break

        i = Emaxcurr.argmax()
        while currind[i] == 0:
            currind[i] = 1
            Emaxcurr[i] = Emaxs[i][1]
            nactive[i] = nlives[i][1]
            nlive += nactive[i]
            i = Emaxcurr.argmax()

        # Get the highest energy configuration
        i = Emaxcurr.argmax()
        Ecurr = Emaxcurr[i]

    newRun = Run(Emaxnew, nlivenew, parent, child,
                 stored=storednew, configs=configsnew, stepsizes=stepsizesnew)
    return newRun

def py_combineAllRuns(runs, parent=None, child=None):
    """
    """
    if parent is None:
        parent = max(
            (run.parent for run in runs), key=lambda r: r.energy)
    if child is None:
        child = min(
            (run.child for run in runs), key=lambda r: r.energy)

    _runs = [
        r.split(parent.energy, child.energy)
        if parent.energy < r.parent.energy else
        r.split(None, child.energy)
        for r in runs]
    runEs = (np.r_[r.parent.energy, r.Emax] for r in _runs)
    runnlive = (np.r_[0,r.nlive,0] for r in runs)
    runndiff = (np.diff(ns) for ns in runnlive)
    runvalues = chain(*(
        izip(Es, chain([False], repeat(True)), diff)
        for Es, diff in izip(runEs, runndiff)))

    combined = np.array(sorted(runvalues, reverse=True))
    Es, flags, diffs = combined.T

    nlive = np.cumsum(diffs)
    keep = flags.nonzero()[0]
    Esnew = Es[keep]
    nlivenew = nlive[keep-1]

    return Run(Esnew, nlivenew, parent, child)

def f90_combineAllRuns(runs, parent=None, child=None):
    """
    """
    runs = list(runs)

    if not runs:
        return Run([], [])

    if parent is None:
        parent = max((run.parent for run in runs), key=lambda r: r.energy)
    if child is None:
        child = min((run.child for run in runs), key=lambda r: r.energy)

    Emaxs = np.empty((len(runs), max([r.Emax.size for r in runs])+1))
    nlives = np.zeros_like(Emaxs, dtype=int)

    for i, r in enumerate(runs):
        ndead = r.Emax.size
        Emaxs[i,0] = r.parent.energy
        Emaxs[i,1:ndead+1] = r.Emax
        nlives[i,1:ndead+1] = r.nlive

    Emaxnew, nlivenew, ndead, info = combine.nestedsamplingutils.combineruns(
        Emaxs, nlives, parent.energy, child.energy)

    return Run(Emaxnew[:ndead], nlivenew[:ndead], parent, child)


def splitRun(run, startrep, endrep):
    """ Splits a run between startrep and endrep
    """
    configs = run.configs
    nlive = run.nlive
    stepsizes = run.stepsizes
    Emax = run.Emax
    stored = run.stored
    volume = run.volume

    istart = Emax.size - Emax[::-1].searchsorted(startrep.energy, side='left')
    iend = Emax.size - Emax[::-1].searchsorted(endrep.energy, side='left')

    jstart, jend = stored.searchsorted([istart, iend], side='left')

    newEmax = Emax[istart:iend]
    newnlive = nlive[istart:iend]
    newStored, newStepsizes, newConfigs = None, None, None
    if stored.size:
        newStored = stored[jstart:jend] - istart
        if stepsizes.size:
            newStepsizes = stepsizes[jstart:jend]
        if configs.size:
            newConfigs = configs[jstart:jend]

    return Run(newEmax, newnlive, startrep, endrep, volume=volume,
               stored=newStored, configs=newConfigs, stepsizes=newStepsizes)


if has_fortran:
    def combineAllRuns(runs, parent=None, child=None, nbatch=10):
        mergeruns = runs
        while len(mergeruns) > 1:
            splitruns = (
                (r for r in batch if r is not None)
                for batch in izip_longest(*[iter(mergeruns)]*nbatch))
            mergeruns = [f90_combineAllRuns(batch, parent, child)
                         for batch in splitruns]
        run, = mergeruns
        return run
else:
    combineAllRuns = py_combineAllRuns

from ..storage.database import Run



if __name__ == "__main__":

    from nestedbasinsampling.storage import Replica

    N = 10
    Emax1 = np.array([
        0.97779171,  0.97585986,  0.91134209,  0.85794262,  0.78839253,
        0.60348486,  0.56608212,  0.39276477,  0.1738798 ,  0.09876979])
    nlive1 = np.array([2, 1, 5, 7, 6, 6, 7, 6, 6, 9])
    Emax2 = np.array([
        0.92114771,  0.77683481,  0.63594495,  0.56132342,  0.54563075,
        0.43931577,  0.43632394,  0.31823357,  0.25653018,  0.09785394,
        0.06055449,  0.0269848 ])
    nlive2 = np.array([8, 2, 2, 6, 7, 1, 8, 3, 4, 1, 2, 1])
    parent = Replica(1., None)
    child = Replica(0., None)

    run1 = Run(Emax1, nlive1, parent=parent, child=child)
    run2 = Run(Emax2, nlive2, parent=parent, child=child)

    runs = [run1, run2]
    py_run = py_combineAllRuns(runs)
    f90_run = f90_combineAllRuns(runs)
