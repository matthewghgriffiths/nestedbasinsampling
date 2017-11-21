# -*- coding: utf-8 -*-

from itertools import izip
from bisect import bisect_left
import cPickle

import numpy as np

#from nestedbasinsampling.utils import SortedCollection
from nestedbasinsampling.storage import Run

def findRunSplit(run, splitf=0.5):
    logt = np.log(run.nlive) - np.log(run.nlive+1)
    logF = logt.cumsum()
    i = np.searchsorted(-logF, -logF[-1]*splitf)
    return run.Emax[i], i

def joinRuns(run1, run2):
    assert run1.child == run2.parent

    stored1    = run1.stored
    configs1   = run1.configs

    stored2    = run2.stored + len(run1.Emax)
    configs2   = run2.configs

    if len(stored1) and len(stored2):
        stored = np.r_ [stored1, stored2]
        configs = np.r_[configs1, configs2]
    elif len(stored2) == 0:
        stored = stored1
        configs = configs1
    elif len(stored1) == 0 and len(run1.Emax):
        stored1 = [len(run1.Emax)-1]
        configs1 = np.array([run1.child.coords])
        stored = np.r_[stored1, stored2]
        configs = np.r_[configs1, configs2]
    else:
        stored = stored2
        configs = configs2

    Emax = np.r_[run1.Emax, run2.Emax]
    nlive = np.r_[run1.nlive, run2.nlive]
    vol = run1.volume
    stepsizes = np.r_[run1.stepsizes, run2.stepsizes]

    return Run(Emax, nlive, run1.parent, run2.child, volume=vol,
               stored=stored, configs=configs, stepsizes=stepsizes)

def combineAllRuns(runs, parent=None, child=None):
    """ Combines all the runs passed into a single combined run, starting and
    finishing at the parent and child replica (if specified).

    If the parent or child are not specified, then this function will pick
    the highest energy parent replica or lowest energy child replica from runs

    Parameters
    ----------
    runs : list of Run
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
    store_stepsize = all(len(config)==len(stepsize)
                         for config, stepsize in izip(configs, stepsizes))

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
            if storeds[i].size and storeds[i][currstored[i]] == currind[i]:
                storednew.append(len(Emaxnew)-1)
                if configs[i]:
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
            i = Emaxcurr.argmax()
            if Emaxcurr[i] == Ecurr and currind[i] == 0:
                currind[i] = 1
                Emaxcurr[i] = Emaxs[i][currind[i]]
                nactive[i] = nlives[i][currind[i]]
                nlive += nactive[i]
        else:
            Emaxcurr[i] = Emaxs[i][currind[i]]
            nactive[i] = nlives[i][currind[i]]
            nlive += nactive[i]

        # Get the highest energy configuration
        i = Emaxcurr.argmax()
        Ecurr = Emaxcurr[i]

    newRun = Run(Emaxnew, nlivenew, parent, child,
                 stored=storednew, configs=configsnew, stepsizes=stepsizesnew)
    return newRun

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

def combineRuns(run1, run2):
    Emax1 = run1.Emax
    nlive1 = run1.nlive
    Ecut1 = run1.parent.energy
    vol1 = run1.volume

    Emax2 = run2.Emax
    nlive2 = run2.nlive
    Ecut2 = run2.parent.energy
    vol2 = run2.volume

    # For storing stepsizes/configurations
    store = len(run1.stored) or len(run2.stored)
    stored1 = run1.stored if len(run1.stored) else np.array([])
    stored2 = run2.stored if len(run2.stored) else np.array([])

    stored = []
    configs = []
    stepsizes = []

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
            if store:
                v1 = np.searchsorted(stored1, i1, side='left')
                if v1 < len(run1.configs) and i1 == stored1[v1]:
                    stored.append(len(Emaxnew)-1)
                    if len(run1.configs):
                        configs.append(run1.configs[v1])
                    if v1 < len(run1.stepsizes): # dirty hack
                        stepsizes.append(run1.stepsizes[v1])
            i1 += 1
        else:
            Emaxnew.append(E2)
            nlive = live2
            if E2 < Ecut1:
                nlive += live1
            nlivenew.append(nlive)
            if store:
                v2 = np.searchsorted(stored2, i2, side='left')
                if v2 < len(run2.configs) and i2 == stored2[v2] :
                    stored.append(len(Emaxnew)-1)
                    if len(run2.configs):
                        configs.append(run2.configs[v2])
                    if v2 < len(run2.stepsizes): # dirty hack
                        stepsizes.append(run2.stepsizes[v2])
            i2 += 1


    parent = max([run1.parent, run2.parent], key=lambda r:r.energy)
    child = min([run1.child, run2.child], key=lambda r:r.energy)
    volume = (vol1 + vol2)/2
    stored = stored if len(stored) else None
    configs = configs if len(configs) else None
    stepsizes = stepsizes if len(stepsizes) else None

    return Run(Emaxnew, nlivenew, parent, child, volume=volume,
               stored=stored, configs=configs, stepsizes=stepsizes)

def calcRunWeights(run):

    ns = run.nlive.astype(float)
    Emax = run.Emax.copy()

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

    weights = dict(Emax=Emax, dF=dF, d2F2=d2F2, dF2=dF2,
                   XdF=XdF, X2dF=X2dF,
                   XdF2=XdF2, Xd2F2=Xd2F2,
                   X2d2F2=X2d2F2, X2dF2=X2dF2,
                   dvarf=dvarf, dvarX=dvarX)

    return weights

def calcAverageValue(weights, func, std=True):
    Emax = weights['Emax']
    dF = weights['dF']
    XdF = weights['XdF']
    X2dF = weights['X2dF']
    d2F2 = weights['d2F2']
    dF2 = weights['dF2']
    Xd2F2 = weights['Xd2F2']
    XdF2 = weights['XdF2']
    X2d2F2 = weights['X2d2F2']
    X2dF2 = weights['X2dF2']

    fj = np.atleast_2d(func(Emax))
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