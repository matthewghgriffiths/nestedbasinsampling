# -*- coding: utf-8 -*-

from itertools import izip
from bisect import bisect_left
import cPickle

import numpy as np

from nestedbasinsampling.utils import SortedCollection
from nestedbasinsampling.disconnectivitydatabase import Run


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
            if storeds[i][currstored[i]] == currind[i]:
                storednew.append(len(Emaxnew)-1)
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

class NestedSamplingRun(object):
    """
    """
    def __init__(self, Vmax=[], nlive=None, volume=1., Vcut=np.inf, configs=None):
        self.Vmax = Vmax
        self.nlive = [1 for V in Vmax] if nlive is None else nlive
        self.volume = volume
        self.Vcut = Vcut
        self.configs = None

    def combine(self, run):
        """
        Joins this nested sampling run with the nested sampling run passed in

        parameters
        ----------
        run : NestedSamplingRun (or derived class)
            the nested sampling run joining with the current class

        returns
        -------
        newrun : NestedSamplingRun (or derived class)
            the combined nested sampling run
        """
        Vmax1 = self.Vmax
        nlive1 = self.nlive
        Vcut1 = self.Vcut

        Vmax2 = run.Vmax
        nlive2 = run.nlive
        Vcut2 = run.Vcut

        n1, n2 = len(Vmax1), len(Vmax2)
        i1, i2 = 0, 0

        Vmaxnew = []
        nlivenew = []

        while(i1!=n1 or i2!=n2):
            V1 = Vmax1[i1] if i1 < n1 else -np.inf
            live1 = nlive1[i1] if i1 < n1 else 0
            V2 = Vmax2[i2] if i2 < n2 else -np.inf
            live2 = nlive2[i2] if i2 < n2 else 0

            if (V1 > V2):
                Vmaxnew.append(V1)
                nlive = live1
                if V1 < Vcut2:
                    nlive += live2
                nlivenew.append(nlive)
                i1 += 1
            else:
                Vmaxnew.append(V2)
                nlive = live2
                if V2 < Vcut1:
                    nlive += live1
                nlivenew.append(nlive)
                i2 += 1

        Vcut = max(Vcut1, Vcut2)

        return type(self)(Vmax=Vmaxnew, nlive=nlivenew,
                          volume=self.volume, Vcut=Vcut)

    __add__ = combine # So Nested Sampling runs can be easily added together

    def calcBasinFracVolume(self):
        nlive = np.array(self.nlive)
        self.frac = (nlive) / (nlive+1.)
        self.fracVolume = np.cumprod(self.frac)
        self.basinVolume = self.fracVolume*self.volume
        return self.fracVolume

    def calcBasinFracDoS(self, Vi, deltaV=None, err=False):
        pass



class LivePoints(SortedCollection):
    """ object to store the results of a nested sampling run, it maintains
    a list of energies in sorted order

    Parameters:
    replicas: list of Replicas
        List of replicas Replica(x, energy)

    """
    def __init__(self, replicas=[], tol=0.):
        self._key = lambda x: x.energy
        decorated = sorted((self._key(item), item) for item in replicas)
        self._keys = [k for k, item in decorated]
        self._items = [item for k, item in decorated]

        self.tol = tol

    def pop(self):
        self._keys.pop()
        return self._items.pop()

    def copy(self):
        return self.__class__(self)

    def pop_largest(self):
        """
        Removes all replicas within tol of the highest energy replica

        Returns
        -------
        Ecuts: list of float
            List of the energies of the replicas removed
        rs: list of Replica

        """

        to_remove = bisect_left(self._keys, self._keys[-1]-self.tol)

        self._keys, Ecuts = self._keys[:to_remove], self._keys[to_remove:]
        self._items, rs = self._items[:to_remove], self._items[to_remove:]

        return Ecuts, rs

    def __repr__(self):
        return "LivePoints({:s})".format(repr(self._keys))


class NestedSampling(object):

    prtstr = "iteration {:6d}, energy cutoff: {:10.5g}, fraction left, {:10.5g}"

    def __init__(self, replicas, sampler, Etol=0., store_config=False,
                 verbose=True, iprint=1, database=None, cpfreq=100,
                 iter_number=0, f=1., run=None):


        self.sampler = sampler

        self.Etol = Etol
        self.replicas = LivePoints(replicas, self.Etol)

        self.iter_number = iter_number
        self.f = f

        self.database = database
        if self.database is not None:
            self.run = self.initialize_run() if run is None else run
        self.cpfreq = cpfreq
        self.store_config = store_config
        self.Emax = []
        self.Nlive = []
        self.Nremove = []
        self.configs = [] if self.store_config else None

        self.verbose = verbose
        self.iprint = iprint

    def initialize_run(self):
        self.run = self.database.add_run([],[],[])
        return self.run

    def save_run(self):
        self.run = self.database.update_run(self.run, self.Emax,
                                            self.Nlive, self.configs)
        return self.run

    def nested_step(self):
        """
        Performs nested sampling step
        """

        if len(self.replicas) == 0:
            return 0.

        Ecut, deadpoints, Nlive, Nremove = self.pop_replicas()

        self.f *= 1. - Nremove/(Nlive+1.)

        newreplicas = self.sampler.new_points(
            Ecut, Nremove, replicas=self.replicas, driver=self)

        for replica in newreplicas:
            if replica.energy < Ecut:
                replica.niter = self.iter_number
                self.replicas.insert(replica)

        if self.verbose and self.iter_number % self.iprint == 0:
            print self.prtstr.format(self.iter_number, Ecut, self.f)

        self.iter_number += 1

        return self.f

    def pop_replicas(self):
        """
        remove the replicas with the largest energies and store them in the max_energies array
        """
        # pull out the replicas with the largest energy

        Nlive = len(self.replicas)
        Ecuts, rs = self.replicas.pop_largest()
        Ecut = Ecuts[0]
        Nremove = len(Ecuts)

        r=rs[-1]

        self.Emax.append(r.energy)
        self.Nlive.append(Nlive)
        self.Nremove.append(Nremove)
        if self.store_config:
            self.configs.append(np.array([_r.x.copy() for _r in rs]))

        # Save configurations
        if self.database is not None and self.iter_number%self.cpfreq == 0:
            self.save_run()


        return Ecut, rs, Nlive, Nremove

    def get_state(self):
        state = dict(f=self.f, iter_number=self.iter_number,
                     replicas=self.replicas, Emax=self.Emax,
                     Nlive=self.Nlive, Nremove=self.Nremove,
                     configs=self.configs, run=self.run)
        return state

    def save_state(self, filename):

        state = self.get_state(self)
        with open(filename, 'wb') as f:
            cPickle.dump(state, f)

    def set_state(self, state):
        self.f          = state["f"]
        self.iter_number= state["iter_number"]
        self.replicas   = state["replicas"]
        self.Emax       = state["Emax"]
        self.Nlive      = state["Nlive"]
        self.Nremove    = state["Nremove"]
        self.configs    = state["configs"]
        self.run        = state["run"]

    def read_state(self, filename):
        with open(filename, 'rb') as f:
            state = cPickle.load(f)

        self.set_state(state)