# -*- coding: utf-8 -*-

from bisect import bisect_left
import cPickle

import numpy as np

from nestedbasinsampling.utils import SortedCollection

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