# -*- coding: utf-8 -*-
from copy import copy
from collections import defaultdict

import numpy as np
from pele.optimize import lbfgs_cpp

from nestedbasinsampling.utils import dict_update_keep, Result

class BasinPotential(object):
    """
    """
    def __init__(self, pot, quench=None, database=None):
        self.pot = pot
        self.quench = quench
        self.database = database
        self.nfev = 0
        if self.database is not None:
            self.storage = self.database.minimumRes_adder()

    def getEnergy(self, coords, **kwargs):
        self.res = self.quench(coords, **kwargs)
        basinE = self.res.energy

        if self.database is not None:
            self.min = self.storage(self.res)
        return basinE

class RecordMinimization(object):

    def __init__(self, pot, minimizer=lbfgs_cpp, **kwargs):

        self.pot = pot
        self.minimizer = minimizer

        events = kwargs.get('events', [])
        events.append(self.store)
        kwargs['events'] = events
        self.min_kw = kwargs

    def __call__(self, coords, **kwargs):
        self.store = defaultdict(list)

        new_kw = kwargs.copy()
        if 'events' in new_kw:
            new_kw['events'] += self.min_kw['events']
        dict_update_keep(new_kw, self.min_kw)

        res = self.minimizer(coords, pot=self.pot, **new_kw)

        if res is None:
            res = Result()
            res.coords = coords.copy()
            res.energy, res.grad = self.pot.getEnergyGradient(coords)
            res.nsteps = 0
            res.success = False
            return res

        for key, item in self.store.iteritems():
            if hasattr(res, key):
                setattr(res, key+'_s', np.array(item))
            else:
                setattr(res, key, np.array(item))
        return res

    def store(self, **kwargs):
        for key, item in kwargs.iteritems():
            self.store[key].append(copy(item))