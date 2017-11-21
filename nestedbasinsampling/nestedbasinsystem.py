# -*- coding: utf-8 -*-

import random

import numpy as np

from pele.optimize import LBFGS_CPP

from nestedbasinsampling.utils import dict_update_copy
from nestedbasinsampling.sampling import GMCSampler, DetectStep
from nestedbasinsampling.sampling.takestep import random_structure
from nestedbasinsampling.structure.constraints import BaseConstraint, HardShellConstraint
from nestedbasinsampling.optimize import AdaptiveNestedOptimizer
from nestedbasinsampling.storage import Database
from nestedbasinsampling.graphs import ReplicaGraph
from nestedbasinsampling.structure.alignment import CompareStructures

class NestedBasinSystem(object):
    """
    """
    def __init__(self, peleSystem, constraint=None, radius=None,
                 sampler=GMCSampler, sampler_kw={},
                 NOPT=AdaptiveNestedOptimizer, nopt_kw={}, nopt_sampler_kw={},
                 Minimizer=LBFGS_CPP, minimizer_kw={},
                 DetectStep=DetectStep, detectstep_kw={},
                 globalbasin=None, database=None, startseed=0):

        self.system = peleSystem
        self.natoms = self.system.natoms
        self.radius = radius

        self.pot = self.system.get_potential()
        if constraint is None:
            if radius is not None:
                self.get_constraint = self.get_hard_sphere_constraint
            else:
                self.get_constraint = BaseConstraint
        else:
            self.get_constraint = constraint

        if radius is None:
            self.get_random_configuration = self.system.get_random_configuration
        else:
            self.get_random_configuration = self.get_random_sphere

        self.sampler = sampler
        self.sampler_kw = sampler_kw

        self.NOPT = NOPT
        self.nopt_kw = nopt_kw
        self.nopt_sampler_kw = nopt_sampler_kw

        self.Minimizer = Minimizer
        self.minimizer_kw = minimizer_kw

        self.DetectStep = DetectStep
        self.detectstep_kw = detectstep_kw

        self.seed = startseed
        self.set_seed(self.seed)

        self.get_potential = self.system.get_potential
        self.get_pgorder = self.system.get_pgorder
        #self.get_get_metric_tensor = self.system.get_get_metric_tensor

    def set_seed(self, startseed):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def get_state(self):
        return dict(py=random.getstate(),np=np.random.get_state())

    def set_state(self, state):
        random.setstate(state['py'])
        np.random.set_state(state['np'])

    def get_compare_structures(self):
        return CompareStructures(niter=100)

    def get_constraint(self):
        return BaseConstraint()

    def get_hard_sphere_constraint(self):
        return HardShellConstraint(self.radius)

    def get_random_sphere(self):
        return lambda : random_structure(self.natoms, self.radius)

    def get_database(self, *args, **kwargs):
        return Database(*args, **kwargs)

    def get_replica_graph(self, database=None, **kwargs):
        database = self.get_database() if database is None else database
        return ReplicaGraph(database, **kwargs)

    def get_sampler(self, **kwargs):
        kwargs = dict_update_copy(kwargs, self.sampler_kw)
        if 'constraint' not in kwargs:
            kwargs['constraint'] = self.get_constraint()
        sampler = self.sampler(self.pot, **kwargs)
        return sampler

    def get_detect_step(self, sampler_kw={}, **kwargs):
        kwargs = dict_update_copy(kwargs, self.detectstep_kw)
        return self.DetectStep(self.get_sampler(**sampler_kw), **kwargs)

    def get_nestedbasinsampling(self):
        raise NotImplementedError

    def _nopt(self, coords, sampler_kw={}, **kwargs):

        sampler_kw = dict_update_copy(sampler_kw, self.nopt_sampler_kw)
        kwargs = dict_update_copy(kwargs, self.nopt_kw)

        pot = self.get_potential() if 'pot' not in kwargs else kwargs.pop('pot')
        ## Saving as module variables to allow inspection of values
        self._sampler = self.get_sampler(**sampler_kw)
        self._nopt = self.NOPT(coords, pot, self._sampler, **kwargs)
        self._nres = self._nopt.run()
        return self._nres

    def _minimize(self, coords, **kwargs):
        kwargs = dict_update_copy(kwargs, self.minimizer_kw)
        pot = self.get_potential() if 'pot' not in kwargs else kwargs.pop('pot')
        self._minimizer = self.Quench(coords, pot, **kwargs)
        self._mres = self._minimizer.run()
        return self._mres

    def get_nestedoptimizer(self):
        return self._nopt

    def get_minimizer(self):
        return self._minimize