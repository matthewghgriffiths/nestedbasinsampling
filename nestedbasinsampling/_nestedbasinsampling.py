# -*- coding: utf-8 -*-

import cPickle

import numpy as np

from pele.systems import BaseSystem
from pele.optimize import lbfgs_cpp

from nestedbasinsampling.utils import dict_update_keep, SortedCollection, Replica
from nestedbasinsampling.database import Database
from nestedbasinsampling.nestedsampling import NestedSampling
from nestedbasinsampling.samplers import MCSampler, GalileanSampler
from nestedbasinsampling.takestep import AdaptiveStepsize, TakestepHyperSphere
from nestedbasinsampling.nestedoptimization import NestedGalileanOptimizer, NestedOptimizer

class BasinPotential(object):

    def __init__(self, pot, quench=None, database=None):

        self.pot = pot
        self.quench = quench
        self.database = database
        self.nfev = 0

        self.database = database
        self.storage = self.database.minimumRes_adder()

    def getEnergy(self, coords):
        res = self.quench(coords)
        basinE = res.energy

        if self.database is not None:
            self.storage(res)

        return basinE

class NestedBasinSystem(BaseSystem):

    def __init__(self, system):
        self.system = system
        self.params = self.system.params
        self.params.nestedsampler = dict(nsteps=10, maxtries=100, verbose=True)
        self.params.nestedbasinsampling = dict(store_config=True,iprint=1,
                                               Etol=1e-5, cpfreq=1000)
        self.params.basin_quench = dict(
            tol=1e-1, alternate_stop_criterion=None,
            events=None, iprint=-1, nsteps=10000, logger=None, debug=False,
            energy=None, gradient=None, sampler_kw = {},
            quench=lbfgs_cpp, quenchtol=1e-6,
            quench_kw=self.params.structural_quench_params)
        self.get_random_configuration = system.get_random_configuration

    def get_potential(self):
        min_kw = self.params.structural_quench_params
        db_kw = self.params.database
        return BasinPotential(self.system.get_potential(),
                              quench=self.get_minimizer(**min_kw),
                              database=self.create_database(**db_kw))

    def get_takestep(self):
        return AdaptiveStepsize(TakestepHyperSphere())

    def get_constraint(self):
        return lambda x: True

    def get_sampler(self, **kwargs):
        dict_update_keep(kwargs, self.params.nestedsampler)
        return MCSampler(self.get_potential(), takestep=self.get_takestep(),
                         constraint=self.get_constraint(), **kwargs)

    def get_minimizer(self, **kwargs):
        """return a function to minimize the structure

        Notes
        The function should be one of the optimizers in `pele.optimize`, or
        have similar structure.

        See Also
        --------
        pele.optimize
        nestedbasinsampling.nestedoptimization
        """
        pot = self.system.get_potential()
        dict_update_keep(kwargs, self.params.basin_quench)
        if 'constraint' not in kwargs:
            kwargs['constraint'] = self.get_constraint()
        return lambda coords: NestedGalileanOptimizer(coords, pot, **kwargs).run()

    def create_database(self, db=None, **kwargs):
        """return a new database object

        See Also
        --------
        nestedbasinsampling.database
        """
        dict_update_keep(kwargs, self.params.database)
        if db is not None:
            kwargs['db'] = db

        try:
            overwrite_properties = kwargs.pop("overwrite_properties")
        except KeyError:
            overwrite_properties = True

        # get a routine to compare the minima as exact
        try:
            if not "compareMinima" in kwargs:
                compare_minima = self.system.get_compare_minima()
                kwargs["compareMinima"] = compare_minima
        except NotImplementedError:
            # compareMinima is optional
            pass

        db = Database(**kwargs)

        db.add_properties(self.system.get_system_properties(),
                          overwrite=overwrite_properties)
        return db

    def get_replicas(self, nreplicas, Ecut=np.inf):
        pot = self.get_potential()
        constraint = self.get_constraint()
        replicas = []

        while(len(replicas)<nreplicas):
            coords = self.get_random_configuration()
            if constraint(coords):
                E = pot.getEnergy(coords)
                if E < Ecut:
                    replicas.append(Replica(coords, E))
        return replicas

    def get_nestedbasinsampling(self, replicas, sampler=None, **kwargs):
        """
        Inputs:
        replicas, integer or list of replicas
        """
        try:
            replicas = self.get_replicas(replicas)
        except TypeError:
            pass

        sampler = self.get_sampler() if sampler is None else sampler

        dict_update_keep(kwargs, self.params.nestedbasinsampling)

        if 'database' not in kwargs:
            kwargs['database'] = self.create_database()
        nbs = NestedSampling(replicas, sampler, **kwargs)
        return nbs


if __name__ == "__main__":
    from pele.systems import LJCluster
    from nestedbasinsampling.constraints import HardShellConstraint
    from nestedbasinsampling.random import random_structure
    import matplotlib.pyplot as plt
    from plottingfuncs.plotting import ax3d

    natoms = 13
    natoms = 31
    niter = 100
    system = LJCluster(natoms)

    nsys = NestedBasinSystem(system)

    radius =  float(natoms) ** (1. / 3)
    rand_config = lambda : random_structure(natoms, radius)
    constraint = HardShellConstraint(radius)

    nsys.get_constraint = lambda : HardShellConstraint(radius)

    nsys.params

    pot = system.get_potential()

    coords = nsys.get_random_configuration()

    sampler = GalileanSampler(pot, stepsize=0.5, nsteps=30,
                              constraint=constraint, verbose=1)

    opt = NestedOptimizer(coords, system.get_potential(), sampler, nsteps=1000, iprint=1)
    opt.run()

    replicas = nsys.get_replicas(10)

    npot = nsys.get_potential()

    raise


    npot = nsys.get_potential()
    nsys.get_potential = lambda : pot

    nsys.params.database['db'] = "nbs.sql"
    nsys.params.database['commit_interval'] = 1000

    npot = nsys.get_potential()
    db = nsys.create_database()

    nbs = nsys.get_nestedbasinsampling(10)

    for i in xrange(10000):
        nbs.nested_step()

    raise

    badmin = [m for m in db.minima() if m.energy > -5.]
    for m in badmin:
        db.session.delete(m)
    db.session.commit()


    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def plot3d(coords, **kwargs):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(*coords.reshape(-1,3).T, **kwargs)

        return ax

    rEs = np.array([r.energy for r in replicas])
    rEs.sort()

    nbs = nestedsystem.get_nestedbasinsampling(100)

    for i in xrange(1000):
        nbs.nested_step()
