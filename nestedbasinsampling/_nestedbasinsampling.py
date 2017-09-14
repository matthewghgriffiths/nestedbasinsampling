# -*- coding: utf-8 -*-

import cPickle

import numpy as np

from pele.systems import BaseSystem
from pele.optimize import lbfgs_cpp

from nestedbasinsampling.utils import dict_update_keep, SortedCollection, Replica
from nestedbasinsampling.database import Database
from nestedbasinsampling.nestedsampling import NestedSampling
from nestedbasinsampling.samplers import MCSampler
from nestedbasinsampling.takestep import AdaptiveStepsize, TakestepHyperSphere
from nestedbasinsampling.nestedoptimization import NestedGalileanOptimizer

class SavePairs(object):
    """
    Class for saving the joint energies
    """
    def __init__(self, pairfile=None, isave=10000):
        self.pairfile = pairfile
        self.isave = isave

        if pairfile is not None:
            self.pairs = self.readresults()
        else:
            self.pairs = SortedCollection()

    def add_pair(self, basinE, instantE):
        self.pairs.insert((basinE,instantE))
        if len(self.pairs) % self.isave == 0:
            self.saveresults()

    def saveresults(self):
        if self.pairfile is not None:
            with open(self.pairfile, 'wb') as f:
                cPickle.dump(self.pairs, f)

    def readresults(self, pairfile):
        with open(pairfile, 'rb') as f:
            pairs = cPickle.load(f)
        return pairs

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

    def get_savepairs(self, pairfile=None, isave=10000):
        return SavePairs(pairfile, isave)

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



from pele.systems import LJCluster

natoms = 13
natoms = 31
niter = 100
system = LJCluster(natoms)

nsys = NestedBasinSystem(system)


def constraint(coords):
    pos = coords.reshape(-1,3)
    pos -= pos.mean(0)[None,:]
    return (np.linalg.norm(coords.reshape(-1,3),axis=1) < 3.).all()

nsys.get_constraint = lambda : constraint

pot = system.get_potential()
npot = nsys.get_potential()
nsys.get_potential = lambda : pot


coords = nsys.get_random_configuration()
while(not constraint(coords) or pot.getEnergy(coords)>0):
    coords = nsys.get_random_configuration()

opt = NestedGalileanOptimizer(coords, system.get_potential(),
                              constraint=constraint, iprint=1)
opt.run()

#replicas = []
#while len(replicas) < 5:
#    replicas += [r for r in nsys.get_replicas(10) if r.energy < -5]

replicas = nsys.get_replicas(5, -5)

raise

nsys.params.database['db'] = "nbs.sql"
nsys.params.database['commit_interval'] = 1000

npot = nsys.get_potential()
db = nsys.create_database()

nbs = nsys.get_nestedbasinsampling(replicas)

for i in xrange(1000):
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
