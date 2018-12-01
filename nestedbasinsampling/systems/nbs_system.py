import logging

from ..optimize import NestedOptimizerKalman
from ..structure import BaseConstraint,  CompareStructures
from ..sampling import NoGUTSSampler, random_structure
from ..storage import Database
from ..concurrent import BaseWorker, NBS_Manager

logger = logging.getLogger('nbs_system')


class NBS_system(object):
    """
    """
    settings = {}
    default_sampler_kws = dict(max_depth=7)
    default_nopt_kws = dict(
        nsteps=2000, MC_steps=5, target_acc=0.4, nsave=20, tol=1e-2, nwait=10)
    default_struct_kws = dict(niter=100)
    default_database_kws = dict()
    default_basin_nopt_kws = dict(
        nsteps=2000, MC_steps=5, target_acc=0.4, nsave=20, tol=1e-2, nwait=10,
        basin_optimization=True, use_quench=False)
    _sampler = None

    def __init__(self, pot, random_configuration=None, constraint=None,
                 stepsize=None, sampler_kws=None, nopt_kws=None,
                 stepsize_kws=None, struct_kws=None, database_kws=None,
                 basin_nopt_kws=None, _Sampler=NoGUTSSampler):
        self.pot = pot
        if random_configuration is not None:
            self.get_random_configuration = random_configuration
        self.constraint = (
            BaseConstraint() if constraint is None else constraint)
        self._Sampler = _Sampler
        self.settings = dict(
            stepsize=stepsize, sampler_kws=sampler_kws, nopt_kws=nopt_kws,
            stepsize_kws=stepsize_kws, struct_kws=struct_kws,
            database_kws=database_kws, basin_nopt_kws=basin_nopt_kws)
        self.initialise(**self.settings)

    def initialise(self, stepsize=None, sampler_kws=None, nopt_kws=None,
                   stepsize_kws=None, struct_kws=None, database_kws=None,
                   basin_nopt_kws=None):
        self.sampler_kws = self.default_sampler_kws.copy()
        if sampler_kws is not None:
            self.sampler_kws.update(sampler_kws)
        self.sampler = self.get_sampler(**self.sampler_kws)

        self.nopt_kws = self.default_nopt_kws.copy()
        if nopt_kws is not None:
            self.nopt_kws.update(nopt_kws)

        self.basin_nopt_kws = self.default_basin_nopt_kws.copy()
        if basin_nopt_kws is not None:
            self.basin_nopt_kws.update(basin_nopt_kws)
        elif nopt_kws is not None:
            self.basin_nopt_kws.update(nopt_kws)

        self.struct_kws = self.default_struct_kws.copy()
        if struct_kws is not None:
            self.struct_kws.update(struct_kws)

        self.database_kws = self.default_database_kws.copy()
        if database_kws is not None:
            self.database_kws.update(database_kws)
        if 'compareMinima' not in self.database_kws:
            self.database_kws['compareMinima'] = self.get_compare_structures()

        if stepsize is None:
            kws = {} if stepsize_kws is None else stepsize_kws
            s = self.determine_stepsize(
                target_acc=self.nopt_kws['target_acc'], **kws)
            self.stepsize = s[-1]
        else:
            self.stepsize = stepsize

    def get_sampler(self, **kwargs):
        self.sampler_kws = self.default_sampler_kws.copy()
        if kwargs is not None:
            self.sampler_kws.update(kwargs)

        sampler = self._Sampler(
            self.pot, constraint=self.constraint, **self.sampler_kws)
        return sampler

    def determine_stepsize(self, coords=None, E=None, **kwargs):
        if coords is None:
            coords = self.get_random_configuration()
        if E is None:
            E = self.pot.getEnergy(coords)
        s = self.sampler.determine_stepsize(coords, E, **kwargs)
        return s

    def get_random_configuration(self):
        raise NotImplementedError

    def nopt(self, coords=None, Ecut=None, stepsize=None, **kwargs):
        if coords is None:
            coords = self.get_random_configuration()
        if Ecut is None:
            Ecut = self.pot.getEnergy(coords)
        if stepsize is None:
            stepsize = self.stepsize

        for key, val in self.nopt_kws.iteritems():
            kwargs.setdefault(key, val)

        opt = NestedOptimizerKalman(
            coords, self.pot, sampler=self.sampler,
            energy=Ecut, stepsize=stepsize, **kwargs)
        return dict(opt.run())

    def get_configuration(self):
        coords = self.get_random_configuration()
        Ecut = self.pot.getEnergy(coords)
        stepsize = self.stepsize
        return coords, Ecut, stepsize

    def get_compare_structures(self):
        return CompareStructures(**self.struct_kws)

    def get_database(self, db=":memory:"):
        logger.info("Connecting to database: {:s}".format(db))
        db = Database(db, **self.database_kws)
        for name, kws in self.settings.items():
            db.add_property(name, kws, overwrite=False)
        return db

    @classmethod
    def Worker(cls):
        from numpy.random import seed
        seed(None)

        class WorkerClass(cls, BaseWorker):
            pass
            
        return WorkerClass

    def get_manager(self, db=":memory:", **kwargs):
        database = self.get_database()
        manager = NBS_Manager(self, database=database, **kwargs)
        return manager


if __name__ == "__main__":
    from pele.potentials import LJ
    from nestedbasinsampling import (
        LOG_CONFIG, random_structure, HardShellConstraint)

    logging.basicConfig(level=logging.INFO, **LOG_CONFIG)
    natoms = 31
    radius = 2.5
    pot = LJ()
    constraint = HardShellConstraint(radius)
    system = NBS_system(
        pot, constraint=constraint,
        random_configuration=lambda: random_structure(natoms, radius),
        stepsize=0.1)
    res = system.nopt()
