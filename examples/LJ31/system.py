import logging

from pele.potentials import LJ
from nestedbasinsampling import (
    NoGUTSSampler, NestedOptimizerKalman, HardShellConstraint, random_structure,
    RecordMinimization, CompareStructures, LOG_CONFIG, Database)


default_sampler_kws = dict(
    max_depth=7, linear_momentum=True, angular_momentum=True)
default_nopt_kws = dict(
    nsteps=2000, MC_steps=5, target_acc=0.4, nsave=10, tol=1e-2)
default_struct_kws = dict(niter=100)
default_database_kws = dict()

class NBS_LJ(object):
    """
    """
    def __init__(self, natoms, radius=None, stepsize=None,
                 sampler_kws=None, nopt_kws=None, stepsize_kw=None,
                 struct_kws=None, database_kws=None):
        self.pot = LJ()
        self.natoms = natoms
        self.radius = float(natoms) ** (1. / 3) if radius is None else radius
        self.constraint = HardShellConstraint(self.radius)

        self.sampler_kws = default_sampler_kws.copy()
        if sampler_kws is not None: self.sampler_kws.update(sampler_kw)
        self.sampler = NoGUTSSampler(
            self.pot, constraint=self.constraint, **self.sampler_kws)

        self.nopt_kws = default_nopt_kws.copy()
        if nopt_kws is not None: self.nopt_kws.update(nopt_kws)

        self.struct_kws = default_struct_kws.copy()
        if struct_kws is not None: self.struct_kws.update(struct_kws)

        self.database_kws = default_database_kws.copy()
        if database_kws is not None: self.database_kws.update(database_kws)
        if 'compareMinima' not in self.database_kws:
            self.database_kws['compareMinima'] = self.get_compare_structures()

        if stepsize is None:
            kws = {} if stepsize_kw is None else stepsize_kw
            s = self.determine_stepsize(
                target_acc=self.nopt_kws['target_acc'], **kws)
            self.stepsize = s[-1]
        else:
            self.stepsize = stepsize

    def determine_stepsize(self, coords=None, E=None, **kwargs):
        if coords is None: coords = self.random_config()
        if E is None: E = self.pot.getEnergy(coords)
        s = self.sampler.determine_stepsize(coords, E, **kwargs)
        return s

    def random_config(self):
        return random_structure(self.natoms, self.radius)

    def nopt(self, coords=None, Ecut=None, stepsize=None):
        if coords is None: coords = self.random_config()
        if Ecut is None: Ecut = self.pot.getEnergy(coords)
        if stepsize is None: stepsize = self.stepsize

        opt = NestedOptimizerKalman(
            coords, self.pot, sampler=self.sampler,
            energy=Ecut, stepsize=stepsize, **self.nopt_kws)
        return dict(opt.run())

    def get_configuration(self):
        coords = self.random_config()
        Ecut = self.pot.getEnergy(coords)
        stepsize = self.stepsize
        return coords, Ecut, stepsize

    def get_compare_structures(self):
        return CompareStructures(**self.struct_kws)

    def get_database(self, db=":memory:"):
        return Database(db, **self.database_kws)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, **LOG_CONFIG)
    system = NBS_LJ(natoms=31, stepsize=0.1)
    res = system.nopt()
