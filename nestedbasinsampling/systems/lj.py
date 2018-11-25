import logging

from pele.potentials import LJ
from ..structure import HardShellConstraint, CompareStructures
from ..sampling import random_structure, NoGUTSSampler
from .nbs_system import NBS_system


logger = logging.getLogger("NBS.LJ_system")


class NBS_LJ(NBS_system):
    """
    """
    default_sampler_kws = dict(
        max_depth=7, remove_linear_momentum=True, remove_angular_momentum=True,
        remove_initial_linear_momentum=False,
        remove_initial_angular_momentum=False)

    def __init__(self, natoms, radius=None, stepsize=None,
                 sampler_kws=None, nopt_kws=None, stepsize_kws=None,
                 struct_kws=None, database_kws=None, _Sampler=NoGUTSSampler):
        self.natoms = natoms
        pot = LJ()
        self.radius = float(natoms) ** (1. / 3) if radius is None else radius
        constraint = HardShellConstraint(self.radius)

        super(NBS_LJ, self).__init__(
            pot, constraint=constraint, stepsize=stepsize,
            sampler_kws=sampler_kws, nopt_kws=nopt_kws,
            stepsize_kws=stepsize_kws,  struct_kws=struct_kws,
            database_kws=database_kws, _Sampler=_Sampler)

    def determine_stepsize(self, coords=None, E=None, **kwargs):
        if coords is None:
            coords = self.get_random_configuration()
        if E is None:
            E = self.pot.getEnergy(coords)
        s = self.sampler.determine_stepsize(coords, E, **kwargs)
        return s

    def get_random_configuration(self):
        return random_structure(self.natoms, self.radius)

    def get_configuration(self):
        coords = self.get_random_configuration()
        Ecut = self.pot.getEnergy(coords)
        stepsize = self.stepsize
        return coords, Ecut, stepsize

    def get_compare_structures(self):
        return CompareStructures(**self.struct_kws)


if __name__ == '__main__':
    from nestedbasinsampling import LOG_CONFIG
    logging.basicConfig(level=logging.DEBUG, **LOG_CONFIG)
    natoms = 31
    radius = 2.5
    nopt_kws = dict(
        nsteps=2000, MC_steps=5, target_acc=0.4, nsave=10, tol=1e-2, iprint=1,
        nwait=5, debug=True)
    system = NBS_LJ(natoms, radius, nopt_kws=nopt_kws)
    res = system.nopt()
