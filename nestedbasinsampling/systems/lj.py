
from pele.potentials import LJ
from ..structure import HardShellConstraint
from ..sampling import random_structure
from .nbs_system import NBS_system

default_sampler_kws = dict(
    max_depth=7, remove_linear_momentum=True, remove_angular_momentum=True,
    remove_initial_linear_momentum=False, remove_initial_angular_momentum=False)

class LJ_system(NBS_system):

    def __init__(self, natoms, radius=None,
                 stepsize=None, sampler_kws=None, nopt_kws=None,
                 stepsize_kw=None, struct_kws=None, database_kws=None):
        self.natoms = natoms
        if radius is None:
            radius = float(natoms) ** (1. / 3)
        self.radius = radius
        pot = LJ()
        constraint = HardShellConstraint(radius)
        random_config = lambda : random_structure(self.natoms, self.radius)

        _sampler_kws = default_sampler_kws.copy()
        if sampler_kws is not None: _sampler_kws.update(sampler_kws)

        super(LJ_system, self).__init__(
            pot, random_config, constraint, database_kws=database_kws,
            stepsize=stepsize, sampler_kws=_sampler_kws, nopt_kws=nopt_kws,
            stepsize_kw=stepsize_kw, struct_kws=struct_kws)
