# -*- coding: utf-8 -*-

import numpy as np

from nestedbasinsampling.utils import dict_update_copy

try:
    from fastoverlap import BranchnBoundAlignment, PeriodicAlignFortran
    have_fortran = True
except ImportError:
    have_fortran = False
    from fastoverlap import SphericalAlign, PeriodicAlign

if have_fortran:
    align_cluster = BranchnBoundAlignment
    align_periodic = PeriodicAlignFortran
else:
    align_cluster = SphericalAlign
    align_periodic = PeriodicAlign

class CompareStructures(object):
    def __init__(self, perm=None, boxvec=None, natoms=None, tol=1e-2,
                 **align_kw):
        self.perm = perm
        self.boxvec = boxvec
        self.tol = tol
        if natoms is None:
            if self.perm is None:
                self.natoms = 1
            else:
                self.natoms = sum(map(len, self.perm))
        self.align = (self.align_cluster if boxvec is None else
                      self.align_periodic)
        self.align_kw = align_kw
                      
    def align_cluster(self, min1, min2, **kwargs):
        pos1 = np.asanyarray(min1.coords).reshape(-1,3)
        pos2 = np.asanyarray(min2.coords).reshape(-1,3)
        
        perm = kwargs.pop('perm') if kwargs.has_key('perm') else self.perm
        kw = dict_update_copy(kwargs, self.align_kw)
        return align_cluster(perm=perm)(pos1, pos2, **kw)

    def align_periodic(self, min1, min2, **kwargs):
        pos1 = np.asanyarray(min1.coords).reshape(-1,3)
        pos2 = np.asanyarray(min2.coords).reshape(-1,3)
        
        natoms = len(pos1)
        perm = kwargs.pop('perm') if kwargs.has_key('perm') else self.perm
        align = align_periodic(natoms, self.boxvec, perm=perm)
        kw = dict_update_copy(kwargs, self.align_kw)
        return align(pos1, pos2, **kw)

    def __call__(self, min1, min2):
        dist = self.align(min1, min2)[0]
        return dist < self.tol

if __name__ == "__main__":
    from nestedbasinsampling.takestep import random_structure
    from nestedbasinsampling.database import Minimum

    natoms = 30

    pos1 = random_structure(natoms)
    pos2 = pos1 + random_structure(natoms, 1e-2)
    pos2 = pos2.reshape(-1,3).dot([[0,1,0],[-1,0,0],[0,0,1]]).flatten()
    
    min1, min2 = Minimum(0, pos1), Minimum(0, pos2)

    print 'Distance = ', align_cluster()(pos1, pos2)[0]
    print 'Same = ', CompareStructures()(min1, min2)

    pos1 = random_structure(natoms)
    pos2 = pos1 + random_structure(natoms, 1e-8)
    pos2 = pos2.reshape(-1,3).dot([[0,1,0],[-1,0,0],[0,0,1]]).flatten()
    min1, min2 = Minimum(0, pos1), Minimum(0, pos2)
    
    print 'Distance = ', align_cluster()(pos1, pos2)[0]
    print 'Same = ', CompareStructures()(min1, min2)
