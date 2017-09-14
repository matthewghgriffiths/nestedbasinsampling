# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm

from pele.potentials import BasePotential

class BaseConstraint(BasePotential):

    def __call__(self, coords):
        return self.getEnergy(coords) <= 0.

class HardShellConstraint(BaseConstraint):


    def __init__(self, radius, ndim=3):
        self.radius = radius
        self.ndim = 3

    def getEnergy(self, coords):
        pos = np.array(coords).reshape((-1,self.ndim))
        dr = norm(pos, axis=1) - self.radius
        return dr[dr>0.].sum()

    def getGradient(self, coords):
        pos = np.array(coords).reshape((-1,self.ndim))
        dr = norm(pos, axis=1)
        outside = (dr - self.radius)>0.

        gradient = np.zeros_like(pos)
        gradient[outside] = pos[outside]/dr[outside,None]

        return gradient.reshape(coords.shape)


coords = np.random.normal(0,0.5,(15,3))

pos = np.array(coords).reshape((-1,3))
pos -= pos.mean(0)[None,:]

dr = norm(pos, axis=1) - 1.
outside = dr>0.

constraint = HardShellConstraint(1.)

constraint.getGradient(coords)
constraint.NumericalDerivative(coords.flatten(),1e-8).reshape(pos.shape)