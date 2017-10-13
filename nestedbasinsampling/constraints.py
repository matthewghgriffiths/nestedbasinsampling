# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm

from pele.potentials import BasePotential

class BaseConstraint(BasePotential):

    def __call__(self, coords):
        return self.getEnergy(coords) <= 0.

    def getEnergy(self, coords):
        return 0.

    def getGradient(self, coords):
        return np.zeros_like(coords)

    def getEnergyGradient(self, coords):
        return self.getEnergy(coords), self.getGradient(coords)

class HardShellConstraint(BaseConstraint):

    def __init__(self, radius, ndim=3):
        self.radius = radius
        self.radius2 = radius**2
        self.ndim = ndim

    def getEnergy(self, coords):
        pos = np.array(coords).reshape((-1,self.ndim))
        r2 = (pos**2).sum(1)
        return (np.sqrt(r2[r2>self.radius2])-self.radius).sum()

    def getGradient(self, coords):
        pos = np.array(coords).reshape((-1,self.ndim))
        r2 = (pos**2).sum(1)

        outside = r2 > self.radius2
        gradient = np.zeros_like(pos)
        gradient[outside] = pos[outside]/np.sqrt(r2[outside,None])
        return gradient.reshape(coords.shape)

    def getEnergyGradient(self, coords):
        pos = np.array(coords).reshape((-1,self.ndim))
        r2 = (pos**2).sum(1)

        outside = r2 > self.radius2
        gradient = np.zeros_like(pos)
        rout = np.sqrt(r2[outside,None])
        gradient[outside] = pos[outside]/rout

        return rout.sum(), gradient.reshape(coords.shape)

class HardShellConstraint2(BaseConstraint):

    def __init__(self, radius, ndim=3):
        self.radius = radius
        self.ndim = ndim

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


if __name__ == "__main__":
    coords = np.random.normal(0,0.3,(15,3))
    p = np.random.normal(0,0.3,(15,3))

    dr = norm(coords, axis=1) - 1.
    outside = dr>0.

    constraint = HardShellConstraint(1.)

    print constraint.getEnergy(coords + p)
    if not constraint(coords + p):
        G = constraint.getGradient(coords + p)
        n = norm(G)
        p -= 2 * p.ravel().dot(G.ravel()) * G

    print constraint.getEnergy(coords + p)

    constraint.getGradient(coords)
    constraint.NumericalDerivative(coords.flatten(),1e-8).reshape(coords.shape)