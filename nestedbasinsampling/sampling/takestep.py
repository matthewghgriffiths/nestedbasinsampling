# -*- coding: utf-8 -*-

import numpy as np

from pele.takestep import TakestepInterface, AdaptiveStepsize

"""
Functions related to random vectors
IMPORTED FROM PELE (POTENTIAL ENERY LANDSCAPE EXPLORATOR)
source can be found at https://github.com/pele-python/pele
"""

def vec_random():
    """uniform random unit vector"""
    p = np.zeros(3)
    u1, u2 = np.random.random(2)
    z = 2 * u1 - 1.
    p[0] = np.sqrt(1 - z * z) * np.cos(2. * np.pi * u2)
    p[1] = np.sqrt(1 - z * z) * np.sin(2. * np.pi * u2)
    p[2] = z
    return p

def random_structure(natoms, radius=1.):
    """
    Random structure where all atoms are selected randomly inside sphere of
    constant radius
    """
    return np.concatenate([vector_random_uniform_hypersphere(3)*radius
                            for _ in xrange(natoms)])

def vec_random_ndim(n):
    """n-dimensional uniform random unit vector"""
    v = np.random.normal(size=n)
    v /= np.linalg.norm(v)
    return v

def vector_random_uniform_hypersphere(k):
    """return a vector sampled uniformly in a hypersphere of dimension k"""
    if k == 3:
        u = vec_random()
    else:
        u = vec_random_ndim(k)
    # draw the magnitude of the vector from a power law density:
    # draws samples in [0, 1] from a power distribution with positive exponent k - 1.
    p = np.random.power(k)
    return p * u

def random_step(coords, stepsize=1.0):
    coords = np.asanyarray(coords)
    p = vec_random_ndim(coords.size).reshape(coords.shape)*stepsize
    return p

def hypersphere_step(coords, stepsize=1.0):
    """
    Generates random direction vector
    """
    coords = np.asanyarray(coords)
    p = vector_random_uniform_hypersphere(coords.size).reshape(coords.shape)*stepsize
    return p

class TakestepHyperSphere(TakestepInterface):

    def __init__(self, stepsize=1.0):
        self.stepsize = stepsize

    def takeStep(self, coords, **kwargs):
        unitstep = vector_random_uniform_hypersphere(coords.size)
        coords += unitstep.reshape(coords.shape) * self.stepsize

    def scale(self, factor):
        self.stepsize *= factor

class AdaptiveTakestepHyperSphere(TakestepHyperSphere):

    def __init__(self, stepsize=1.0, acc_ratio=0.5,
                 alpha=0.001, P=1.0):

        self.stepsize = stepsize
        self.acc_ratio = acc_ratio

        self.alpha = alpha
        self.ialpha = 1. - alpha
        self.P = P
        self.ema = acc_ratio

        self.record = []

    def updateStep(self, accepted, **kwargs):

        self.ema *= self.ialpha
        if accepted:
            self.ema += self.alpha

        err = self.ema-self.acc_ratio

        force = np.exp(self.P*err)
        self.scale(force)

        self.record.append([accepted, err, self.Ierr, force, self.stepsize])
