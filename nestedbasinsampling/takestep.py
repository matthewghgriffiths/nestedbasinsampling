# -*- coding: utf-8 -*-

import numpy as np

from pele.takestep import TakestepInterface, AdaptiveStepsize
from nestedbasinsampling.random import vector_random_uniform_hypersphere

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