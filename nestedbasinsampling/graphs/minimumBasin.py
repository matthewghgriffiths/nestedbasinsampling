
from itertools import chain, izip
from random import choice, seed, sample

from math import log, exp, sqrt
import numpy as np
from scipy.special import loggamma

from scipy.spatial import ConvexHull

import networkx as nx

from pele.thermodynamics import logproduct_freq2
from pele.optimize import lbfgs_cpp, LBFGS_CPP

from nestedbasinsampling.utils import \
    dict_update_keep, dict_update_copy, weighted_choice, iter_minlength
from nestedbasinsampling.samplers import \
    SamplingError, GMCSampler, DetectStep
from nestedbasinsampling.structure.constraints import BaseConstraint
from nestedbasinsampling.nestedoptimization import \
    BasinPotential, AdaptiveNestedOptimizer, RecordMinimization
from nestedbasinsampling.disconnectivitydatabase import \
    Minimum, Replica, Run, Database
from nestedbasinsampling.graphs import \
    ReplicaGraph, BasinGraph, SuperBasin, FunctionGraph
from nestedbasinsampling.random.stats import CDF, AndersonDarling, AgglomerativeCDFClustering
from nestedbasinsampling.nestedsampling import \
    findRunSplit, joinRuns, combineRuns


class MinimumBasin(object):

    def __init__(self, minimum, system, volMin=1., nsym=1, tol=1e-3):
        self.min = minimum
        self.replicas = frozenset([self.min])
        self.coords = self.min.coords.copy()
        self.energy = self.min.energy
        self.pot = pot
        self.system = system

        self.ndim = self.system.get_ndof()
        self.nzero = self.system.get_nzero_modes()
        self.ndof = self.ndim - self.nzero
        self.perm = self.system.get_permlist()

        self.nsym = nsym

        self.freq2, self.modes = self.system.get_normalmodes(self.coords)
        if m.fvib is None:
            n, fvib = logproduct_freq2(self.freq2, self.nzero)
            m.fvib = fvib
        if m.pgorder is None:
            m.pgorder = self.system.get_pgorder(coords)
        self.fvib = m.fvib
        self.pgorder = m.pgorder

        self.volMin = volMin
        self.logVolMin = log(volMin)
        self.logSphereVol = self.ndof * log(2) - loggamma(self.ndof+1).real
        self.logConfig = self.calclogConfig()

        self.lognperm = sum(loggamma(len(p) +1).real for p in self.perm)
        self.lognisomer = log(self.nsym) + self.lognperm - log(self.pgorder)

    def calclogConfig(self):
        return self.logVolMin + self.logSphereVol - 0.5 * self.fvib

    def logConfigVol(self, E):
        EV = E-self.energy
        logE = np.where(EV>0., np.log(EV), -np.inf * np.ones_like(EV))
        return self.logConfig + self.ndof * logE

    def logDoS(self, E):
        EV = E-self.energy
        logE = np.where(EV>0., np.log(EV), -np.inf * np.ones_like(EV))
        return self.logConfig + (self.ndof-1) * logE - log(self.ndof)

    def configVol(self, E):
        return np.log(self.logConfigVol(E))

    def DoS(self, E):
        return np.exp(self.logDoS(E))

import seaborn as sns

from pele.systems import LJCluster
from pele.utils.rotations import random_q, q2mx, vec_random
from nestedbasinsampling.constraints import HardShellConstraint, CombinedPotConstraint
from nestedbasinsampling.takestep import random_structure
from nestedbasinsampling.alignment import CompareStructures

import matplotlib.pyplot as plt
from plottingfuncs.plotting import ax3d


natoms = 13
system = LJCluster(natoms)
radius =  float(natoms) ** (1. / 3)
rand_config = lambda : random_structure(natoms, radius)
constraint = HardShellConstraint(radius)
pot = system.get_potential()

coords = np.array([-0.54322426, -0.38203861,  0.52203412,  1.26544146,  0.4464261 ,
       -0.32877187, -0.1064755 , -0.7742898 , -0.4522979 ,  0.40450016,
        0.09264135, -0.98264503,  1.28655638, -0.20185368,  0.60569126,
        0.283552  ,  1.02068545, -0.33608509,  0.36110853,  0.03219396,
        0.09663129, -0.30217412,  0.7273169 ,  0.59385779,  0.82869275,
        0.83867761,  0.64556041,  1.02439098, -0.66292951, -0.40059499,
       -0.56433927,  0.26624146, -0.41242867,  0.31771689, -0.02825336,
        1.1759076 ,  0.43866499, -0.95629744,  0.52934806])
energy = pot.getEnergy(coords)

m = Minimum(energy, coords)

self = MinimumBasin(m, system)

def random_rotation():
    return q2mx(random_q())

def rand_orientation(natoms, radius):
    return random_structure(natoms, radius), random_rotation()

logtotalvol = natoms * log(4*np.pi/3 * radius**3)

hull = ConvexHull(coords.reshape(-1,3))
convex = coords.reshape(-1,3)[hull.vertices]
meanconvex = convex.mean(0)
convex -= meanconvex

convex_r = np.linalg.norm(convex, axis=1)

max_disp = max(radius - convex_r)
vol = (4*np.pi/3 * max_disp**3) * 4 * np.pi**2

max_disp *= 2
availvol = (4*np.pi/3 * max_disp**3) * 4 * np.pi**2

pos = coords.reshape(-1,3) - meanconvex
constraints = [constraint((pos + random_structure(1, max_disp)).ravel())
               for i in xrange(10000)]

N = np.arange(1, len(constraints)+1, dtype=float)
frac = np.cumsum(constraints, dtype=float)/N

fstd = np.sqrt((frac - frac**2)/N)

plt.plot(frac);plt.plot(frac+fstd);plt.plot(frac-fstd)

print frac[-1], fstd[-1], vol/availvol, (vol/availvol)**0.5


coords = np.array([vec_random() for i in xrange(200)])

pos = coords.reshape(-1,3)
scale = np.array([1,1.1,1.2])
pos *= scale

vol = (4*np.pi/3 * scale.prod() )

r = radius-1.

max_disp = r
mcvol = (4*np.pi/3 * max_disp**3)

constraints = [constraint((pos + random_structure(1, max_disp)).ravel())
               for i in xrange(10000)]

N = np.arange(1, len(constraints)+1, dtype=float)
frac = np.cumsum(constraints, dtype=float)/N

fstd = np.sqrt((frac - frac**2)/N)

f = vol/mcvol
plt.plot((frac - f)/fstd)

print frac[-1], fstd[-1], (frac[-1] - f)/fstd[-1]



max_disp =


fig, ax = ax3d()
ax.scatter(*coords.T)