# -*- coding: utf-8 -*-
import logging

import numpy as np
from numpy.random import uniform
from scipy.special import psi, polygamma

from pele.optimize import lbfgs_cpp
from pele.mindist import findrotation

from ..structure import BaseConstraint
from ..utils import Result, LinearKalmanFilter, SamplingError
from .galilean import BaseSampler
from .takestep import random_step

try:
    from fortran.noguts import noguts
    has_fortran = True
except ImportError:
    noguts = None
    has_fortran = False

logger = logging.getLogger('NBS.NoGUTS')


class NoGUTSSampler(BaseSampler):
    """This class suffers from PNS syndrome
    """
    noguts = noguts
    has_fortran = has_fortran

    def __init__(
            self, pot, genstep=None, constraint=None,
            remove_linear_momentum=False, nsteps=10, stepsize=0.1,
            max_depth=None, remove_angular_momentum=False,
            remove_initial_linear_momentum=False, min_rotation=False,
            remove_initial_angular_momentum=False, fix_centroid=False,
            seed=None, rand_state=None, testinitial=True, fixConstraint=False,
            maxreject=1000, debug=False):

        self.pot = pot
        if genstep is not None:
            self.genstep = genstep
        self.constraint = \
            BaseConstraint() if constraint is None else constraint
        self.stepsize = stepsize
        self.nsteps = nsteps
        self.testinitial = testinitial
        self.max_depth = max_depth
        self.debug = debug
        self.remove_linear_momentum = remove_linear_momentum
        self.remove_angular_momentum = remove_angular_momentum
        self.remove_initial_linear_momentum = remove_initial_linear_momentum
        self.remove_initial_angular_momentum = remove_initial_angular_momentum
        self.min_rotation = min_rotation
        self.fix_centroid = fix_centroid

        self._set_fortran(self.has_fortran)
        self.set_random_state(seed, rand_state)

        if callable(fixConstraint):
            self.fixConstraint = fixConstraint
        elif fixConstraint:
            self.fixConstraint = self._fixConstraint

    @property
    def seed(self):
        if has_fortran:
            return self.noguts.seed.item()
        else:
            return None

    @seed.setter
    def seed(self, seed):
        if seed is None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)
        if has_fortran:
            self.noguts.seed = seed
        np.random.seed(seed)

    def get_random_state(self):
        np_state = np.random.get_state()
        if has_fortran:
            return self.noguts.seed.item(), np_state
        else:
            return None, np_state

    def set_random_state(self, seed, state=None):
        self.seed = seed
        if state is not None:
            np.random.set_state(state)

    def _set_fortran(self, has_fortran):
        if has_fortran:
            self.build_tree = self._f_build_tree
            self.stop_criterion = self._f_stop_criterion
            self.seed = self.noguts.seed
            self.noguts.remove_linear_momentum = self.remove_linear_momentum
            self.noguts.remove_angular_momentum = self.remove_angular_momentum
        else:
            self.build_tree = self._py_build_tree
            self.stop_criterion = self._py_stop_criterion

    def genstep(self, coords):
        p = random_step(coords)

        if self.remove_initial_linear_momentum:
            p3d = p.reshape(-1, 3)
            p3d -= p3d.mean(0)[None, :]

        if self.remove_initial_angular_momentum:
            p[:] = self.zero_angular_momentum(coords, p)

        if (self.remove_initial_linear_momentum or
            self.remove_initial_angular_momentum):
            p /= np.linalg.norm(p)

        return p

    def zero_angular_momentum(self, coords, p):
        coords = np.asanyarray(coords)
        p3d = p.reshape(-1, 3)
        X = coords.reshape(-1, 3)
        X0 = X - X.mean(0)[None,:]
        X2, XY, ZX, Y2, YZ, Z2 = (X0[:, [0, 0, 0, 1, 1, 2]] *
                                  X0[:, [0, 1, 2, 1, 2, 2]]).sum(0)
        I = np.array([[Y2 + Z2, -XY, -ZX],
                      [-XY, X2 + Z2, -YZ],
                      [-ZX, -YZ, X2 + Y2]])
        L = np.cross(X0, p3d).sum(0)
        omega = np.linalg.inv(I).dot(L)
        p0 = (p3d - np.cross(omega, X0)).reshape(coords.shape)
        return p0

    def take_step(self, coords, p, Ecut, epsilon=1.0):
        # TODO create interface to FORTRAN takestep subroutine
        new_coords = coords + p * epsilon #* uniform()
        new_p = p.copy()

        Econ, Gcon = self.constraint.getEnergyGradient(new_coords)
        Enew, Gnew = self.pot.getEnergyGradient(new_coords)

        Caccept = Econ <= 0
        Eaccept = Enew <= Ecut
        reflect = not (Eaccept and Caccept)
        if reflect:
            if Eaccept and not Caccept:
                N = Gcon
            elif Caccept and not Eaccept:
                N = Gnew
            else:
                n1 = Gnew/np.linalg.norm(Gnew)
                n2 = Gcon/np.linalg.norm(Gcon)
                N = n1 + n2
            new_p -= 2. * N * N.ravel().dot(p.ravel()) / N.ravel().dot(N.ravel())

        return new_coords, (new_p, p), (Enew, Gnew), (Eaccept, Caccept, reflect)

    def integrator(self, coords, p, Ecut, epsilon=0.01):
        coords = np.array(coords)
        p = np.array(p)
        while True:
            coords, (p, _), (newE, newG), (_, _, reject) = self.take_step(
                coords, p, Ecut, epsilon=epsilon)
            yield (newE, newG, coords, p, not reject)

    def integrate(self, coords, p, Ecut, epsilon=0.01, nsteps=16):
        integrator = self.integrator(coords, p, Ecut, epsilon=epsilon)
        Es, Gs, path, ps, accepts = map(
            np.array, zip(*(
                point for _, point in zip(xrange(nsteps), integrator))))
        return Es, Gs, path, ps, accepts

    def _f_build_tree(self, X, p, EG, Ecut, v, j, epsilon):
        X = np.asanyarray(X)
        p_f, p_b = np.asanyarray(p[0]), np.asanyarray(p[1])
        X_pls = X.copy()
        X_min = X.copy()
        p_pls_f, p_pls_b = p_f.copy(), p_b.copy()
        p_min_f, p_min_b = p_f.copy(), p_b.copy()

        X_n, E_n, G_n, naccept, nreject, tot_accept, tot_reject, valid = \
            self.noguts.build_tree(
                X_pls, p_pls_f, p_pls_b, X_min, p_min_f, p_min_b, v, j, Ecut,
                epsilon, self.pot.getEnergyGradient, self.constraint.getEnergyGradient)

        EG_n = E_n, G_n
        p_m =  p_min_f, p_min_b
        p_p = p_pls_f, p_pls_b
        return (X_min, p_m, None, X_pls, p_p, None, X_n, None, EG_n,
                naccept, nreject, tot_accept, tot_reject, valid)

    def _py_build_tree(self, X, p, EG, Ecut, v, j, epsilon):
        """The main recursion."""
        if (j == 0):
            # Base case: Take a single leapfrog step in the direction v.
            X_n, p_n, EG_n, (Eaccept, Caccept, reflect) = \
                self.take_step(X, p[v < 0], Ecut, v * epsilon)

            p_n = p_n[::v]  # reverse momentum pair if going backwards
            # Set the return values---minus=plus for all things here, since the
            # "tree" is of depth 0.
            X_m = X_n[:]
            X_p = X_n[:]
            p_m = p_n
            p_p = p_n
            EG_m = EG_n
            EG_p = EG_n

            accept = Eaccept and Caccept
            # Count the number of reflections
            tot_reject = nreject = int(reflect)
            # Is the new point in the slice?
            tot_accept = naccept = int(accept)
            valid = True
        else:
            # Recursion: Implicitly build the height j-1 left and right subtrees.
            X_m, p_m, EG_m, X_p, p_p, EG_p, X_n, p_n, EG_n, \
            naccept, nreject, tot_accept, tot_reject, valid = self._py_build_tree(
                X, p, EG, Ecut, v, j - 1, epsilon)
            # No need to keep going if the stopping criteria were met in
            # the first subtree.
            if valid:
                if (v == -1):
                    X_m, p_m, EG_m, _, _, _, X_n2, p_n2, EG_n2, \
                    naccept2, nreject2, tot_accept2, tot_reject2, valid2 \
                        = self._py_build_tree(
                            X_m, p_m, EG_m, Ecut, v, j - 1, epsilon)
                else:
                    _, _, _, X_p, p_p, EG_p, X_n2, p_n2, EG_n2, \
                    naccept2, nreject2, tot_accept2, tot_reject2, valid2 \
                        = self._py_build_tree(
                            X_p, p_p, EG_p, Ecut, v, j - 1, epsilon)
                # Choose which subtree to propagate a sample up from.
                if valid2:
                    if naccept2:
                        prob_new_sample =  (float(naccept2)
                            / max(float(int(naccept) + int(naccept2)), 1.))
                        take_new_sample = np.random.uniform() < prob_new_sample
                        if take_new_sample:
                            X_n = X_n2
                            EG_n = EG_n2

                        # Update the number of valid points.
                        naccept += naccept2
                    # Update the acceptance probability statistics.
                    nreject += nreject2

                    # Update the stopping criterion.
                    valid = self.stop_criterion(X_m, X_p, p_m, p_p)
                else:
                    valid = False
                tot_accept += tot_accept2
                tot_reject += tot_reject2

        return (X_m, p_m, EG_m, X_p, p_p, EG_p, X_n, p_n, EG_n,
                naccept, nreject, tot_accept, tot_reject, valid)

    def nuts_step(self, coords, Ecut, epsilon=1., energy=None, grad=None,
                  _depth=0):
        """
        """

        p0 = self.genstep(coords)

        if energy is None or grad is None:
            energy, grad = self.pot.getEnergyGradient(coords)
        EG = energy, grad

        X_m = coords[:]
        X_p = coords[:]
        p_m = (p0, p0)
        p_p = (p0, p0)
        EG_m = EG
        EG_p = EG

        res = Result()
        res.start = coords
        res.coords = coords
        res.p = p_m
        res.energy, res.grad = EG
        res.Ecut = Ecut
        res.success = False
        res.naccept = 0
        res.nreject = 0
        res.epsilon = epsilon
        res.depth = 0
        res.NUTS = None

        j = 0  # initial height j = 0
        tot_accept = naccept = 0
        tot_reject = nreject = 0
        valid = True  # Main loop: will keep going until accept == False

        while valid:
            # Choose a direction. -1 = backwards, 1 = forwards.
            v = int(2 * (np.random.uniform() < 0.5) - 1)

            # Double the size of the tree.
            if (v == -1):
                X_m, p_m, EG_m, _, _, _, X_n, p_n, EG_n, \
                naccept2, nreject2, tot_accept2, tot_reject2, valid2 \
                    = self.build_tree(X_m, p_m, EG_m, Ecut, v, j, epsilon)
            else:
                _, _, _, X_p, p_p, EG_p, X_n, p_n, EG_n, \
                naccept2, nreject2, tot_accept2, tot_reject2, valid2 \
                    = self.build_tree(X_p, p_p, EG_p, Ecut, v, j, epsilon)

            # Use Metropolis-Hastings to decide whether or not to move to a
            # point from the half-tree we just generated.
            tot_accept += tot_accept2
            tot_reject += tot_reject2
            if valid2:
                _tmp = float(naccept2) / max(float(naccept), 1.)
                naccept += naccept2 # not sure if this should be here
                nreject += nreject2
                if _tmp > 1. or (uniform() < _tmp):
                    res.coords = X_n[:]
                    res.p = p_n
                    res.energy, res.grad = EG_n
                    res.success = True
                    res.NUTS = True
                res.naccept = naccept
                res.nreject = nreject

                # Update number of valid points we've seen.
                # Decide if it's time to stop.
                if not self.stop_criterion(X_m, X_p, p_m, p_p):
                    res.success = True
                    res.NUTS = True
                    break
                else:
                    # Increment depth.
                    j += 1
                    # break out of loop if j is at max depth
                    if j == self.max_depth:
                        res.success = True
                        res.NUTS = False
                        break
            else:
                # Next tree proposal not valid, rejecting and ending loop
                res.success = True
                res.NUTS = False
                break

        if res.naccept == 0:
            _depth += 1
            if _depth < 50:
                res = self.nuts_step(
                    coords, Ecut, epsilon=epsilon, energy=energy, grad=grad,
                    _depth=_depth)
                res.depth = _depth
            else:
                raise SamplingError((
                    "Max recursion of nuts_step, try decreasing stepsize "
                    "current stepsize={:6.3g}, current energy={:10.5g}".format(
                        epsilon, Ecut)))

        res.tot_accept = tot_accept
        res.tot_reject = tot_reject
        res.nfev = res.tot_accept + res.tot_reject
        res.nstep = res.naccept + res.nreject

        return res

    def __call__(self, Ecut, coords, stepsize=None, nsteps=None):
        """ Samples a new point within the energy contour defined by Ecut
        starting from coords.
        """
        stepsize = self.stepsize if stepsize is None else stepsize
        nsteps = self.nsteps if nsteps is None else nsteps

        res = Result()
        res.naccept = 0
        res.nreject = 0
        res.tot_accept = 0
        res.tot_reject = 0
        res.nfev = 0
        res.stepsize = stepsize
        res.niter = nsteps
        res.energies = []

        res.coords = np.array(coords)

        if self.testinitial:
            Econ = self.constraint.getEnergy(res.coords)
            if Econ > 0:
                try:
                    res.coords = self.fixConstraint(res.coords)
                except NotImplementedError:
                    raise SamplingError(
                        "Starting configuration doesn't satisfy constraint",
                        Estart=Ecut, Econstraint=Econ)

        E, G = None, None
        i = 0
        while i < nsteps:
            newres = self.nuts_step(
                res.coords, Ecut, epsilon=stepsize, energy=E, grad=G)
            if newres.naccept:
                res.coords = newres.coords
                E = newres.energy
                G = newres.grad
                res.naccept += newres.naccept
                res.nreject += newres.nreject
                res.tot_accept += newres.tot_accept
                res.tot_reject += newres.tot_reject
                res.nfev += newres.nfev

                res.energies.append(E)
                i += 1

        res.energy = newres.energy
        res.grad = newres.grad
        res.nsteps = res.nfev

        return res

    def _f_stop_criterion(self, X_m, X_p, p_m, p_p):
        """ Compute the stop condition in the main loop
        dot(dtheta, rminus) >= 0 & dot(dtheta, rplus >= 0)

        INPUTS
        ------
        thetaminus, thetaplus: ndarray[float, ndim=1]
            under and above position
        rminus, rplus: ndarray[float, ndim=1]
            under and above momentum

        OUTPUTS
        -------
        criterion: bool
            return if the condition is valid
        """
        return self.noguts.stop_criterion(X_p, p_p[0], X_m, p_m[1])

    def _py_stop_criterion(self, X_m, X_p, p_m, p_p):
        """ Compute the stop condition in the main loop
        dot(dtheta, rminus) >= 0 & dot(dtheta, rplus >= 0)

        INPUTS
        ------
        thetaminus, thetaplus: ndarray[float, ndim=1]
            under and above position
        rminus, rplus: ndarray[float, ndim=1]
            under and above momentum

        OUTPUTS
        -------
        criterion: bool
            return if the condition is valid
        """
        if self.fix_centroid:
            X_p = (X_p.reshape(-1,3) - X_p.reshape(-1,3).mean(0)[None,:])
            X_m = (X_m.reshape(-1,3) - X_m.reshape(-1,3).mean(0)[None,:])

        if self.min_rotation:
            _, M = findrotation(X_p, X_m)
            X_m = X_m.reshape(-1,3).dot(M.T)
            p_m = (p_m[0], p_m[1].reshape(-1,3).dot(M.T).ravel())

        delta = X_p.ravel() - X_m.ravel()
        return ((np.dot(delta, p_m[1].T) >= 0) and
                (np.dot(delta, p_p[0].T) >= 0))

    def _fixConstraint(self, coords, **kwargs):
        coords = np.array(coords) # Make a copy
        conpot = self.constraint.getGlobalPotential(coords)
        disp = np.zeros(3)
        res = lbfgs_cpp(disp, conpot, **kwargs)
        pos = coords.reshape(-1,3)
        pos += res.coords[None,:]
        return coords

    def fixConstraint(self, *arg, **kwargs):
        raise NotImplementedError

    def determine_stepsize(self, coords, Ecut, stepsize=None, target_acc=0.4,
                           nsteps=200, nadapt=30):
        stepsize = self.stepsize if stepsize is None else stepsize

        logger.info(
            ("determining stepsize, initial stepsize={:6.2g}, "
             "target_acc={:6.2g}, nsteps={:3d}, nadapt={:2d}").format(
                stepsize, target_acc, nsteps, nadapt))
        f = (1 - target_acc)/target_acc
        kalman = LinearKalmanFilter(z=np.log(stepsize) - np.log(f), R=1.)
        stepsizes = [stepsize]
        for i in xrange(nadapt):
            p = self.genstep(coords)
            last = stepsize
            Es, Gs, path, ps, accepts = self.integrate(
                coords, p, Ecut, epsilon=stepsize, nsteps=nsteps)
            naccept = accepts.sum()
            if naccept:
                coords = path[accepts][-1]
            nreject = nsteps - naccept
            acc = float(naccept)/nsteps
            n, m = naccept/10. + 0.5, nreject/10. + 0.5
            m = psi(n) - psi(m)
            v = polygamma(1, n) + polygamma(1, m)
            a0 = m + np.log(stepsize)
            a1, v1 = kalman(a0, v)
            stepsize = np.exp(a1)*f
            stepsizes.append(stepsize)

            logger.debug(
                "naccept={:3d}, nreject={:3d}, stepsize={:6.2g}".format(
                    naccept, nreject, stepsize))

        logger.info("final stepsize={:6.2g}, acc={:6.2g}".format(
            stepsize, float(naccept)/float(naccept + nreject)))
        return stepsizes


class NoGUTSWalker(NoGUTSSampler):
    """Interface to work with nested_sampling/sens
    """
    def __call__(self, x0, stepsize, Emax, energy, seed=None):
        self.seed = seed
        super(NoGUTSWalker, self).__call__(Emax, x0, stepsize=stepsize)


if __name__ == "__main__":
    import numpy as np
    from scipy.special import betaln
    from itertools import izip
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    from pele.potentials import BasePotential
    from nestedbasinsampling.storage import Database, Minimum, Run, Replica
    from nestedbasinsampling.sampling.takestep import (
    vec_random_ndim, vector_random_uniform_hypersphere)
    from nestedbasinsampling.structure.constraints import (
    BaseConstraint, HardShellConstraint)
    from nestedbasinsampling.sampling.stats import (
    CDF, AndersonDarling, AgglomerativeCDFClustering)
    from nestedbasinsampling.optimize.nestedbasinoptimization import (
    NestedBasinOptimizer)
    from nestedbasinsampling.utils import Result
    from tqdm import tqdm

    plt.ion()

    class MyPot(BasePotential):
        def __init__(self, M):
            self.Ecalls = 0
            self.Gcalls = 0
            self.M = np.array(M, float)

        def getEnergy(self, x):
            self.Ecalls += 1
            return  0.5*self.M.dot(x).dot(x)

        def getGradient(self, x):
            self.Gcalls += 1
            return self.M.dot(x)

        def getEnergyGradient(self, x):
            self.Ecalls += 1
            self.Gcalls += 1
            G = self.M.dot(x)
            return 0.5*G.dot(x), G

    M = np.diag(np.array([  #0., 0., 0.,
         11.63777605,   19.75825574,   22.2571117 ,   24.41295908,
         26.32612811,   31.30715704,   35.27360319,   37.34413361,
         41.24811749,   42.66902559,   45.00513907,   48.71488414,
         49.89979232,   53.0797042 ,   55.39317634,   56.84512961,
         60.77859882,   60.93608218,   62.49575527,   65.40116213,
         69.59126898,   71.32244177,   71.59182786,   73.51372578,
         81.19666404,   83.07758741,   84.5588217 ,   86.37683242,
         94.65859144,   95.40770789,   95.98119526,  102.45620344,
        102.47916283,  104.40832154,  104.86404787,  112.80895254,
        117.10380584,  123.6500204 ,  124.0540132 ,  132.17808513,
        136.06966301,  136.60709658,  138.73165763,  141.45541009,
        145.23595258,  150.31676718,  150.85458655,  155.15681296,
        155.25203667,  155.87048385,  158.6880457 ,  162.77205271,
        164.92793349,  168.44191483,  171.4869683 ,  186.92271992,
        187.93659725,  199.78966333,  203.05115652,  205.41580397,
        221.54815121,  232.16086835,  233.13187687,  238.45586414,
        242.5562086 ,  252.18391589,  264.91944949,  274.141751  ,
        287.58508273,  291.47971184,  296.03725173,  307.39663841,
        319.38453549,  348.68884953,  360.54506854,  363.87206193,
        381.72011237,  384.1627136 ,  396.94159259,  444.72185599,
        446.48921839,  464.50930109,  485.99776331,  513.57334376,
        680.97359437,  740.68419553,  793.64807121]))

    pot = MyPot(M)
    n = len(M)
    u = M.diagonal()
    p = u > 1e-5
    k = p.sum()
    v = np.eye(n)
    up = u[p]
    vp = v[:,p]
    up2 = (2./up)**0.5

    def random_coords(E):
        x = vector_random_uniform_hypersphere(k) * E**0.5
        return vp.dot(up2 * x)

    m = Minimum(0, np.zeros(len(M)))

    nuts = NoGUTSSampler(pot)
    self = nuts
    Ecut = 1.
    epsilon = 0.02

    nsamples = 1000
    a = np.arange(nsamples) + 1
    b = nsamples + 1 - a
    l = np.log(a) - np.log(a+b)
    l2 = l + np.log(a+1) - np.log(a+b+1)
    lstd = np.log1p(np.sqrt(np.exp(l2 - 2 * l) - 1))
    Es = np.array([pot.getEnergy(random_coords(Ecut)) for i in range(nsamples)])


    coords = random_coords(Ecut)
    results = []
    for i in tqdm(xrange(nsamples)):
        rs = []
        pos = coords
        while len(rs) < 10:
            r = nuts.nuts_step(pos, Ecut, epsilon=0.02)
            if r.naccept:
                rs.append(r)
                pos = rs[-1].coords
        results.append(rs)

    nEs = np.array([[r.energy for r in rs] for rs in results])
    nEs.sort(0)
    for Es in nEs.T[5:]:
        plt.plot(Es**(0.5*k), ((l - 0.5*k*np.log(Es))/lstd))



    pstr = "E_m={:6.5g}, E_p={:6.5g}, E_n={:6.5g}, a={:2d}, r={:2d}, ta={:2d}, tr={:2d}, v={:2d}"

    def print_output(coords, p, v, j, Ecut, epsilon, build_tree):
        X_m, p_m, EG_m, X_p, p_p, EG_p, X_n, p_n, EG_n, \
        nprime, nreflect, tot_accept, tot_reject, valid = build_tree(
            coords, (p, p), None, Ecut, v, j, epsilon)
        print pstr.format(
            pot.getEnergy(X_m), pot.getEnergy(X_p), EG_n[0],
            nprime, nreflect, tot_accept, tot_reject, valid)
        return (X_m, p_m, EG_m, X_p, p_p, EG_p, X_n, p_n, EG_n,
                nprime, nreflect, tot_accept, tot_reject, valid)


    coords = random_coords(Ecut)
    p = nuts.genstep(coords)

    for j in range(10):
        print j
        v = -1
        print_output(coords, p, v, j, Ecut, epsilon, nuts._py_build_tree)
        print_output(coords, p, v, j, Ecut, epsilon, nuts._f_build_tree)
        v = 1
        print_output(coords, p, v, j, Ecut, epsilon, nuts._py_build_tree)
        print_output(coords, p, v, j, Ecut, epsilon, nuts._f_build_tree)


    X_m, p_m, EG_m, X_p, p_p, EG_p, X_n, p_n, EG_n, \
        nprime, nreflect, tot_accept, tot_reject, valid = \
            print_output(coords, p, v, j, Ecut, epsilon, nuts._py_build_tree)
    print nuts.stop_criterion(X_m, X_p, p_m, p_p), noguts.stop_criterion(X_p, p_p[0], X_m, p_m[1])
    X_m, p_m, EG_m, X_p, p_p, EG_p, X_n, p_n, EG_n, \
        nprime, nreflect, tot_accept, tot_reject, valid = \
            print_output(coords, p, v, j, Ecut, epsilon, nuts._f_build_tree)
    print nuts.stop_criterion(X_m, X_p, p_m, p_p), noguts.stop_criterion(X_p, p_p[0], X_m, p_m[1])

if 0:
    results = []
    for i in tqdm(xrange(nsamples)):
        rs = []
        pos = coords
        while len(rs) < 20:
            r = nuts.nuts_step(pos, Ecut, epsilon=0.02)
            if r.naccept:
                rs.append(r)
                pos = rs[-1].coords
        results.append(rs)

    nEs = np.array([[r.energy for r in rs] for rs in results])
    nEs.sort(0)
    for Es in nEs.T[5:]:
        plt.plot(Es**(0.5*k), ((l - 0.5*k*np.log(Es))/lstd))


    from scipy.stats import powerlaw
    dist = powerlaw(0.5*k-1)
    x = np.linspace(0, 1, 2**9+1)
    plt.plot([dist.logpdf(Es).sum() for Es in nEs.T])
    for i, Es in enumerate(nEs.T):
        print i, dist.logpdf(Es).sum()


    mean, sd = dist.stats()

    M = 10000
    coords = random_coords(Ecut)
    results = []
    for _ in tqdm(xrange(nsamples)):
        res = nuts.nuts_step(coords, Ecut, epsilon=epsilon)
        results.append(res)
        coords = res.coords

    Es = np.array([_r.energy for _r in results])
    Ds = Es - mean
    corr = np.fft.ifftshift(
        np.correlate(Es-mean, Es-mean, mode='same'))
    corr /= sd * (M - np.arange(M))

    for Mcut in xrange(M-1):
        if corr[Mcut+1] < 0.05:
            break

    n_effective = 1/(1 + 2 * ((1 - np.arange(1, Mcut)/M) * corr[1:Mcut]).sum())
