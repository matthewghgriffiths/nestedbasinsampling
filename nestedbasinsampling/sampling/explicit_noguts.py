
from math import sqrt
from collections import deque
import numpy as np
from numpy.random import uniform

from nestedbasinsampling.structure.constraints import BaseConstraint
from nestedbasinsampling.sampling.galilean import BaseSampler
from nestedbasinsampling.sampling.takestep import random_step
from nestedbasinsampling.utils import Result


class ExplicitNoGUTSSampler(BaseSampler):
    """
    """
    def __init__(
        self, pot, genstep=random_step, constraint=None,
        nsteps=10, nadapt=10, stepsize=0.1, acc_ratio=0.1,
        testinitial=True, fixConstraint=False, maxreject=1000, debug=False):

        self.pot = pot
        self.genstep = random_step
        self.constraint = BaseConstraint() if constraint is None else constraint


    def take_step(self, coords, p, Ecut, epsilon=1.0):
        new_coords = coords + p * epsilon #* uniform()
        new_p = p.copy()
        #print 's0', np.linalg.norm(p*epsilon), np.linalg.norm(p)
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

    def build_tree(self, coords, p, E, G, Ecut, v, j, epsilon):
        """The main recursion."""
        if (j == 0):
            # Base case: Take a single leapfrog step in the direction v.
            coords_prime, p_prime, EGp, (Eaccept, Caccept, reflect) = \
                self.take_step(coords, p[v<0], Ecut, v * epsilon)
            trajectory = deque([EGp + (coords_prime,) + (p_prime[::v],)])
            # p[v<0] selects the momentum in the right direction
            # p_prime[::v] ensures that forward momentum is first
            valid = True
        else:
            # Recursion: Implicitly build the height j-1 left and right subtrees.
            trajectory, valid = self.build_tree(
                coords, p, E, G, Ecut, v, j - 1, epsilon)
            # No need to keep going if the stopping criteria were met in
            # the first subtree.
            E_minus, G_minus, coords_minus, p_minus = trajectory[0]
            E_plus, G_plus, coords_plus, p_plus = trajectory[-1]
            if valid:
                if (v == -1):
                    trajectory_minus, valid_p = self.build_tree(
                        coords_minus, p_minus, E_minus, G_minus, Ecut,
                        v, j - 1, epsilon)
                    if valid_p:
                        trajectory_minus.extend(trajectory)
                        trajectory = trajectory_minus
                        E_minus, G_minus, coords_minus, p_minus = trajectory[0]
                    else:
                        rejected = trajectory_minus
                else:
                    trajectory_plus, valid_p = self.build_tree(
                        coords_plus, p_plus, E_plus, G_plus, Ecut,
                        v, j - 1, epsilon)
                    if valid_p:
                        trajectory.extend(trajectory_plus)
                        E_plus, G_plus, coords_plus, p_plus = trajectory[-1]
                    else:
                        rejected = trajectory_plus
                # Update the stopping criterion.
                valid = int(valid_p and
                            self.stop_criterion(coords_minus, coords_plus,
                                                p_minus, p_plus))

        return trajectory, valid

    def nuts_step(self, coords, Ecut, epsilon=1.,
                  energy=None, grad=None, p0=None):
        """
        """
        p0 = self.genstep(coords) if p0 is None else p0

        if energy is None or grad is None:
            energy, grad = self.pot.getEnergyGradient(coords)

        trajectory = deque([(energy, grad, coords, (p0, p0), -1)])
        E_minus, G_minus, coords_minus, p_minus, _ = trajectory[0]
        E_plus, G_plus, coords_plus, p_plus, _ = trajectory[-1]

        j = 0  # initial height j = 0
        n = 0  # Initially the only valid point is the initial point.
        nreflect = 0 # number of reflections
        valid = True  # Main loop: will keep going until accept == False
        while valid:
            # Choose a direction. -1 = backwards, 1 = forwards.
            v = int(2 * (np.random.uniform() < 0.5) - 1)
            # Double the size of the tree.
            if (v == -1):
                E_minus, G_minus, coords_minus, p_minus, _ = trajectory[0]
                #print 'ns m', np.linalg.norm(p_minus)
                trajectory_p, valid_p = self.build_tree(
                    coords_minus, p_minus, E_minus, G_minus, Ecut,
                    v, j, epsilon)
                if valid_p:
                    trajectory_p = deque(
                            point + (j,) for point in trajectory_p)
                    trajectory_p.extend(trajectory)
                    trajectory = trajectory_p
                    #E_minus, G_minus, coords_minus, p_minus, _ = trajectory[0]
            else:
                E_plus, G_plus, coords_plus, p_plus, _ = trajectory[-1]
                #print 'ns p', np.linalg.norm(p_plus)
                trajectory_p, valid_p = self.build_tree(
                    coords_plus, p_plus, E_plus, G_plus, Ecut,
                    v, j, epsilon)
                if valid_p:
                    trajectory.extend(
                            point + (j,) for point in trajectory_p)
                    #E_plus, G_plus, coords_plus, p_plus, _ = trajectory[-1]

            valid = valid_p and self.stop_criterion(
                coords_minus, coords_plus, p_minus, p_plus)
            # Increment depth.
            j += 1

        return trajectory, v, None if valid_p else trajectory_p

    @staticmethod
    def stop_criterion(coords_minus, coords_plus, p_minus, p_plus):
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
        delta = coords_plus.ravel() - coords_minus.ravel()
        return ((np.dot(delta, p_minus[1].T) >= 0) & \
                (np.dot(delta, p_plus[0].T) >= 0))



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
            self.M = np.array(M, float)

        def getEnergy(self, x):
            return  0.5*self.M.dot(x).dot(x)

        def getGradient(self, x):
            return - self.M.dot(x)

        def getEnergyGradient(self, x):
            G = self.M.dot(x)
            return 0.5*G.dot(x), -G

    M = np.diag([1.,20.])
    a, b = (0.5*M.diagonal())**-0.5
    theta = np.linspace(0, 2*np.pi, 256)

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

    nuts = ExplicitNoGUTSSampler(pot)
    self = nuts
    Ecut = 1.
    epsilon = 0.2
    energy = None
    grad = None

    def plot_trajectory(
            trajectory, linec= 'k', gradc='b', acceptc='g', rejectc='r'):
        Es, Gs, path, ps = map(np.array, zip(*trajectory))[:4]
        X, Y = path.T
        dX, dY = Gs.T
        U, V = ps.T * epsilon
        cutoff = (Es > Ecut)

        plt.plot(X, Y, c=linec)
        plt.scatter(X[cutoff], Y[cutoff], c=rejectc, marker='x')
        plt.scatter(X[~ cutoff], Y[~ cutoff], c=acceptc, marker='o')
        plt.quiver(X[cutoff], Y[cutoff], dX[cutoff], dY[cutoff],
                   angles='xy', scale_units='xy',color=gradc)
        plt.quiver(X[0], Y[0], *(-0.5*ps[0,1]*epsilon), color=linec,
                   angles='xy', scale_units='xy', scale=1.,
                   headlength=3, headwidth=3, width=0.008)
        plt.quiver(X[-1], Y[-1], *(0.5*ps[-1,0]*epsilon), color=linec,
                   angles='xy', scale_units='xy', scale=1.,
                   headlength=3, headwidth=3, width=0.008)

    def integrate(coords, p, Ecut, epsilon=0.01):
        coords = np.array(coords)
        p = np.array(p)
        while True:
            new_coords, (new_p), newEG, _ = \
                nuts.take_step(coords, p, Ecut, epsilon=epsilon)
            yield newEG + (new_coords, new_p)
            coords = new_coords
            p = new_p[0]

    coords = random_coords(Ecut)
    trajectory, v, rejected = nuts.nuts_step(coords, Ecut, epsilon=epsilon)

    Es, Gs, path, ps, ds = map(np.array, zip(*trajectory))

    integrator = integrate(
        trajectory[0][2], -trajectory[0][3][1], Ecut, epsilon=epsilon)
    _, back = zip(*zip(tqdm(xrange(len(trajectory))), integrator))
    integrator = integrate(
        trajectory[-1][2], trajectory[-1][3][0], Ecut, epsilon=epsilon)
    _, forward = zip(*zip(tqdm(xrange(len(trajectory))), integrator))


    plt.xlim(-a-2*epsilon, a+2*epsilon)
    plt.ylim(-b-2*epsilon, b+2*epsilon)
    plt.plot(a*np.cos(theta)*Ecut**0.5, b*np.sin(theta)*Ecut**0.5)
    plt.gca().set_aspect('equal')
    if rejected:
        if v < 0:
            plot_trajectory(
                back, linec='r', acceptc=[[1.,0.,0.,0.5]], rejectc=[[1.,0.,0.,0.5]])
        else:
            plot_trajectory(
                forward, linec='r', acceptc=[[1.,0.,0.,0.5]], rejectc=[[1.,0.,0.,0.5]])
    plot_trajectory(trajectory)
    print ds


    plt.xlim(-a-2*epsilon, a+2*epsilon)
    plt.ylim(-b-2*epsilon, b+2*epsilon)
    plt.plot(a*np.cos(theta)*Ecut**0.5, b*np.sin(theta)*Ecut**0.5, c='k')
    plt.gca().set_aspect('equal')
    for i in xrange(0):
        coords = random_coords(Ecut) * 0.
        trajectory, rejected, rejected2 = nuts.nuts_step(coords, Ecut, epsilon=epsilon)
        plot_trajectory(trajectory)
        if rejected:
            plot_trajectory(rejected, linec='r', gradc=[[0.,0.,0.,0.]],
                            acceptc=[[1.,0.,0.,1.]], rejectc=[[1.,0.,0.,1.]])
            Es, Gs, path, ps = map(np.array, zip(*rejected))
        if rejected2:
            plot_trajectory(rejected2, linec='r', gradc=[[0.,0.,0.,0.]],
                            acceptc=[[1.,0.,0.,0.5]], rejectc=[[1.,0.,0.,0.5]])
            Es, Gs, path, ps = map(np.array, zip(*rejected2))
