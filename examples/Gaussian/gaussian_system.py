
import logging
import numpy as np
from pele.potentials import BasePotential
from nestedbasinsampling import (
    NBS_system, vector_random_uniform_hypersphere, LOG_CONFIG)

class MyPot(BasePotential):

    def __init__(self, M):
        self.M = M

    def getEnergy(self, coords):
        return 0.5*np.dot(coords, coords*self.M)

    def getGradient(self, coords):
        return self.M*coords

    def getEnergyGradient(self, coords):
        G = self.getGradient(coords)
        E = 0.5 * np.dot(coords, G)
        return E, G

M = np.array([
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
    680.97359437,  740.68419553,  793.64807121])


pot = MyPot(M)
n = len(M)
u = M
p = u > 1e-5
k = p.sum()
v = np.eye(n)
up = u[p]
vp = v[:,p]
up2 = (2./up)**0.5

def random_coords(E):
    x = vector_random_uniform_hypersphere(k) * E**0.5
    return vp.dot(up2 * x)

Ecut = 1000.
stepsize = 0.1
random_config = lambda : random_coords(Ecut)
system_kws = dict(
    pot=pot, random_configuration=random_config, stepsize=stepsize,
    sampler_kws=dict(max_depth=8, nsteps=30), nopt_kws=dict(iprint=10))
get_system = lambda : NBS_system(**system_kws)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    system = get_system()
    pot = system.pot
    nuts = system.sampler
    plt.ion()


    Ecut=10000.
    k = 87
    epsilon = 0.02
    nsamples = 1000
    a = np.arange(nsamples) + 1
    b = nsamples + 1 - a
    l = np.log(a) - np.log(a+b)
    l2 = l + np.log(a+1) - np.log(a+b+1)
    lstd = np.log1p(np.sqrt(np.exp(l2 - 2 * l) - 1))

    coords = random_coords(Ecut)
    nuts_results = []
    for i in tqdm(xrange(nsamples)):
        nuts_results.append(nuts(Ecut, coords, stepsize=epsilon))

    nEs = np.array([r.energies for r in nuts_results])
    nEs.sort(0)
    for i in xrange(4, nEs.shape[1],5):
        Es = nEs.T[i]
        plt.plot(
            Es**(0.5*k), ((l - 0.5*k*(np.log(Es)-np.log(Ecut)))/lstd), label=i)
    plt.legend()
    plt.show()
