

from scipy.special import psi, polygamma

def determine_stepsize(coords, Ecut, sampler, stepsize, target=0.4,
                       nsteps=200, nadapt=30):

    f = (1 - target)/target
    kalman = LinearKalmanFilter(z=np.log(stepsize) - np.log(f), R=1.)
    stepsizes = [stepsize]
    for i in xrange(nadapt):
        p = sampler.genstep(coords)
        last = stepsize
        Es, Gs, path, ps, accepts = sampler.integrate(
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

        print "{:6.3g}+/-{:6.2g}, {:6.3g}+/-{:6.2g}, {:3d}, {:3d}, {:6.2g}, {:6.2g}".format(
            a0, v**0.5, a1, v1**0.5, naccept, nreject, last, stepsize)
    return stepsizes
