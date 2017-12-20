# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc import logsumexp

def calcRunWeights(run):
    """
    """
    ns = run.nlive.astype(float)
    Emax = run.Emax.copy()

    Xs = (ns/(ns+1.)).cumprod()
    n1n2 = ((ns+1.)/(ns+2.)).cumprod()
    X2s = Xs * n1n2
    n2n3 = ((ns+2.)/(ns+3.)).cumprod()
    n3n4 = ((ns+3.)/(ns+4.)).cumprod()

    X = Xs[-1] if Xs.size else 0.
    X2 = X2s[-1] if X2s.size else 0.

    dF = Xs / ns

    d2F2 = (n1n2/(ns+1.))
    dF2  = 2 * Xs / ns

    XdF = X * n1n2 / ns
    X2dF = X2 * n2n3 / ns

    Xd2F2 = n2n3/(ns+2.)
    XdF2 = 2 * n1n2/(ns+1.) * X

    X2d2F2 = n3n4/(ns+3.)
    X2dF2 = 2 * n2n3/(ns+2.) * X2

    dvarf = ( (2*(ns+1)/(ns+2) *
              ((ns+2)**2/(ns+1)/(ns+3)).cumprod()).mean() -
             (2*ns/(ns+1) * ((ns+1)**2/ns/(ns+2)).cumprod()).mean())
    dvarX = (np.exp((2*np.log(ns+2)-np.log(ns+1)-np.log(ns+3)).sum()) -
             np.exp((2*np.log(ns+1)-np.log(ns+0)-np.log(ns+2)).sum()) )

    if dvarf < -1. or dvarf >= 0:
        dvarf = -1.
    elif np.isnan(dvarf):
        dvarf = -1.
    if dvarX < -1. or dvarX >= 0:
        dvarX = -1.

    weights = dict(Emax=Emax, dF=dF, d2F2=d2F2, dF2=dF2,
                   XdF=XdF, X2dF=X2dF,
                   XdF2=XdF2, Xd2F2=Xd2F2,
                   X2d2F2=X2d2F2, X2dF2=X2dF2,
                   dvarf=dvarf, dvarX=dvarX)

    return weights

def calcRunAverageValue(weights, func, std=True):
    """
    """
    Emax = weights['Emax']
    dF = weights['dF']
    XdF = weights['XdF']
    X2dF = weights['X2dF']
    d2F2 = weights['d2F2']
    dF2 = weights['dF2']
    Xd2F2 = weights['Xd2F2']
    XdF2 = weights['XdF2']
    X2d2F2 = weights['X2d2F2']
    X2dF2 = weights['X2dF2']

    fj = np.atleast_2d(func(Emax))
    f = fj.dot(dF)
    Xf = fj.dot(XdF)
    if std:
        X2f = fj.dot(X2dF)
        f2 = np.einsum("ij,j,ij->i", fj, d2F2, (fj*dF2).cumsum(1))
        Xf2 = np.einsum("ij,j,ij->i", fj, Xd2F2, (fj*XdF2).cumsum(1))
        X2f2 = np.einsum("ij,j,ij->i", fj, X2d2F2, (fj*X2dF2).cumsum(1))
        return f, Xf, X2f, f2, Xf2, X2f2
    else:
        return f, Xf


def logtrapz(logy, logx=None, logdx=1.0, axis=-1):
    """
    Integrate along the given axis using the composite trapezoidal rule.
    Integrate `y` (`x`) along given axis.
    Parameters
    ----------
    logy : array_like
        log of input array to integrate.
    logx : array_like, optional
        The log of the sample points corresponding to the `logy` values.
        If `logx` is None, the sample points are assumed to be evenly
        spaced `exp(logdx)` apart. The default is None.
    logdx : scalar, optional
        The spacing between sample points when `logx` is None. The default is 0.
    axis : int, optional
        The axis along which to integrate.
    Returns
    -------
    logtrapz : float
        Log of the definite integral as approximated by trapezoidal rule.
    See Also
    --------
    sum, cumsum
    Notes
    -----
    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
    will be taken from `y` array, by default x-axis distances between
    points will be 1.0, alternatively they can be provided with `x` array
    or with `dx` scalar.  Return value will be equal to combined area under
    the red lines.
    References
    ----------
    .. [1] Wikipedia page: http://en.wikipedia.org/wiki/Trapezoidal_rule
    .. [2] Illustration image:
           http://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png
    Examples
    --------
    >>> np.trapz([1,2,3])
    4.0
    >>> np.exp(logtrapz(np.log([1,2,3])))
    4.0
    >>> np.trapz([1,2,3], x=[4,6,8])
    8.0
    >>> np.exp(logtrapz(np.log([1,2,3]), np.log([4,6,8])))
    8.0
    >>> np.trapz([1,2,3], dx=2)
    8.0
    >>> a = np.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.trapz(a, axis=0)
    array([ 1.5,  2.5,  3.5])
    >>> np.trapz(a, axis=1)
    array([ 2.,  8.])
    """
    logy = np.asanyarray(logy)
    if logx is None:
        logd = 0.
    else:
        logx = np.asanyarray(logx)
        if logx.ndim == 1:
            logdx = logx[1:] + np.log1p(-np.exp(logx[:-1]-logx[1:]))
            # reshape to correct shape
            shape = [1]*logy.ndim
            shape[axis] = logdx.shape[0]
            logd = logdx.reshape(shape)
        else:
            logd = logx[1:] + np.log1p(-np.exp(logx[:-1]-logx[1:]))

    nd = logy.ndim
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    ret = logsumexp(
        logd + np.logaddexp(logy[slice1], logy[slice2]) - np.log(2.),
        axis=axis)
    return ret

