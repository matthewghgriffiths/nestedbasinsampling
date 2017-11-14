# -*- coding: utf-8 -*-

import copy
from bisect import bisect_left, bisect_right
from itertools import islice

import numpy as np
from scipy.special import gamma

def hyperspherevol(n, R=1.):
    return np.pi**(0.5*n) * R**n / gamma(0.5*n + 1)

def weighted_choice(ps):
    cdf = np.r_[0.,ps].cumsum()
    cdf /= cdf[-1]
    rand = np.random.rand()
    return cdf.searchsorted(rand)-1

def iter_minlength(iterable, minlength):
    try:
        islice(iterable, minlength-1, None).next()
        return True
    except StopIteration:
        return False

class Result(dict):
    """A container for the return values of an optimizer

    Attributes
    ----------
    initialcoords : ndarray
        The starting point of the optimization.
    coords : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    energy : ndarray
        energy at the solution
    grad : ndarray
        gradient at the solution
    Emax :
        list of energies sampled

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver.

    Also, since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


    def __repr__(self):
        if self.keys():
            m = max(map(len, self.keys())) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.iteritems()])
        else:
            return self.__class__.__name__ + "()"

class NestedSamplingError(Exception):
    """
    The exception to return if there is a problem with nested sampling.
    """
    reprstate = "Energy cutoff {:10.12g}, Replica Energy {:10.12g} "
    def __init__(self, Ecut, Enew, message=""):
        self.Ecut = Ecut
        self.Enew = Enew
        self.message = ""

    def __repr__(self):
        return self.reprstate.format(self.Ecut, self.Enew) + self.message

    def __str__(self):
        return repr(self)

def dict_update_keep(toupdate, update):
    """
    updates dict toupdate with values from dict update, except if key already
    present in toupdate
    """
    toupdate.update(pair for pair in update.iteritems()
                    if pair[0] not in toupdate or toupdate[pair[0]] is None)
    return toupdate

def dict_update_copy(primary, update):
    """
    updates dict toupdate with values from dict update, except if key already
    present in toupdate

    does not modify primary
    """
    toupdate = dict()
    toupdate.update(update)
    toupdate.update(pair for pair in primary.iteritems()
                    if pair[1] is not None or pair[0] not in update)
    return toupdate

def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    helper.__name__= func.__name__
    return helper
