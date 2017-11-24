# -*- coding: utf-8 -*-

from itertools import islice, izip, count
from collections import deque

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

def len_iter(iterable):
    """
    Consume an iterable and return the number of items in the iterator
    """
    counter = count()
    deque(izip(iterable, counter), maxlen=0)
    return next(counter)

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
