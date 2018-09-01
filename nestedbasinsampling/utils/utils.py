# -*- coding: utf-8 -*-

from itertools import islice, izip, count
from collections import deque

import numpy as np
from math import sin, cos, acos
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


def angle_axis2mat(vector):
    ''' Rotation matrix of angle `theta` around `vector`
    Parameters
    ----------
    vector : 3 element sequence
       vector specifying axis for rotation. Norm of vector gives angle of
       rotation.
    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation
    Notes
    -----
    From: https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    '''
    vector = np.asanyarray(vector)
    theta = np.linalg.norm(vector)
    if theta==0.:
        return np.eye(3)
    x, y, z = vector/theta
    c, s = cos(theta), sin(theta)
    C = 1 - c
    xs, ys, zs = x * s, y * s, z * s
    xC, yC, zC = x * C, y * C, z * C
    xyC, yzC, zxC = x * yC, y * zC, z * xC
    return np.array([[x * xC + c, xyC - zs, zxC + ys],
                     [xyC + zs, y * yC + c, yzC - xs],
                     [zxC - ys, yzC + xs, z * zC + c]])

def mat2angle_axis(M):
    ''' Calculates rotation vector where the norm of the rotation vector
    indicates the angle of rotation from a rotation matrix M

    Parameters
    ----------
    M : (3,3) array like
        matrix encoding rotation matrix

    Returns
    -------
    v: array shape (3)
        rotation vector
    '''
    M = np.asanyarray(M)
    theta = acos(0.5*np.trace(M)-0.5)
    v = np.array([M[2,1]-M[1,2],M[0,2]-M[2,0],M[1,0]-M[0,1]])
    v *= 0.5*theta/sin(theta)
    return v
