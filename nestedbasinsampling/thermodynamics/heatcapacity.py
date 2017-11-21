# -*- coding: utf-8 -*-

import numpy as np

def heat_capacity_func(E, T=[1.], Emin=-44.):
    T = np.atleast_1d(T)
    E = np.atleast_1d(E)-Emin
    ET = (E[None,:]/T[:,None])
    return ET**2 * np.exp(-ET)