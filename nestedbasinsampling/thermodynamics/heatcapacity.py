# -*- coding: utf-8 -*-

import numpy as np

def heat_capacity_func(E, T=[1.], Emin=-44.):
    T = np.atleast_1d(T)
    E = np.atleast_1d(E)-Emin
    ET = (E[None,:]/T[:,None])
    return ET**2 * np.exp(-ET)

def partition_func(E, T=[1.], Emin=-44.):
    T = np.atleast_1d(T)
    E = np.atleast_1d(E)-Emin
    ET = (E[None,:]/T[:,None])
    return np.exp(-ET)


def E2_func(E, T=[1.], Emin=-44.):
    T = np.atleast_1d(T)
    E = np.atleast_1d(E)-Emin
    ET = (E[None,:]/T[:,None])
    return ET**2 * np.exp(-ET)

def E1_func(E, T=[1.], Emin=-44.):
    T = np.atleast_1d(T)
    E = np.atleast_1d(E)-Emin
    ET = (E[None,:]/T[:,None])
    return ET * np.exp(-ET)

def E0_func(E, T=[1.], Emin=-44.):
    T = np.atleast_1d(T)
    E = np.atleast_1d(E)-Emin
    ET = (E[None,:]/T[:,None])
    return np.exp(-ET)

def harmE0_func(Emin, Emax, k=1, T=[1.], Emin=-44.):
    pass