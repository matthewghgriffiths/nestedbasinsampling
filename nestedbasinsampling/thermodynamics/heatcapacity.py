# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np

from ..nestedsampling.integration import logtrapz


def _calc_thermodynamics(Es, log_vol, Ts, Emin=None, log_f=0., ntrap=5000):
    if Emin is None:
        Emin = Es[-1]

    stride = max(Es.size/ntrap, 1)
    Es = Es[::-stride] - Emin
    log_vol = log_vol[::-stride]
    assert Es[0] >= 0

    ET = -Es[None, :] / Ts[:, None]
    logEs = np.log(Es)[None, :]

    logZ = logtrapz(ET, log_vol, axis=1) + log_f
    ET += logEs
    logE1 = logtrapz(ET, log_vol, axis=1) + log_f
    ET += logEs
    logE2s = logtrapz(ET, log_vol, axis=1) + log_f

    return logZ, logE1, logE2s, Emin


def _calcCv(lZ, lE1, lE2, Ts, k, Emin=0.):
    U = np.exp(lE1 - lZ) + Emin
    U2 = np.exp(lE2 - lZ) + 2*Emin*U - Emin**2
    V = U - 0.5 * k * Ts
    V2 = U2 - U**2 + V**2

    Cv = 0.5 * k + (V2 - V ** 2) * Ts ** -2

    Ret = namedtuple("CvReturn", "lZ U U2 Cv")
    return Ret(lZ=lZ, U=U, U2=U2, Cv=Cv)


def calc_CV(Es, log_vol, Ts, k):
    logZ, logE1, logE2, Emin = _calc_thermodynamics(Es, log_vol, Ts)
    return _calcCv(logZ, logE1, logE2, Ts, k, Emin)
