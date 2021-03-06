
import numpy as np
import cPickle

from nestedbasinsampling import Database, Run
from nestedbasinsampling.nestedsampling.combine import combineAllRuns
from nestedbasinsampling.nestedsampling.integration import logsumexp, logtrapz
from collections import Counter, namedtuple

import matplotlib.pyplot as plt
plt.ion()

def calc_thermodynamics(Es, log_vol, Ts, Emin=None, log_f=0., ntrap=5000):
    if Emin is None:
        Emin = Es[-1]

    stride = max(Es.size/ntrap, 1)
    Es = Es[::-stride] - Emin
    log_vol = log_vol[::-stride]
    assert Es[0] >= 0

    ET = -Es[None,:]/Ts[:,None]
    logEs = np.log(Es)[None,:]

    logZ = logtrapz(ET, log_vol,axis=1) + log_f
    ET += logEs
    logE1 = logtrapz(ET, log_vol,axis=1) + log_f
    ET += logEs
    logE2s = logtrapz(ET, log_vol,axis=1) + log_f

    return logZ, logE1, logE2s, Emin

def calcCv(lZ, lE1, lE2, Ts, Emin=0.):
    U = np.exp(lE1 - lZ) + Emin
    U2 = np.exp(lE2 - lZ) + 2*Emin*U - Emin**2
    V = U - 0.5*k * Ts
    V2 = U2 - U**2 + V**2

    Cv = 0.5 * k + (V2 - V ** 2) * Ts ** -2

    Ret = namedtuple("CvReturn", "lZ U U2 Cv")
    return Ret(lZ=lZ, U=U, U2=U2, Cv=Cv)

Ts = np.linspace(0.01,0.6,1000)

def calc_CV(Es, log_vol, Ts):
    logZ, logE1, logE2, Emin = calc_thermodynamics(Es, log_vol, Ts)
    return calcCv(logZ, logE1, logE2, Ts, Emin)



db = Database('lj31_21.sqlite')

print "\n".join("{:s}:\n{:s}".format(p.name, str(p.value)) for p in db.properties())
print len(db.runs())


k = 31*3 - 6

runs = [r for r in db.runs() if len(r.Emax) > 100]
paths = db.paths()
m_count = Counter(p.child for p in paths)
mins = sorted(m_count.items(), key=lambda x:x[1])

min_split = set(m for m, c in mins[-1:])
m_replicas = set(p.parent for p in paths if p.child in min_split)
o_replicas = set(p.parent for p in paths if p.child not in min_split)

m_run = combineAllRuns([r for r in runs if r.child in m_replicas])
mres = calc_CV(m_run.Emax, m_run.log_frac, Ts)
plt.plot(Ts, mres.Cv, label='min')

o_run = combineAllRuns([r for r in runs if r.child in o_replicas])
ores = calc_CV(o_run.Emax, o_run.log_frac, Ts)
plt.plot(Ts, ores.Cv, label='other')

grun = combineAllRuns([m_run, o_run])
gres = calc_CV(grun.Emax, grun.log_frac, Ts)
plt.plot(Ts, gres.Cv, label='all')
plt.legend()

<<<<<<< HEAD
# import random
# runs = db.runs()
# for i in xrange(10):
#     random.shuffle(runs)
#     r = combineAllRuns(runs[:5000])
#     res = calc_CV(r.Emax, r.log_frac, Ts)
#     plt.plot(Ts, res.Cv)
=======
raise Exception


import random
runs = db.runs()
for i in xrange(10):
    random.shuffle(runs)
    r = combineAllRuns(runs[:5000])
    res = calc_CV(r.Emax, r.log_frac, Ts)
    plt.plot(Ts, res.Cv)
>>>>>>> cb91866c58e64cfff9d8128ba4c6e30b7a42c972

with open("rephd/lj31_noguts.pkl", 'wb') as f:
    cPickle.dump(gres._asdict(), f)


def get_min_run(m):
    replicas = [p.parent for p in paths if p.child == m]
    run = combineAllRuns([r for r in runs if r.child in replicas])
    return run
#run = combineAllRuns(runs)
#log_vol = run.log_frac
plt.ion()

grun = combineAllRuns(db.runs())

gres = calc_CV(grun.Emax, grun.log_frac, Ts)

plt.plot(Ts, gres.Cv)


f, axes = plt.subplots(2)
for m,c in mins[-10:]:
    run = get_min_run(m)
    res = calc_CV(run.Emax, run.log_frac, Ts)
    axes[0].plot(Ts, res.Cv)
    axes[1].plot(Ts, res.U)




#m_ret = calc_CV(m_run.Emax, m_run.log_frac, Ts)
#plt.plot(Ts, m_ret.Cv)
#o_ret = calc_CV(o_run.Emax, o_run.log_frac, Ts)
#plt.plot(Ts, o_ret.Cv)

plt.show()
