# -*- coding: utf-8 -*-
import numpy as np
import logging, sys

from system import NBS_LJ
from nestedbasinsampling import LOG_CONFIG, Database, Replica, Result
from nestedbasinsampling.concurrent import RemoteManager, BaseManager

logger = logging.getLogger('nbs.lj_manager')

class NBS_Manager(BaseManager):

    def __init__(self, nbs_system, database=None, max_iter=10,
                 receive_funcs=None):
        self.nbs_system = nbs_system
        if database is None:
            database = self.nbs_system.get_database()
        self.database = database

        self.max_iter = max_iter
        self.curr_iter = 0

        self.results = []

        replicas = self.database.session.query(Replica).\
            order_by(Replica.energy).limit(1).all()
        if replicas:
            self.g_replica, = replicas
        else:
            self.g_replica = self.database.addReplica(
                np.inf, None, stepsize=self.nbs_system.stepsize)

        nfev = self.database.get_property('nfev')
        if nfev is None:
            nfev = self.database.add_property('nfev', 0)
        self.nfev_property = nfev

        self.receive_funcs = dict(nopt=self.add_run)
        if receive_funcs is not None: self.receive_funcs.update(receive_funcs)

    def get_job(self):
        logger.debug('creating job')
        rep = self.g_replica
        return 'nopt', (rep.coords, np.inf, self.nbs_system.stepsize), {}

    def _receive_work(self, work):
        self.results.append(work)
        logger.info('RECEIVED')
        self.curr_iter += 1
        logger.info('received work, total = {:d}'.format(self.curr_iter))

    def add_run(self, work):
        logger.debug('received work')
        job, res = work
        res = Result(res)
        Es = np.r_[res.initialenergy, res.Emax]
        nlive = np.ones_like(Es)

        # Adding to database
        child = self.database.addReplica(Es[-1], res.coords, commit=False)
        m = self.database.addMinimum(res.energy, res.coords, commit=False)
        assert job[1][0] == None, job[1][1] == np.inf
        parent = self.g_replica
        path = self.database.addPath(Es[-1], child, m, commit=False)
        run = self.database.addRun(Es, nlive, parent, child, commit=False)
        self.nfev_property.value += res.nfev

        self.database.session.commit()
        self.curr_iter += 1

        logger.info(
            ("received nested optimisation run,  minE={:10.5g}, nsteps={:4d}, "
             "nfev={:7d}".format(m.energy, Es.size, res.nfev)))
        return run, child, m, path

    def receive_work(self, work):
        job, output = work
        job_name = job[0]
        self.receive_funcs[job_name](work)

    def stop_criterion(self):
        return self.curr_iter > self.max_iter


def main():
    pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, **LOG_CONFIG)
    logger.info("running %s" % " ".join(sys.argv))
    logger.debug("running %s" % " ".join(sys.argv))

    system = NBS_LJ(natoms=31, stepsize=0.1)
    manager = NBS_Manager(system, max_iter=5)
    remote_manager = RemoteManager(manager)
    remote_manager.main()
