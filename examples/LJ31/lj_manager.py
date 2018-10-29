# -*- coding: utf-8 -*-
import numpy as np
import logging, sys

from system import NBS_LJ
from nestedbasinsampling import LOG_CONFIG, Database, Replica, Result, Run
from nestedbasinsampling.concurrent import RemoteManager, BaseManager, utils

logger = logging.getLogger('nbs.lj_manager')

class NBS_Manager(BaseManager):

    def __init__(self, nbs_system, database=None, max_iter=10,
                 receive_funcs=None):
        self.nbs_system = nbs_system
        if database is None:
            database = self.nbs_system.get_database()
        self.database = database

        self.max_iter = int(max_iter)
        self.curr_iter = 0

        self.results = []

        replicas = self.database.session.query(Replica).\
            order_by(Replica.energy.desc()).limit(1).all()
        if replicas:
            self.g_replica, = replicas
        else:
            self.g_replica = self.database.addReplica(
                np.inf, None, stepsize=self.nbs_system.stepsize)

        print self.g_replica.energy

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
        nruns = self.database.session.query(Run).count()
        logger.info("database has {:d} runs".format(nruns))
        return nruns > self.max_iter


def main():
    options, args = utils.parse_args()
    level = getattr(logging, options.verbosity)
    logging.basicConfig(level=level, **LOG_CONFIG)
    logger.info("running %s" % " ".join(sys.argv))
    logger.info("received options={:}".format(options))

    niter = options.niter
    nameserver_kw = dict(host=options.nameserver, port=options.nsport)
    daemon_kw = dict(host=options.host, port=options.port)

    system = NBS_LJ(natoms=31, stepsize=0.1)
    database = options.database
    if database is not None:
        logger.info("connecting to database: %s" % database)
        database = system.get_database(database)

    manager = NBS_Manager(system, database=database, max_iter=niter)
    remote_manager = RemoteManager(
        manager, nameserver_kw=nameserver_kw, daemon_kw=daemon_kw)
    remote_manager.main()

if __name__ == '__main__':
    main()
