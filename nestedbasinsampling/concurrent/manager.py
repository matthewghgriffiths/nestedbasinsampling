# -*- coding: utf-8 -*-
import os
import logging
import Queue as queue
from itertools import count

import numpy as np
import Pyro4

from . import utils
from .base import BasePyro
from ..storage import Replica, Run
from ..utils import Result


logger = logging.getLogger('nbs.concurrent.manager')
Pyro4.config.SERVERTYPE = "multiplex"


class BaseManager(object):
    """
    """
    settings = {}

    def initalise_worker(self, worker_name, worker):
        logger.info("initialising {:s}".format(worker_name))
        worker.initialise(**self.settings)

    def receive_work(self, work):
        raise NotImplementedError

    def get_job(self):
        raise NotImplementedError

    def stop_criterion(self):
        return False


class NBS_Manager(BaseManager):
    def __init__(self, nbs_system, database=None, max_iter=10,
                 receive_funcs=None):
        self.nbs_system = nbs_system
        if database is None:
            database = self.nbs_system.get_database()
        self.database = database
        self.settings = getattr(self.nbs_system, 'settings', {})

        self.max_iter = int(max_iter)
        self.curr_iter = 0
        self._g_replica = None

        self.results = []
        nfev = self.database.get_property('nfev')
        if nfev is None:
            nfev = self.database.add_property('nfev', 0)
        self.nfev_property = nfev

        self.receive_funcs = dict(nopt=self.add_run)
        if receive_funcs is not None:
            self.receive_funcs.update(receive_funcs)

    @property
    def g_replica(self):
        if self._g_replica is None:
            replicas = self.database.session.query(Replica).\
                order_by(Replica.energy.desc()).limit(1).all()
            if replicas:
                self._g_replica, = replicas
            else:
                self._g_replica = self.database.addReplica(
                    np.inf, None, stepsize=self.nbs_system.stepsize)

        return self._g_replica

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
        assert job[1][0] is None, job[1][1] == np.inf

        res = Result(res)
        Es = np.r_[res.initialenergy, res.Emax]
        nlive = np.ones_like(Es)

        # Adding to database
        child = self.database.addReplica(Es[-1], res.coords, commit=False)
        m = self.database.addMinimum(res.energy, res.coords, commit=False)
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


@Pyro4.expose
class RemoteManager(BasePyro):
    """
    """
    def __init__(self, manager, pyro_name='nbs.remote_manager', max_jobs=10,
                 worker_name='nbs.worker',
                 pyro_metadata=None, daemon_kw={}, nameserver_kw={}):
        super(RemoteManager, self).__init__(
            pyro_name=pyro_name, pyro_metadata=pyro_metadata,
            nameserver_kw=nameserver_kw, daemon_kw=daemon_kw,
            random_suffix=False)

        self.manager = manager
        self.max_jobs = max_jobs
        self.worker_name = worker_name
        self.job_queue = queue.Queue(maxsize=self.max_jobs)
        self.finished_work = queue.Queue()
        self.count = count()
        self.workers = {}

    @Pyro4.oneway
    def connect(self, worker_name):
        logger.debug("{:s} connecting".format(worker_name))
        with utils.getNS(**self.nameserver_kw) as ns:
            worker = Pyro4.Proxy(ns.lookup(worker_name))
            self.workers[worker_name] = worker
            self.manager.initalise_worker(worker_name, worker)
            worker.request_job()

    @Pyro4.expose
    def get_job(self, worker_name):
        logger.debug("{:s} requesting a new job".format(worker_name))
        job_id, job = self.job_queue.get(block=False)
        logger.debug("{:s} sent job# {:d}".format(worker_name, job_id))
        return job_id, job

    @Pyro4.oneway
    def receive_output(self, worker_name, job_id, work):
        logger.debug("%s finished job #%i" % (worker_name, job_id))
        self.finished_work.put((job_id, work), block=True)
        if worker_name not in self.workers:
            with utils.getNS(**self.nameserver_kw) as ns:
                worker = Pyro4.Proxy(ns.lookup(worker_name))
                self.workers[worker_name] = worker
        self.workers[worker_name].request_job()

    def main_loop(self):
        """This method is repeatedly called whilst this object is running
        """
        while not self.finished_work.empty():
            try:
                job_id, work = self.finished_work.get()
                self.manager.receive_work(work)
            except queue.Empty:
                pass

        if self.manager.stop_criterion():
            self.exit()
            return False

        while not self.job_queue.full():
            job_id = next(self.count)
            job = self.manager.get_job()
            self.job_queue.put((job_id, job))

        return True

    def get_workers(self):
        """Return pyro URIs of all registered workers.
        """
        with utils.getNS(**self.nameserver_kw) as ns:
            return ns.list(prefix=self.worker_name)

    @Pyro4.expose
    def exit(self):
        """
        Terminate all registered workers and then the dispatcher.
        """
        for workerid, worker in self.workers.items():
            if utils.proxy_alive(worker):
                logger.info("terminating worker %s" % workerid)
                worker.exit()
        logger.info("terminating remote manager")
        # exit the whole process (not just this thread ala sys.exit())
        os._exit(0)
