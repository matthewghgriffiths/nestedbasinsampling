# -*- coding: utf-8 -*-
import os
import logging
import Queue as queue
from itertools import count
from collections import deque
import time

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
    """
    basin_opts : dict[str, (Minimum, n_runs)]
    """
    def __init__(self, nbs_system, database=None, max_iter=10,
                 basin_opts=None, mins=None, n_remember=20):
        self.nbs_system = nbs_system
        if database is None:
            database = self.nbs_system.get_database()
        self.database = database
        self.settings = getattr(self.nbs_system, 'settings', {})

        self.time_received = deque(maxlen=n_remember)
        self.max_iter = int(max_iter)
        self.curr_iter = 0
        self._g_replica = None

        self.results = []
        nfev = self.database.get_property('nfev')
        if nfev is None:
            nfev = self.database.add_property('nfev', 0)
        self.nfev_property = nfev

        self.jobs_left = {'nopt': max_iter}
        self.basins = {'nopt': (None, self.g_replica)}  # for standard nopts

        if basin_opts is not None:
            # for basin nopts
            for basin, (m, m_iter) in basin_opts.items():
                self.jobs_left[basin] = m_iter
                basin_rep = self.database.addReplica(
                    np.inf, m.coords, stepsize=self.nbs_system.stepsize)
                self.database.addPath(np.inf, basin_rep, m)
                self.basins[basin] = m, basin_rep

        if mins is not None:
            self.parse_mins(mins)

    def parse_mins(self, mins):
        pot = self.nbs_system.pot
        for basinm in mins:
            logger.info("parsing {:s}".format(basinm))
            min_path, m_iter = basinm.split('=')
            _, min_name = os.path.split(min_path)
            m_coords = np.loadtxt(min_path).flatten()
            E = pot.getEnergy(m_coords)
            m = self.database.addMinimum(E, m_coords)
            q = self.database.session.query(
                Replica).filter(Replica.energy==np.inf)
            for basin_rep in q:
                # check if there already exists appropriate replica
                if (basin_rep.coords is not None and
                        self.database.compareMinima(m, basin_rep)):
                    break
            else:
                # if none exist then make new one
                basin_rep = self.database.addReplica(
                    np.inf, m.coords, stepsize=self.nbs_system.stepsize)
            self.database.addPath(np.inf, basin_rep, m)

            # adding jobs
            self.basins[min_name] = (m, basin_rep)
            self.jobs_left[min_name] = int(m_iter)

    @property
    def g_replica(self):
        if self._g_replica is None:
            replicas = self.database.session.query(Replica).\
                order_by(Replica.energy.desc())
            try:
                self._g_replica, = (
                    r for r in replicas if r.coords is None)
            except ValueError:
                # if no g_replicas exist make one
                self._g_replica = self.database.addReplica(
                    np.inf, None, stepsize=self.nbs_system.stepsize)

        return self._g_replica

    def get_job(self):
        # get nonzero jobs left
        basins, n_left = zip(*(
            item for item in self.jobs_left.items() if item[1]))
        n_basins = len(basins)  # number of basins still active

        if n_basins:
            if n_basins > 1:
                p = np.random.dirichlet(n_left)
                basin = basins[np.random.multinomial(1, p).nonzero()[0][0]]
            else:
                basin, = basins

            if basin == 'nopt':
                return self.nopt_job()
            else:
                return self.basin_nopt_job(basin)
        else:
            return None

    def nopt_job(self):
        logger.debug('creating job')
        rep = self.g_replica
        args = (rep.coords, np.inf, self.nbs_system.stepsize)
        return 'nopt', args, {}, 'nopt'

    def basin_nopt_job(self, basin):
        m, basin_rep = self.basins[basin]
        args = (m.coords, np.inf, self.nbs_system.stepsize)
        kwargs = self.nbs_system.basin_nopt_kws
        return ('nopt', args, kwargs, basin)

    def find_basin(self, coords):
        for basin, (m, basin_rep) in self.basins.iteritems():
            if (m.coords == coords).all():
                break
        else:
            raise Exception("coords not found")
        return basin

    def _receive_work(self, work):
        self.results.append(work)

    def _get_rate(self):
        if len(self.time_received) > 1:
            return (len(self.time_received) - 1) / (
                self.time_received[-1] - self.time_received[0])
        else:
            return 0.

    def _ttc(self):
        rate = self._get_rate()
        if rate:
            njobs = sum(self.jobs_left.values())
            t = njobs / rate
            mins, s = divmod(int(t), 60)
            h, m = divmod(mins, 60)
            if h:
                return '{0:d}:{1:02d}:{2:02d}'.format(h, m, s)
            else:
                return '{0:02d}:{1:02d}'.format(m, s)
        else:
            return '?'

    def add_run(self, work):
        logger.debug('adding run')
        self.time_received.append(time.time())
        job, res = work
        method, args, kwargs, label = job
        m, parent = self.basins[label]

        res = Result(res)
        Es = np.r_[res.initialenergy, res.Emax]
        nlive = np.ones_like(Es)

        # Adding to database
        child = self.database.addReplica(Es[-1], res.coords, commit=False)
        if m is None:
            m = self.database.addMinimum(res.energy, res.coords, commit=False)
        path = self.database.addPath(Es[-1], child, m, commit=False)
        run = self.database.addRun(
            Es, nlive, parent, child, commit=False)
        if 'stepsize' in res:
            run.stepsizes = res.stepsize
        self.nfev_property.value += res.nfev

        self.database.session.commit()

        if self.jobs_left[label]:
            self.jobs_left[label] -= 1

        logger.info((
            "received nested optimisation run, for basin {}, minE={:10.5g}, "
            "nsteps={:4d}, nfev={:7d}, rate={:6.2g} it/s".format(
                label, m.energy, Es.size, res.nfev, self._get_rate())))
        return run, child, m, path

    receive_work = add_run

    def stop_criterion(self):
        nruns = self.database.session.query(Run).count()
        logger.info((
            "database has {:d} runs, time left: {:s}, jobs left: {:s}".format(
                nruns,
                self._ttc(),
                ", ".join("{}={}".format(*item)
                          for item in self.jobs_left.items()))))
        criterion = not any(self.jobs_left.values())
        return criterion


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
            if job is None:
                return False
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
        # Doesn't seem to work...
        # for workerid, worker in self.workers.items():
        #     try:
        #         worker.exit()
        #         logger.info("terminating worker %s" % workerid)
        #     except Exception:
        #         pass
        logger.info("terminating remote manager")
        # exit the whole process (not just this thread ala sys.exit())
        os._exit(0)
