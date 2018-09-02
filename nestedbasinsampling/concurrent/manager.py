
import os, logging
import Queue as queue
from itertools import count

import Pyro4
from .base import BasePyro, LOG_CONFIG
from . import utils

logger = logging.getLogger('nbs.concurrent.manager')
Pyro4.config.SERVERTYPE = "multiplex"

class BaseManager(object):

    def receive_work(self, work):
        raise NotImplementedError

    def get_job(self):
        raise NotImplementedError

    def stop_criterion(self):
        return False

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
                self.workers[worker_name] =  Pyro4.Proxy(
                    ns.lookup(worker_name))
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
