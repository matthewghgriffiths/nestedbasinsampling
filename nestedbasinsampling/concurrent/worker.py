import sys
import logging
import threading
import Queue as queue

import Pyro4
from .base import BasePyro
from . import utils

logger = logging.getLogger('nbs.concurrent.worker')
HUGE_TIMEOUT = 365 * 24 * 60 * 60  # one year
Pyro4.config.SERVERTYPE = "multiplex"
sys.excepthook = Pyro4.util.excepthook


class BaseWorker(object):

    def initialise(self, *args, **kwargs):
        pass

    def __call__(self, job):
        method, args, kwargs, label = job
        return job, getattr(self, method)(*args, **kwargs)


@Pyro4.expose
class RemoteWorker(BasePyro):
    def __init__(self, worker=None, max_threads=2,
                 manager_name='nbs.remote_manager',
                 pyro_name='nbs.worker',  pyro_metadata=('worker',),
                 daemon_kw={}, nameserver_kw={}):
        super(RemoteWorker, self).__init__(
            pyro_name=pyro_name, pyro_metadata=pyro_metadata,
            nameserver_kw=nameserver_kw, daemon_kw=daemon_kw,
            random_suffix=True)

        self.worker = worker
        self.max_threads = max_threads
        self.manager_name = manager_name
        self._remote_manager = None
        self.lock = threading.Lock()
        self.initialised = False

    @property
    def remote_manager(self):
        if self._remote_manager is None:
            with utils.getNS(**self.nameserver_kw) as ns:
                self._remote_manager = Pyro4.Proxy(
                    ns.lookup(self.manager_name))
        return self._remote_manager

    @Pyro4.oneway
    def initialise(self, *args, **kwargs):
        logger.info("initialising {:s}".format(self.name))
        self.worker.initialise(*args, **kwargs)
        self.initialised = True
        self.request_job()

    # As worker job may not be threadsafe, ensure that only one thread
    # can call this at anyone time
    @Pyro4.oneway
    @utils.synchronous('lock')
    def request_job(self):
        try:
            if self.initialised:
                job_id, job = self.remote_manager.get_job(self.name)
                logger.debug(
                    "successfully requested job #{:d}, starting:".format(job_id))
                try:
                    work = self.worker(job)
                    logger.debug(
                        "successfully finished job, sending".format(job_id))
                    self.remote_manager.receive_output(self.name, job_id, work)
                    logger.debug(
                        "%s sent work back to manager" % self.name)
                except Exception as e:
                    logger.critical(
                        "job #{:d} failed with error:".format(job_id))
                    logger.exception(e)
            else:
                self.remote_manager.connect(self.name)

        except queue.Empty:
            logger.debug("job queue empty")
        except Pyro4.errors.CommunicationError:
            logger.critical(
                "%s coudn't connect to remote manager, exiting" % self.name)
            self.exit()

    def main_loop(self):
        """
        """
        self.request_job()
        return True
