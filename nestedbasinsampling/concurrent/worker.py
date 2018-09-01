
#from __future__ import with_statement
import os, sys, logging, time
import threading
import tempfile
import Queue as queue

import Pyro4
from .base import BasePyro, LOG_CONFIG
from . import utils

logger = logging.getLogger('nbs.worker')
SAVE_DEBUG = 0 # save intermediate models after every SAVE_DEBUG updates (0 for never)
HUGE_TIMEOUT = 365 * 24 * 60 * 60 # one year


Pyro4.config.SERVERTYPE = "multiplex"
sys.excepthook=Pyro4.util.excepthook


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

    @property
    def remote_manager(self):
        if self._remote_manager is None:
            with utils.getNS(**self.nameserver_kw) as ns:
                self._remote_manager = Pyro4.Proxy(
                    ns.lookup(self.manager_name))
        return self._remote_manager

    @Pyro4.oneway
    def request_job(self):
        try:
            job_id, job = self.remote_manager.get_job(self.name)
            logger.debug(
                "successfully requested job #{:d}, starting:".format(job_id))
            try:
                work = self.worker(job)
                logger.debug(
                    "successfully finished job sending".format(job_id))
                self.remote_manager.receive_output(self.name, job_id, work)
            except Exception as e:
                logger.critical(
                    "job #{:d} failed with error:".format(job_id))
                logger.exception(e)
        except queue.Empty:
            logger.debug("job queue empty")
        except Pyro4.errors.CommunicationError:
            logger.critical(
                "%s coudn't connect to remote manager" % self.name)

    def main_loop(self):
        """
        """
        self.request_job()
        return True
