# -*- coding: utf-8 -*-
import logging, sys
import numpy as np

from system import NBS_LJ
from nestedbasinsampling import LOG_CONFIG
from nestedbasinsampling.concurrent import RemoteWorker, BaseWorker

logger = logging.getLogger('nbs.LJ_worker')

class NBS_worker(NBS_LJ, BaseWorker):

    def dummy(self, *args, **kwargs):
        return np.empty(50000)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, **LOG_CONFIG)
    logger.info("running %s" % " ".join(sys.argv))
    logger.debug("running %s" % " ".join(sys.argv))


    worker = NBS_worker(natoms=31, stepsize=0.1)
    remote_worker = RemoteWorker(worker)
    remote_worker.main()
