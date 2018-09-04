# -*- coding: utf-8 -*-
import logging, sys
import numpy as np

from system import NBS_LJ
from nestedbasinsampling import LOG_CONFIG
from nestedbasinsampling.concurrent import RemoteWorker, BaseWorker, utils

logger = logging.getLogger('nbs.LJ_worker')

class NBS_worker(NBS_LJ, BaseWorker):

    def dummy(self, *args, **kwargs):
        return np.empty(50000)

def main():
    options, args = utils.parse_args()
    level = getattr(logging, options.verbosity)
    logging.basicConfig(level=level, **LOG_CONFIG)
    logger.info("running %s" % " ".join(sys.argv))
    logger.info("received options={:}".format(options))

    nameserver_kw = dict(host=options.nameserver, port=options.nsport)
    daemon_kw = dict(host=options.host, port=options.port)

    worker = NBS_worker(natoms=31, stepsize=0.1)
    remote_worker = RemoteWorker(
        worker, nameserver_kw=nameserver_kw, daemon_kw=daemon_kw)
    remote_worker.main()

if __name__ == '__main__':
    main()
