#!/usr/bin/python
import sys
import logging
from nestedbasinsampling import NBS_LJ, LOG_CONFIG
from nestedbasinsampling.concurrent import RemoteWorker, utils

logger = logging.getLogger('nbs.LJ_worker')
LJ_worker = NBS_LJ.Worker()


def main():
    options, args = utils.parse_args()

    level = getattr(logging, options.verbosity)
    logging.basicConfig(level=level, **LOG_CONFIG)
    logger.info("running %s" % " ".join(sys.argv))
    logger.info("received options={:}".format(options))

    nameserver_kw = dict(host=options.nameserver, port=options.nsport)
    daemon_kw = dict(host=options.host, port=options.port)
    RemoteWorker(LJ_worker(natoms=31, radius=2.5), pyro_name=options.worker,
                 manager_name=options.manager, nameserver_kw=nameserver_kw,
                 daemon_kw=daemon_kw).main()


if __name__ == '__main__':
    main()
