#!/usr/bin/python
import logging
import sys
from nestedbasinsampling import NBS_LJ, LOG_CONFIG
from nestedbasinsampling.concurrent import RemoteManager, utils

"""

example run:

python lj_manager.py -n dexter\
    -i 5000 \
    -m ../../data/LJ31/min-133.10.txt=1000\
    -m ../../data/LJ31/min-133.18.txt=1000\
    -m ../../data/LJ31/min-133.29.txt=1000\
    -m ../../data/LJ31/min-133.59.txt=1000\
    --database=lj31_02.sqlite

python lj_manager.py -n dexter\
    -i 0 \
    -m ../../data/LJ31/min-133.10.txt=5\
    -m ../../data/LJ31/min-133.18.txt=5\
    -m ../../data/LJ31/min-133.29.txt=5\
    -m ../../data/LJ31/min-133.59.txt=5\

python lj_manager.py \
    -i 5 \
    -m ../../data/LJ31/min-133.10.txt=5\
    -m ../../data/LJ31/min-133.18.txt=5\
    -m ../../data/LJ31/min-133.29.txt=5\
    -m ../../data/LJ31/min-133.59.txt=5\
    --database=lj31_01.sqlite

"""

logger = logging.getLogger('nbs.LJ_manager')

settings = dict(
    natoms=31, stepsize=0.1,
    sampler_kws=dict(
        max_depth=7, remove_linear_momentum=True,
        remove_angular_momentum=True,
        remove_initial_linear_momentum=False,
        remove_initial_angular_momentum=False),
    nopt_kws=dict(
        nsteps=2000, MC_steps=10, target_acc=0.4, nsave=40, tol=1e-2,
        nwait=10, kalman_discount=100., max_tries=5))


def main():
    parser = utils.get_default_parser()
    parser.add_option(
        "-m", "--min", dest='mins', action='append', help="mins to load")
    options, args = utils.parse_args(parser)

    level = getattr(logging, options.verbosity)
    logging.basicConfig(level=level, **LOG_CONFIG)
    logger.info("running %s" % " ".join(sys.argv))
    logger.info("received options={:}".format(options))

    niter = options.niter
    nameserver_kw = dict(host=options.nameserver, port=options.nsport)
    daemon_kw = dict(host=options.host, port=options.port)
    manager_kws = dict(max_iter=niter, mins=options.mins)
    if options.database is not None:
        manager_kws['db'] = options.database

    system = NBS_LJ(**settings)
    manager = system.get_manager(**manager_kws)
    remote_manager = RemoteManager(
        manager, pyro_name=options.manager, worker_name=options.worker,
        nameserver_kw=nameserver_kw, daemon_kw=daemon_kw)
    remote_manager.main()

if __name__ == '__main__':
    main()
