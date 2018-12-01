#!/usr/bin/python
import logging
from nestedbasinsampling import NBS_LJ, LOG_CONFIG
from nestedbasinsampling.concurrent import RemoteWorker

LJ_worker = NBS_LJ.Worker()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, **LOG_CONFIG)
    RemoteWorker(LJ_worker(natoms=31, radius=2.5)).main()
