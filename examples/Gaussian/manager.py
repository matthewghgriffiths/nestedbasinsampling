import logging
from nestedbasinsampling import NBS_manager, LOG_CONFIG, SamplingError, Replica
from gaussian_system import get_system, system_kws, Ecut
import sqlalchemy

logging.basicConfig(level=logging.INFO, **LOG_CONFIG)


system = get_system()
db = system.get_database('gaussian3.sqlite')
replicas = db.session.query(Replica).\
    order_by(Replica.energy.desc()).limit(1).all()
if replicas:
    g_replica, = replicas
else:
    g_replica = db.addReplica(
        Ecut, None, stepsize=system.stepsize)
manager = NBS_manager(system, database=db)

jobs = [manager.get_job() for _ in xrange(1000)]
for job in jobs:
    try:
        manager.receive_work(manager(job))
    except (sqlalchemy.exc.OperationalError, SamplingError):
        try:
            db.session.rollback()
        except sqlalchemy.exc.OperationalError:
            pass
