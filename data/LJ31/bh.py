
import numpy as np
from pele.systems import LJCluster


system = LJCluster(31)


db = system.create_database()
bh = system.get_basinhopping(db)

bh.run(10000)
minima = db.minima()
for m in minima[:20]:
    np.savetxt(
        "min-{:0.5g}.txt".format(-m.energy),
        m.coords.reshape(-1, 3))
