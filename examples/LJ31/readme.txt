

To run this system on a distributed server you need to start a nameserver that is accessible to
all the worker nodes, this can be started with the following command:

python -m Pyro4.naming --host=[HOSTNAME]

Where [HOSTNAME] is the name of the host that the process is running on, e.g. dexter, sinister etc...

You then need to start the manager process, this is probably best done on an interactive prompt with the following
command:

python lj_manager.py --nameserver=[HOSTNAME] --database=[path-to-database] 

Where [HOSTNAME] is the same for as for the naming server

The worker process can then be started by running

python lj_worker.py --nameserver=[HOSTNAME]

and these will automatically connect to the manager to receive, calculate and send back jobs.
