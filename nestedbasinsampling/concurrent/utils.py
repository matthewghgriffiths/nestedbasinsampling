# -*- coding: utf-8 -*-
"""
This module contains various utility functions for the concurrent modules.
"""
import random
import logging
import threading
from functools import wraps
from future.utils import iteritems


logger = logging.getLogger('nbs.concurrent.utils')

# timeout for the Queue object put/get blocking methods.
# it should really be infinity, but then keyboard interrupts don't work.
# so this is really just a hack, see http://bugs.python.org/issue1360
HUGE_TIMEOUT = 365 * 24 * 60 * 60 # one year
LOG_FORMAT = '%(asctime)s.%(msecs)03d : %(name)s : %(levelname)s : %(message)s'
LOG_FORMAT_THREAD = \
    '%(asctime)s.%(msecs)03d : %(name)s : %(levelname)s : %(thread)d : %(message)s'
LOG_DATEFMT ='%Y-%m-%d %H:%M:%S'
LOG_CONFIG = dict(format=LOG_FORMAT, datefmt=LOG_DATEFMT)
LOG_CONFIG_THREAD = dict(format=LOG_FORMAT_THREAD, datefmt=LOG_DATEFMT)

def random_hex(n=6):
    return "".join("{:1x}".format(random.randint(0,15)) for _ in xrange(n))

def getNS(host=None, port=None, broadcast=True, hmac_key=None):
    import Pyro4
    import subprocess
    try:
        return Pyro4.locateNS(
            host=host, port=port, broadcast=broadcast, hmac_key=hmac_key)
    except Pyro4.errors.NamingError:
        logger.info("Pyro name server not found; starting a new one")

    subprocess.Popen(['python', '-m', 'Pyro4.naming', '-n',
                      '0.0.0.0' if host is None else host])
    # TODO: spawn a proper daemon ala http://code.activestate.com/recipes/278731/ ?
    # like this, if there's an error somewhere, we'll never know... (and the loop
    # below will block). And it probably doesn't work on windows, either.
    while True:
        try:
            return Pyro4.locateNS(
                host=host, port=port, broadcast=broadcast, hmac_key=hmac_key)
        except:
            pass


def pyro_daemon(name, obj, random_suffix=False, ip=None,
                loopCondition=lambda: True,
                host=None, port=None, broadcast=True, hmac_key=None):
    """
    Register object with name server (starting the name server if not running
    yet) and block until the daemon is terminated. The object is registered under
    `name`, or `name`+ some random suffix if `random_suffix` is set.
    """
    import Pyro4

    if random_suffix:
        name += '.' + hex(random.randint(0, 0xffffff))[2:]
    with getNS(host=host, port=port, broadcast=broadcast, hmac_key=hmac_key) as ns:
        with Pyro4.Daemon(ip or get_my_ip(), port or 0) as daemon:
            # register server for remote access
            uri = daemon.register(obj, name)
            ns.remove(name)
            ns.register(name, uri)
            logger.info("%s registered with nameserver (URI '%s')" % (name, uri))
            daemon.requestLoop(loopCondition=loopCondition)


def proxy_alive(proxy):
    """ This tests whether a Pyro4 proxy object is still alive/
    can still be connected to
    """
    import Pyro4
    try:
        proxy._pyroBind()
        return True
    except Pyro4.errors.CommunicationError:
        return False


def get_my_ip():
    """
    Try to obtain our external ip (from the pyro nameserver's point of view)
    This tries to sidestep the issue of bogus `/etc/hosts` entries and other
    local misconfigurations, which often mess up hostname resolution.
    If all else fails, fall back to simple `socket.gethostbyname()` lookup.
    """
    import socket

    try:
        import Pyro4
        # we know the nameserver must exist, so use it as our anchor point
        ns = Pyro4.naming.locateNS()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((ns._pyroUri.host, ns._pyroUri.port))
        result, port = s.getsockname()
    except:
        try:
            # see what ifconfig says about our default interface
            import commands
            result = commands.getoutput("ifconfig").split("\n")[1].split()[1][5:]
            if len(result.split('.')) != 4:
                raise Exception()
        except:
            # give up, leave the resolution to gethostbyname
            result = socket.gethostbyname(socket.gethostname())
    return result


def threaded_eventloop(daemons, loopCondition=lambda :True):
    """
    Runs a threaded requestLoop for the Pyro daemons passed, whilst
    loopCondition returns true
    """
    import select
    from itertools import chain

    while loopCondition():
        logger.debug('Waiting for requests')
        # create sets of the socket objects we will be waiting on
        # (a set provides fast lookup compared to a list)
        daemonSockets = [set(daemon.sockets) for daemon in daemons]
        daemonEvents = [[] for daemon in daemons]
        sockets = list(chain(*daemonSockets))
        rs, _, _ = select.select(sockets, [], [], 3)
        for s in rs:
            for events, _sockets in zip(daemonEvents, daemonSockets):
                if s in _sockets:
                    events.append(s)
        for daemon, events in zip(daemons, daemonEvents):
            if events:
                logger.debug("a daemon received a request")
                daemon.events(events)


def gethostname():
    import socket
    return socket.gethostname()

# maybe not needed?
def resolve_name(obj):
    import Pyro4
    if isinstance(obj, Pyro4.Proxy):
        uri = obj._pyroUri.asString()
    elif isinstance(obj, Pyro4.URI):
        uri = obj.asString()
    else:
        uri = obj
    return uri[5:].split('@')[0]


def parse_args(args=None):
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option(
        "-n", "--nameserver", dest="nameserver", help="hostname of nameserver")
    parser.add_option(
        "-q", "--nsport", dest="nsport", type="int", default=0,
        help="port of nameserver")
    parser.add_option(
        "-l", "--host", dest="host", help="hostname to bind server on")
    parser.add_option(
        "-p", "--port", dest="port", type="int", default=0,
        help="port to bind server on (0=random)")
    parser.add_option(
        "-i", "--niter", dest='niter', help="number of iterations to run",
        default=1000, type='int')
    parser.add_option(
        "-d", "--database", dest='database', help='location of database file')
    parser.add_option(
        "-v", "--verbosity", dest='verbosity', default='INFO',
        help='set logging level, options: DEBUG, INFO, CRITICAL, ERROR')

    options, args = parser.parse_args(args)
    if options.host is None: options.host = gethostname()
    return options, args


def synchronous(tlockname):
    """
    A decorator to place an instance-based lock around a method.
    Adapted from http://code.activestate.com/recipes/577105-synchronization-decorator-for-class-methods/
    """
    def _synched(func):
        @wraps(func)
        def _synchronizer(self, *args, **kwargs):
            tlock = getattr(self, tlockname)
            logger.debug(
                "acquiring lock %r for %s" % (tlockname, func.__name__))

            # use lock as a context manager to perform safe acquire/release pairs
            with tlock:
                logger.debug(
                    "acquired lock %r for %s" % (tlockname, func.__name__))
                result = func(self, *args, **kwargs)
                logger.debug(
                    "releasing lock %r for %s" % (tlockname, func.__name__))
                return result

        return _synchronizer
    return _synched
