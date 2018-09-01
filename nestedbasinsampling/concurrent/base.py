
import os, logging
from . import utils
import Pyro4
from Pyro4.util import SerializerBase, SerpentSerializer

logger = logging.getLogger('nbs.base')
SAVE_DEBUG = 0
LOG_CONFIG = utils.LOG_CONFIG

__all__ = ['BasePyro', 'LOG_CONFIG']
Pyro4.config.SERVERTYPE = "multiplex"

@Pyro4.expose
class BasePyro(object):

    def __init__(self, pyro_name='fermi.base', pyro_metadata=None,
                 daemon_kw={}, nameserver_kw={}, random_suffix=False):

        self.pyro_name = pyro_name
        self.pyro_metadata = pyro_metadata
        self.nameserver_kw = nameserver_kw
        self.daemon_kw = daemon_kw
        self.random_suffix = random_suffix
        self.callback = None

    def _unique_name(self, name, nameserver=None):
        if nameserver is None:
            ns =  utils.getNS(**self.nameserver_kw)
        else:
            ns = nameserver

        newname = name + '.' + utils.random_hex()
        if newname in ns.list():
            newname = name + '.' + utils.random_hex()

        if nameserver is None:
            ns._pyroRelease()

        return newname

    def _register(self, daemon=None):
        if daemon is None:
            daemon = Pyro4.Daemon(**self.daemon_kw)
        with utils.getNS(**self.nameserver_kw) as ns:
            # set name
            if self.random_suffix:
                self.name = self._unique_name(self.pyro_name, ns)
            else:
                self.name = self.pyro_name

            # register server for remote access
            self.uri = daemon.register(self, self.name)
            ns.remove(self.name)
            ns.register(self.name, self.uri,
                        safe=True, metadata=self.pyro_metadata)
            if self.callback is None:
                self.callback = Pyro4.Proxy(self.uri)
            logger.info(
                "{:s} registered with nameserver (URI '{:s}')".format(
                    self.name, str(self.uri)))
        return self.name, daemon

    def Daemon(self, daemon=None):
        name, daemon = self._register(daemon)
        return daemon

    def main(self):
        with self.Daemon() as daemon:
            utils.threaded_eventloop([daemon], self.main_loop)

    def main_loop(self):
        return True

    @Pyro4.oneway
    def exit(self):
        """
        Terminate the program.
        """
        logger.info("terminating %s" % self.name)
        os._exit(0)

    def reset(self):
        pass


def _register_numpy_serializers():
    import numpy as np

    def int_to_dict(value):
        return dict(
            __class__ = 'numpy.' + type(value).__name__,
            value = int(value))

    def float_to_dict(value):
        return dict(
            __class__ = 'numpy.' + type(value).__name__,
            value = float(value))

    def ndarray_to_dict(array):
        return dict(
            __class__ = 'numpy.ndarray',
            data = array.tolist(),
            dtype = array.dtype.str,
        )

    def dict_to_np(classname, d):
        return getattr(np, classname[6:])(d['value'])

    def dict_to_ndarray(classname, d):
        return np.array(d['data'], dtype=d['dtype'])

    int_types = (np.int, np.int32, np.int64)
    float_types = (np.float, np.float32, np.float64)
    SerializerBase.register_class_to_dict(int_types, int_to_dict)
    SerializerBase.register_class_to_dict(float_types, float_to_dict)
    np_names = ('numpy.'+cls.__name__ for cls in int_types + float_types)
    for name in np_names:
        SerializerBase.register_dict_to_class(name, dict_to_np)

    SerializerBase.register_class_to_dict(np.ndarray, ndarray_to_dict)
    SerializerBase.register_dict_to_class('numpy.ndarray', dict_to_ndarray)


def _register_errors():
    try:
        import Queue as queue
    except ImportError:
        import queue

    exceptions = [queue.Empty]
    exceptions_names = dict((e, "".join((e.__module__,'.',e.__name__)))
                        for e in exceptions)

    def error_to_dict(error):
        return dict(__class__=exceptions_names[type(error)])
    def dict_to_error(classname, d):
        return queue.Empty

    for error in exceptions:
        SerializerBase.register_class_to_dict(error, error_to_dict)
        SerializerBase.register_dict_to_class(exceptions_names[error], dict_to_error)



# This ensures that the numpy arrays and queue exceptions can be serialized by Pyro
_register_numpy_serializers()
_register_errors()
