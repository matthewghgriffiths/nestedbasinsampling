# -*- coding: utf-8 -*-

import copy
from bisect import bisect_left, bisect_right

import inspect
from weakref import WeakSet, WeakKeyDictionary

import numpy as np
from scipy.special import gamma

def hyperspherevol(n, R=1.):
    return np.pi**(0.5*n) * R**n / gamma(0.5*n + 1)

class Signal(object):
    """ class for signal slot concept

    Example
    -------

    A simple example for a callback is
    >>> event = Signal()
    >>> event.connect(mfunc)
    >>> # raise the signal
    >>> event("hello")
    >>>
    >>> # functions can be disconnected
    >>> even.disconnect(myfunc)

    Since weak references are used, care has to be taken with object functions

    >>> obj = MyClass()
    >>> event.connect(obj.myfunc) # works
    >>> event.connect(MyClass().myfunc) # will not work

    The second example for member functions will not work since the Signal class
    uses weakref and therefore does not increase the reference counter. MyClass()
    only exists for the time of the function call and will be deleted afterwards
    and the weakref will become invalid.

    """

    def __init__(self):
        self._functions = WeakSet()
        self._methods = WeakKeyDictionary()

    def __call__(self, *args, **kargs):
        """ raise the event """
        # Call handler functions
        for func in self._functions:
            func(*args, **kargs)

        # Call handler methods
        for obj, funcs in self._methods.items():
            for func in funcs:
                func(obj, *args, **kargs)

    def connect(self, slot):
        """ connect a function / member function to the signal """
        if inspect.ismethod(slot):
            if slot.__self__ not in self._methods:
                self._methods[slot.__self__] = set()

            self._methods[slot.__self__].add(slot.__func__)

        else:
            self._functions.add(slot)

    def disconnect(self, slot):
        """ disconnect a function from the signal """
        if inspect.ismethod(slot):
            if slot.__self__ in self._methods:
                self._methods[slot.__self__].remove(slot.__func__)
        else:
            if slot in self._functions:
                self._functions.remove(slot)

    def clear(self):
        """ remove all callbacks from the signal """
        self._functions.clear()
        self._methods.clear()


class Replica(object):
    """object to represent the state of a system

    also attached is some additional information

    Parameters
    ----------
    x : array
        the structural coordinates
    energy : float
        the energy of the structure
    niter : int
        the number of MC iterations this structure has already been through
    from_random : bool
        if true, this replica started life as a completely random configuration
    """
    def __init__(self, x, energy, niter=0, from_random=True):
        self.x = x.copy()
        self.energy = float(energy)
        self.niter = niter
        self.from_random = from_random

    def copy(self):
        """return a complete copy of self"""
        return copy.deepcopy(self)

    def __repr__(self):
        repstr = "Replica(x.shape={},energy=({:10.12g}))"
        return repstr.format(repr(self.x.shape), self.energy)



class Result(dict):
    """A container for the return values of an optimizer

    Attributes
    ----------
    initialcoords : ndarray
        The starting point of the optimization.
    coords : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    energy : ndarray
        energy at the solution
    grad : ndarray
        gradient at the solution
    Emax :
        list of energies sampled

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver.

    Also, since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


    def __repr__(self):
        if self.keys():
            m = max(map(len, self.keys())) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.iteritems()])
        else:
            return self.__class__.__name__ + "()"

class NestedSamplingError(Exception):
    """
    The exception to return if there is a problem with nested sampling.
    """
    reprstate = "Energy cutoff {:10.12g}, Replica Energy {:10.12g} "
    def __init__(self, Ecut, Enew, message=""):
        self.Ecut = Ecut
        self.Enew = Enew
        self.message = ""

    def __repr__(self):
        return self.reprstate.format(self.Ecut, self.Enew) + self.message

    def __str__(self):
        return repr(self)

def dict_update_keep(toupdate, update):
    """
    updates dict toupdate with values from dict update, except if key already
    present in toupdate
    """
    toupdate.update(pair for pair in update.iteritems()
                    if pair[0] not in toupdate)
    return toupdate

def call_counter(func):
    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)
    helper.calls = 0
    helper.__name__= func.__name__
    return helper

class SortedCollection(object):

    def __init__(self, iterable=(), key=None):

        key = (lambda x: x) if key is None else key
        self._key = key
        decorated = sorted((self._key(item), item) for item in iterable)
        self._keys = [k for k, item in decorated]
        self._items = [item for k, item in decorated]

    def _getkey(self):
        return self._key

    def _setkey(self, key):
        if key is not self._key:
            self.__init__(self._items, key=key)

    def _delkey(self):
        self._setkey(None)

    key = property(_getkey, _setkey, _delkey, 'key function')

    def clear(self):
        self.__init__([], self._key)

    def copy(self):
        return self.__class__(self, self._key)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __iter__(self):
        return iter(self._items)

    def __reversed__(self):
        return reversed(self._items)

    def __repr__(self):
        return '%s(%r, key=%s)' % (
            self.__class__.__name__,
            self._items,
            getattr(self._given_key, '__name__', repr(self._given_key))
        )

    def __reduce__(self):
        return self.__class__, (self._items, self._given_key)

    def __contains__(self, item):
        k = self._key(item)
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return item in self._items[i:j]

    def index(self, item):
        'Find the position of an item.  Raise ValueError if not found.'
        k = self._key(item)
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return self._items[i:j].index(item) + i

    def count(self, item):
        'Return number of occurrences of item'
        k = self._key(item)
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return self._items[i:j].count(item)

    def insert(self, item):
        'Insert a new item.  If equal keys are found, add to the left'
        k = self._key(item)
        i = bisect_left(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, item)

    def insert_right(self, item):
        'Insert a new item.  If equal keys are found, add to the right'
        k = self._key(item)
        i = bisect_right(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, item)

    def remove(self, item):
        'Remove first occurence of item.  Raise ValueError if not found'
        i = self.index(item)
        del self._keys[i]
        del self._items[i]

    def find(self, k):
        'Return first item with a key == k.  Raise ValueError if not found.'
        i = bisect_left(self._keys, k)
        if i != len(self) and self._keys[i] == k:
            return self._items[i]
        raise ValueError('No item found with key equal to: %r' % (k,))

    def find_le(self, k):
        'Return last item with a key <= k.  Raise ValueError if not found.'
        i = bisect_right(self._keys, k)
        if i:
            return self._items[i-1]
        raise ValueError('No item found with key at or below: %r' % (k,))

    def find_lt(self, k):
        'Return last item with a key < k.  Raise ValueError if not found.'
        i = bisect_left(self._keys, k)
        if i:
            return self._items[i-1]
        raise ValueError('No item found with key below: %r' % (k,))

    def find_ge(self, k):
        'Return first item with a key >= equal to k.  Raise ValueError if not found'
        i = bisect_left(self._keys, k)
        if i != len(self):
            return self._items[i]
        raise ValueError('No item found with key at or above: %r' % (k,))

    def find_gt(self, k):
        'Return first item with a key > k.  Raise ValueError if not found'
        i = bisect_right(self._keys, k)
        if i != len(self):
            return self._items[i]
        raise ValueError('No item found with key above: %r' % (k,))