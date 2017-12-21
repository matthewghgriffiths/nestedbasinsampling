# -*- coding: utf-8 -*-

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
