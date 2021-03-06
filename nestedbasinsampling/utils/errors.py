

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

class SamplingError(Exception):
    """
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return str(self.args) + "\n" + str(self.kwargs)

class GraphError(Exception):
    """
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        if self.kwargs.keys():
            m = max(map(len, self.kwargs.keys())) + 1
            kwstr = "\n" + '\n'.join([k.rjust(m) + ': ' + repr(v)
                                      for k, v in self.kwargs.iteritems()])
        else:
            kwstr = ''
        return '\n'.join(str(v) for v in self.args) + kwstr
