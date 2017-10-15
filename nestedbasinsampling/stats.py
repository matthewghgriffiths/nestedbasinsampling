
import numpy as np
import matplotlib.pyplot as plt

class AndersonDarling(object):
    """class for calculating the Anderson-Darling fit statistic and its
    significance    
    """
                                
    @staticmethod
    def calculateSignificance(A2kN, ns):
        """ Calculates the significance and critical values
        of the Anderson-Darling fit obtained from k-samples
        
        code adapted from scipy
        
        Parameters
        ----------
        A2kN : float
            The Anderson-Darling fit statistic
        ns : list of ints
            The number of observations in each sample
            
        Returns
        -------
        p : float
            An approximate significance level at which the null hypothesis 
        for the provided samples can be rejected.
        A2 : float
            The normalised Anderson-Darling fit statistic
        critical : array
            The critical values for significance levels 
        25%, 10%, 5%, 2.5%, 1%.
        """
    
        ns = np.asanyarray(ns)
        k = len(ns)
        N = sum(ns)
        
        H = (1. / ns).sum()
        hs_cs = (1. / np.arange(N - 1, 1, -1)).cumsum()
        h = hs_cs[-1] + 1
        g = (hs_cs / np.arange(2, N)).sum()

        a = (4*g - 6) * (k - 1) + (10 - 6*g)*H
        b = (2*g - 4)*k**2 + 8*h*k + (2*g - 14*h - 4)*H - 8*h + 4*g - 6
        c = (6*h + 2*g - 2)*k**2 + (4*h - 4*g + 6)*k + (2*h - 6)*H + 4*h
        d = (2*h + 6)*k**2 - 4*h*k
        sigmasq = ((a*N**3 + b*N**2 + c*N + d) / 
                   ((N - 1.) * (N - 2.) * (N - 3.)))
        m = k - 1
        A2 = (A2kN - m) / np.sqrt(sigmasq)
        
        # The b_i values are the interpolation coefficients from Table 2
        # of Scholz and Stephens 1987
        b0 = np.array([0.675, 1.281, 1.645, 1.96, 2.326])
        b1 = np.array([-0.245, 0.25, 0.678, 1.149, 1.822])
        b2 = np.array([-0.105, -0.305, -0.362, -0.391, -0.396])
        critical = b0 + b1 / np.sqrt(m) + b2 / m
        pf = np.polyfit(critical, 
                        np.log(np.array([0.25, 0.1, 0.05, 0.025, 0.01])), 2)
        
        if A2 < critical.min() or A2 > critical.max():
            # Pass warning?
            pass
            
        p = np.exp(np.polyval(pf, A2))
        
        return p, A2, critical
        
    @classmethod
    def compareDistributions(cls, cdfs, midrank=True):
        """ The Anderson-Darling test for k-samples.
        
        The k-sample Anderson-Darling test is a modification of the
        one-sample Anderson-Darling test. It tests the null hypothesis
        that k-samples are drawn from the same population without having
        to specify the distribution function of that population. The
        critical values depend on the number of samples.
        
        Parameters
        ----------
        cdfs : list of observations (either iterable or CDF)
            The list of observations of the k different samples, can
            pass the observations directly as a list, or can pass a list
            of CDF objects
            
        midrank : bool (optional)        
            Type of Anderson-Darling test which is computed. Default
            (True) is the midrank test applicable to continuous and
            discrete populations. If False, the right side empirical
            distribution is used.
            
        Returns
        -------
        p : float
            An approximate significance level at which the null hypothesis 
        for the provided samples can be rejected.
        A2 : float
            The normalised Anderson-Darling fit statistic
        critical : array
            The critical values for significance levels 
        25%, 10%, 5%, 2.5%, 1%.
        
        References
        ----------
            Scholz, F. W and Stephens, M. A. (1987), K-Sample
            Anderson-Darling Tests, Journal of the American Statistical
            Association, Vol. 82, pp. 918-924.
            
        """
        try:
            cdfs = [CDF(cdf) for cdf in cdfs]
        except TypeError:
            assert all(type(cdf) is CDF for cdf in cdfs)
        
        H = sum(cdfs[1:], cdfs[0])
        A2kn = sum(cdf.calcAndersonFit(H, midrank=midrank) for cdf in cdfs)
        return cls.calculateSignificance(A2kn, map(len, cdfs))
        
class CDF(object):
    """Class for storing and manipulating the CDF of a univariate 
    distribution.
    
    The distribution can be continuous or discrete
    
    Attributes
    ----------
    Xs : numpy.array
        sorted array of unique values
    ws : numpy.array
        the weights associated with these values
    Fs : numpy.array
        the cumulative distribution of the values
        
    Methods
    -------
    calcAndersonRight
    """

    def __init__(self, Xs, ws=None, n=None):
    
        # Checking Xs is iterable
        iter(Xs)
        
        Xs = np.asanyarray(Xs).ravel()
        argsort = Xs.argsort()
        
        self.n = len(Xs) if n is None else n
        if ws is None:
            self._ws = np.ones_like(Xs,dtype=float)/self.n
        else:
            self._ws = np.array(ws)[argsort]
        self._Xs = Xs[argsort]
        
        unique = np.r_[0,np.diff(self._Xs).nonzero()[0]+1]
        self.ws = np.array(map(lambda x:x.sum(), 
                               np.split(self._ws, unique[1:])))
        self.Xs = self._Xs[unique]
        self.Fs = np.cumsum(self.ws)
        
    def __add__(self, cdf2):
        Xs = np.r_[self._Xs, cdf2._Xs]
        n = self.n + cdf2.n
        ws = np.r_[self.n * self._ws, cdf2.n * cdf2._ws] / n
        return self.__class__(Xs, ws=ws, n=n)
        
    def __len__(self):
        return self.n
        
    def calcAndersonFit(self, H, midrank=True):
        """ Calculates the Anderson-Darling statistic between
        This CDF and a target distribution. 
        
        References
        ----------
            Scholz, F. W and Stephens, M. A. (1987), K-Sample
            Anderson-Darling Tests, Journal of the American Statistical
            Association, Vol. 82, pp. 918-924.
        """
        # H must contain all the values in Xs
        assert all(np.in1d(self.Xs, H.Xs, assume_unique=True))
        
        if midrank:
            inds = self.Xs.searchsorted(H.Xs, side='right')
            w0 = H.ws
            H0 = H.Fs - w0/2.
            F = np.r_[self.Fs,0.][inds-1]
            F -= np.where(self.Xs[inds-1] == H.Xs, 
                          np.r_[self.ws,0.][inds-1]/2., 0.)
            A2akN = self.n * ( w0 * (F-H0)**2 / (H0*(1.-H0)- w0/4.) ).sum()
        else:
            inds = np.searchsorted(self.Xs, H.Xs[:-1], side='right')
            w0 = H.ws[:-1]
            H0 = H.Fs[:-1]
            F = np.r_[self._Fs,0.][inds-1]
            A2akN = self.n * ( w0 * (F-H0)**2 / (H0*(1.-H0)) ).sum()
        
        return A2akN * (H.n - 1.) / H.n
        
    def plot(self, start=None, end=None, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        Xs, Fs = self.Xs, self.Fs
        start = 2*Xs[0] - Xs[1] if start is None else start
        end = 2*Xs[-1] - Xs[-2] if end is None else end
        Xs = np.r_[start, Xs, end]
        Fs = np.r_[0.,Fs,1.]
        step = ax.step(Xs, Fs, where='post', **kwargs)
        return ax, step
       
    
if __name__ == "__main__":
    from scipy.stats import anderson_ksamp, norm, t
    
    ns = np.array((200,20,32,13))
    N = sum(ns)
   
    print "Comparing samples from Uniform distribution"
    samples = [np.random.randint(100, size=(n,)) for n in ns]
    Fs = [CDF(Xs) for Xs in samples]
    H = sum(Fs[1:], Fs[0])
    print "pvalue = {:5.2g}".format(
        AndersonDarling.compareDistributions(Fs)[0])
    print anderson_ksamp(samples)

    print "Comparing samples from uniform integer distribution [0,100]"
    samples = [np.random.randint(100, size=(n,)) for n in ns]
    Fs = [CDF(Xs) for Xs in samples]
    H = sum(Fs[1:], Fs[0])
    print "pvalue = {:5.2g}".format(
        AndersonDarling.compareDistributions(Fs)[0])
    print anderson_ksamp(samples)
    
    print "Comparing student t distribution vs normal"
    samples = [norm.rvs(size=20), t.rvs(3, size=20)]
    Fs = [CDF(Xs) for Xs in samples]
    print "pvalue = {:5.2g}".format(
        AndersonDarling.compareDistributions(samples)[0])
    print anderson_ksamp(samples)
    Fs[0].plot(); Fs[1].plot()
    
    print "Comparing normal vs normal"
    samples = [norm.rvs(size=10), norm.rvs(size=10)]
    Fs = [CDF(Xs) for Xs in samples]
    print "pvalue = {:5.2g}".format(
        AndersonDarling.compareDistributions(samples)[0])
    print anderson_ksamp(samples)
    Fs[0].plot(); Fs[1].plot()
    
    



