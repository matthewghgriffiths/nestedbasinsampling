# -*- coding: utf-8 -*-

# Code modified from scikit-learn
# Author: Wei Xue <xuewei4d@gmail.com>
# Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
# License: BSD 3 clause

import warnings

import numpy as np
from scipy.special import gammaln, betaln, digamma

from sklearn import cluster
from sklearn.mixture.base import BaseMixture
#from sklearn.mixture import BayesianGaussianMixture
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state
from sklearn.exceptions import ConvergenceWarning

def _log_dirichlet_norm(dirichlet_concentration):
    """Compute the log of the Dirichlet distribution normalization term.
    Parameters
    ----------
    dirichlet_concentration : array-like, shape (n_samples,)
        The parameters values of the Dirichlet distribution.
    Returns
    -------
    log_dirichlet_norm : float
        The log normalization of the Dirichlet distribution.
    """
    return (gammaln(np.sum(dirichlet_concentration)) -
            np.sum(gammaln(dirichlet_concentration)))

def _estimate_beta_parameters(ab, resp):
    """
    Parameters
    ----------
    ab : array-like, shape (n_samples, n_features, 2)
        The input data array.

    resp : array-like, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    Returns
    -------
    nk : array-like, shape (n_components,)
        The numbers of data samples in the current components.

    betas : array-like, shape (n_components, n_features)
        The centers of the current components.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    betas = np.einsum("ij,ikl->jkl", resp, ab)
    ratios = betas[...,0]/betas.sum(2)
    return nk, ratios, betas

def _estimate_log_beta_prob(ab, ratios):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    ab : array-like, shape (n_samples, n_features, 2)

    ratios : array-like, shape (n_components, n_features)

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """

    return (- betaln(*ab.T).T[:,None,:]
            + (ab[:,None,:,0]-1) * np.log(ratios[None,:,:])
            + (ab[:,None,:,1]-1) * np.log1p(- ratios[None,:,:])).sum(2)

class BayesianBetaMixture(BaseMixture):

    def __init__(self, n_components=1, tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1,
                 init_params='kmeans',
                 weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=None,
                 beta_prior=1.,
                 random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10):

        super(BayesianBetaMixture, self).__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.beta_prior = beta_prior

    def _check_weights_parameters(self):
        """Check the parameter of the Dirichlet distribution."""
        if self.weight_concentration_prior is None:
            self.weight_concentration_prior_ = 1. / self.n_components
        elif self.weight_concentration_prior > 0.:
            self.weight_concentration_prior_ = (
                self.weight_concentration_prior)
        else:
            raise ValueError("The parameter 'weight_concentration_prior' "
                             "should be greater than 0., but got %.3f."
                             % self.weight_concentration_prior)

    def _check_parameters(self, X):
        """Check that the parameters are well defined.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        if (self.weight_concentration_prior_type not in
                ['dirichlet_process', 'dirichlet_distribution']):
            raise ValueError(
                "Invalid value for 'weight_concentration_prior_type': %s "
                "'weight_concentration_prior_type' should be in "
                "['dirichlet_process', 'dirichlet_distribution']"
                % self.weight_concentration_prior_type)

        self._check_weights_parameters()
        self._check_beta_prior()

    def _check_beta_prior(self):
        if self.beta_prior <= 0.:
            raise ValueError("The parameter 'beta_prior' "
                             "should be greater than 0., but got %.3f."
                             % self.beta_prior)
    def _check_is_fitted(self):
        check_is_fitted(self, ['weight_concentration_', 'ratios_'])

    def _estimate_weights(self, nk):
        """Estimate the parameters of the Dirichlet distribution.

        Parameters
        ----------
        nk : array-like, shape (n_components,)
        """
        if self.weight_concentration_prior_type == 'dirichlet_process':
            # For dirichlet process weight_concentration will be a tuple
            # containing the two parameters of the beta distribution
            self.weight_concentration_ = (
                1. + nk,
                (self.weight_concentration_prior_ +
                 np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))))
        else:
            # case Variationnal Gaussian mixture with dirichlet distribution
            self.weight_concentration_ = self.weight_concentration_prior_ + nk

    def _estimate_betas(self, nk, bk):
        """Estimate the parameters of the Dirichlet distribution.

        Parameters
        ----------
        nk : array-like, shape (n_components,)
        bk : array-like, shape (n_components, n_features, 2)
        """
        betas = bk + self.beta_prior
        self.ratios_ = betas[:,:,0]/betas.sum(2)

    def _initialize(self, X, resp):
        """Initialization of the mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        self._m_step(X, np.log(resp))

    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features, 2)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        nk, rk, bk = _estimate_beta_parameters(X, np.exp(log_resp))
        self._estimate_weights(nk)
        self._estimate_betas(nk, bk)

    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            Logarithm of the probability of each sample in X.

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return log_prob_norm, log_resp

    def _estimate_log_weights(self):
        if self.weight_concentration_prior_type == 'dirichlet_process':
            digamma_sum = digamma(self.weight_concentration_[0] +
                                  self.weight_concentration_[1])
            digamma_a = digamma(self.weight_concentration_[0])
            digamma_b = digamma(self.weight_concentration_[1])
            return (digamma_a - digamma_sum +
                    np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1])))
        else:
            # case Variationnal Gaussian mixture with dirichlet distribution
            return (digamma(self.weight_concentration_) -
                    digamma(np.sum(self.weight_concentration_)))

    def _estimate_log_prob(self, X):
        log_pcount = _estimate_log_beta_prob(X, self.ratios_)
        return log_pcount

    def _compute_lower_bound(self, log_resp, log_prob_norm):
        """Estimate the lower bound of the model.
        The lower bound on the likelihood (of the training data with respect to
        the model) is used to detect the convergence and has to decrease at
        each iteration.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        log_prob_norm : float
            Logarithm of the probability of each sample in X.
        Returns
        -------
        lower_bound : float
        """
        # Contrary to the original formula, we have done some simplification
        # and removed all the constant terms.

        prior = np.array([self.beta_prior, self.beta_prior])[None,None,:]
        log_beta = _estimate_log_beta_prob(prior, self.ratios_).sum()

        if self.weight_concentration_prior_type == 'dirichlet_process':
            log_norm_weight = -np.sum(betaln(self.weight_concentration_[0],
                                             self.weight_concentration_[1]))
        else:
            log_norm_weight = _log_dirichlet_norm(self.weight_concentration_)

        return (- np.sum(np.exp(log_resp) * log_resp) + log_prob_norm.sum()
                - log_beta - log_norm_weight)

    def _get_parameters(self):
        return (self.weight_concentration_, self.ratios_)

    def _set_parameters(self, params):
        (self.weight_concentration_, self.ratios_) = params

        # Weights computation
        if self.weight_concentration_prior_type == "dirichlet_process":
            weight_dirichlet_sum = (self.weight_concentration_[0] +
                                    self.weight_concentration_[1])
            tmp = self.weight_concentration_[1] / weight_dirichlet_sum
            self.weights_ = (
                self.weight_concentration_[0] / weight_dirichlet_sum *
                np.hstack((1, np.cumprod(tmp[:-1]))))
            self.weights_ /= np.sum(self.weights_)
        else:
            self. weights_ = (self.weight_concentration_ /
                              np.sum(self.weight_concentration_))

    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)

        random_state: RandomState
            A random number generator instance.
        """
        n_samples = X.shape[0]

        if self.init_params == 'kmeans':
            resp = np.zeros((n_samples, self.n_components))
            m = X[:,:,0]/X.sum(2)
            label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                                   random_state=random_state).fit(m).labels_
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == 'random':
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            try:
                resp = np.zeros((n_samples, self.n_components))
                resp[np.arange(n_samples), self.init_params] = 1.
            except Exception as e:
                if isinstance(self.init_params, basestring):
                    raise ValueError("Unimplemented initialization method '%s'"
                                     % self.init_params)
                else:
                    raise e

        self._initialize(X, resp)

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        The method fit the model `n_init` times and set the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a `ConvergenceWarning` is raised.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not(self.warm_start and hasattr(self, 'converged_'))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)
                self.lower_bound_ = -np.infty

            for n_iter in range(self.max_iter):
                prev_lower_bound = self.lower_bound_

                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)
                self.lower_bound_ = self._compute_lower_bound(
                    log_resp, log_prob_norm)

                change = self.lower_bound_ - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(self.lower_bound_)

            if self.lower_bound_ > max_lower_bound:
                max_lower_bound = self.lower_bound_
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn('Initialization %d did not converged. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          % (init + 1), ConvergenceWarning)

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter

        return self

    def label_likelihoods(self, X, labels):
        """
        """
        save_params = self.init_params
        self.init_params = labels
        random_state = check_random_state(self.random_state)
        self._initialize_parameters(X, random_state)
        self.init_params = save_params

        log_prob_norm, log_resp = self._e_step(X)
        self._m_step(X, log_resp)
        return self._compute_lower_bound(log_resp, log_prob_norm)


    def predict(self, X, y=None):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        self._check_is_fitted()
        return self._estimate_weighted_log_prob(X).argmax(axis=1)
