
import numpy as np

class KalmanFilter(object):
    """
    """
    def predict_state(self, X, P):
        X_pred = self.F.dot(X)
        P_pred = self.F.dot(P.dot(self.F.T)) + self.Q
        return X_pred, P_pred

    def update_prediction(self, z, R, X_pred, P_pred):
        z_pred = self.H.dot(X_pred)
        R_pred = self.H.dot(P_pred.dot(self.H.T)) + R

        K = P_pred.dot(self.H.T.dot(np.linalg.pinv(R_pred)))
        X_updated = X_pred + K.dot(z - z_pred)
        P_updated = P_pred - K.dot(self.H.dot(P_pred))
        return X_updated, P_updated, K

    def filter(self, z_new, R_new):
        X_p, P_p = self.predict_state(self.X, self.P)
        X, P, self.K = self.update_prediction(z_new, R_new, X_p, P_p)
        self.X, self.P = X, P
        return X, P

    def forecast(self, n=1, X=None, P=None):
        X = self.X if X is None else X
        P = self.P if P is None else P
        for _ in range(n):
            X, P = self.predict_state(X, P)
        return X, P


class LinearKalmanFilter(KalmanFilter):
    """
    """
    def __init__(self, z=None, R=None, ndim=None, Q=None, X=None, P=None):
        self.ndim = None
        z = ((z if z is None else np.atleast_1d(z))
             if ndim is None else np.zeros(ndim))
        self.set_initial_state(z=z, R=R, X=X, P=P)
        # Transition Covariance
        self.Q = 1e-2*np.eye(2*self.ndim) if Q is None else np.array(Q)

    def set_initial_state(self, z=None, R=None, X=None, P=None):
        if X is None:
            self.X = np.r_[np.array(z, dtype=float), np.zeros_like(z)]
        else:
            self.X = np.array(X)
        ndim = len(self.X)//2

        if P is None:
            if R is None:
                self.P = np.eye(2*ndim)
            else:
                self.P = np.zeros((2*ndim, 2*ndim))
                self.P[:ndim, :ndim] = R
                self.P[ndim:, ndim:] = R
        else:
            self.P = np.array(P)

        assert self.X.shape == (2*ndim, )
        assert self.P.shape == (2*ndim, 2*ndim)

        self.z = self.X[:ndim]
        self.R = self.P[:ndim, :ndim]
        if ndim != self.ndim:
            self._set_matrices(ndim)

    def _set_matrices(self, ndim):
        self.ndim = ndim
        self.ndim_state = 2*ndim
        self.F = np.eye(2*ndim)
        n = np.arange(ndim)
        self.F[n, n + ndim] = 1.
        self.H = np.zeros((ndim, 2*ndim))
        self.H[n, n] = 1.

    def __call__(self, z, R=None):
        R = self.R if R is None else R
        R = np.diag(R) if np.ndim(R) == 1 else np.atleast_2d(R)
        sz, sR, ndim = np.shape(z), np.shape(R), self.ndim
        X, P = self.filter(np.atleast_1d(z), R)
        self.z, self.R = X[:ndim], P[:ndim, :ndim]
        if sz == ():
            return self.z[0], self.R[0,0]
        else:
            return self.z.reshape(sz), self.R.reshape(sR)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.ion()

    n = 100
    x = np.exp(np.linspace(0, 2, n))
    std = 0.1
    P = std**2 * np.ones_like(x)
    noise = np.random.normal(scale=std, size=n)
    Z = x + noise

    x_hat = np.zeros_like(x)
    v_hat = np.zeros_like(x)
    p_hat = np.zeros_like(x)
    kz = LinearKalmanFilter(z=x[0], R=P[0])

    for i, (z, R) in enumerate(zip(Z, P)):
        x_hat[i], p_hat[i] = kz(z, R*1000**0.5)

    self = kz
    z_new, R_new = z, R

    f, axes = plt.subplots(2)
    axes[0].plot(x)
    axes[0].scatter(np.arange(n), Z, marker='+')
    axes[0].errorbar(np.arange(n), y=x_hat, yerr=p_hat**0.5)

    axes[1].plot((x_hat-x)/p_hat**0.5)
