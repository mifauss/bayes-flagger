import numpy as np
import numpy.typing as npt
import pandas as pd

from flagger import anyfloat, BayesFlaggerBeta, SVMFlagger, PosteriorModel
from scipy.integrate import dblquad  # type: ignore
from scipy.special import betaln  # type: ignore
from scipy.stats import binom  # type: ignore
from tmvbeta import TMVBeta
from typing import Callable, Tuple

# Custom types
Sampler = Callable[[int], Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]]


def detection_rate_bound(N: int, K: int, r: anyfloat) -> np.float64:
    """Upper bound on detected share of critical group members []"""
    return binom.pmf(np.arange(N + 1), N, r) @ np.minimum(1, K / np.clip(np.arange(N + 1), 1, N))


def get_sampler_model(r: anyfloat, P0: TMVBeta, P1: TMVBeta) -> Sampler:
    """Construct sampler from model (beta distributed marginals with Gaussian copula)."""

    def sampler(N: int) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        n_c = binom.rvs(N, r)
        c = np.array(n_c * [True] + (N - n_c) * [False])
        X1 = np.atleast_2d(P1.rvs(size=n_c))
        X0 = np.atleast_2d(P0.rvs(size=N - n_c))
        X = np.concatenate((X1, X0))
        return c, X

    return sampler


def get_sampler_data(r: anyfloat, df0: pd.core.frame.DataFrame, df1: pd.core.frame.DataFrame) -> Sampler:
    """Construct sampler from Pandas data frames."""

    def sampler(N: int) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        n_c = binom.rvs(N, r)
        c = np.array(n_c * [True] + (N - n_c) * [False])
        X1 = df1.sample(n_c, replace=True)
        X0 = df0.sample(N - n_c, replace=True)
        X = np.concatenate((X1, X0))
        return c, X

    return sampler


class ModelError:
    """Error metrics for posterior model evaluation."""
    
    def __init__(
        self, r: anyfloat, a: npt.NDArray[np.float64], b: npt.NDArray[np.float64], cov: npt.NDArray[np.float64]
    ):
        """Initialize class with true model parameters."""
        self.r = r
        self.a = a
        self.b = b
        self.cov = cov

    def mse_r(self, model: PosteriorModel) -> np.float64:
        """MSE of critical group member share `R` for given posterior."""
        return (model.v[1] / model.v.sum()) * ((model.v[1] + 1) / (model.v.sum() + 1) - 2 * self.r) + self.r**2

    def mse_cov(self, i: int, model: PosteriorModel) -> npt.NDArray[np.float64]:
        """MSE of covariance matrix of `X_i` for given posterior."""
        c2 = 1 / ((model.v[i] - model.M) * (model.v[i] - model.M - 1) * (model.v[i] - model.M - 3))
        c1 = (model.v[i] - model.M - 2) * c2
        EC = model.psi[i] / (model.v[i] - model.M - 1)
        mse_cov = (
            (c1 + c2) * np.trace(model.psi[i] @ model.psi[i])
            + c2 * np.trace(model.psi[i]) ** 2
            - 2 * np.trace(EC @ self.cov[i])
            + np.trace(self.cov[i] @ self.cov[i])
        )
        return mse_cov

    def mse_beta(self, i: int, m: int, model: PosteriorModel) -> np.float64:
        """MSE of beta marginal parameters for given posterior."""
        chi0, chi1 = model.chi_a[i, m], model.chi_b[i, m]

        def unscaled_pdf(eta0, eta1):
            return np.exp(eta0 * chi0 + eta1 * chi1 - model.v[i] * betaln(eta0, eta1))

        def weighted_error(eta0, eta1):
            return ((self.a[i, m] - eta0) ** 2 + (self.b[i, m] - eta1) ** 2) * unscaled_pdf(eta0, eta1)

        try:
            p_mass = dblquad(unscaled_pdf, 0, np.inf, 0, np.inf)[0]
            mse_beta = dblquad(weighted_error, 0, np.inf, 0, np.inf)[0] / p_mass
        except ZeroDivisionError:
            print("MSE could not be evaluated numerically, try 'map_square_error_beta' instead.")
            mse_beta = None

        return mse_beta

    def map_square_error_r(self, model: PosteriorModel) -> np.float64:
        """Square error of MAP estimate of `R` for given posterior."""
        return (model.v[1] / model.v.sum() - self.r) ** 2

    def map_square_error_cov(self, i: int, model: PosteriorModel) -> np.float64:
        """Square error of MAP estimate of covarinace matrix of `X_i` for given posterior."""
        return np.linalg.norm(model.psi[i] / (model.v[i] + model.M + 1) - self.cov[i]) ** 2

    def map_square_error_beta(self, a: anyfloat, b: anyfloat, i: int, m: int, model: PosteriorModel) -> np.float64:
        """Square error of MAP estimate of beta marginal parameters for given posterior."""
        return (self.a[i, m] - model.a[i, m]) ** 2 + (self.b[i, m] - model.b[i, m])


class BayesFlaggerBetaTest:
    """Wrapper class to simulate BayesFlagger."""

    def __init__(self, flagger: BayesFlaggerBeta, sampler: Sampler, N: int) -> None:
        """Initialize test with given `flagger`, `sampler` and sample size `N`."""
        self.flagger = flagger
        self.sampler = sampler
        self.N = N

    def run(self, T: int, verbose: bool = False):
        """Run simulation for `T` admins and log results."""
        # Reset flagger
        self.flagger.reset()

        # Track model
        model = [self.flagger.model]

        # Track detections
        n_total = [0]
        n_detected = [0]

        # Track phi parameter (only relevant for mixed policy)
        phi = [1]

        for t in range(T):
            if verbose:
                print(f"t = {t}")

            c, X = self.sampler(self.N)
            self.flagger.observe(X)
            self.flagger.update_posterior(update_model=False)
            self.flagger.flag()
            self.flagger.review(c)
            self.flagger.update_posterior(update_model=True)

            model.append(self.flagger.model)
            n_total.append(n_total[-1] + int(np.sum(c)))
            n_detected.append(self.flagger.n_detected)
            phi.append(self.flagger.phi)

        return np.array(n_total), np.array(n_detected), model, np.array(phi)


class SVMFlaggerTest:
    """Wrapper class to simulate SVMFlagger."""

    def __init__(self, flagger: SVMFlagger, sampler: Sampler, N: int) -> None:
        """Initialize test with given `flagger`, `sampler` and sample size `N`."""
        self.flagger = flagger
        self.sampler = sampler
        self.N = N

    def run(self, T: int, verbose: bool = False):
        """Run simulation for `T` admins and log results."""
        # Reset flagger
        self.flagger.reset()

        # Track detections
        n_total = [0]
        n_detected = [0]

        for t in range(T):
            if verbose:
                print(f"t = {t}")

            c, X = self.sampler(self.N)
            self.flagger.observe(X)
            self.flagger.update_df()
            self.flagger.flag()
            self.flagger.review(c)
            self.flagger.update_svm()

            n_total.append(n_total[-1] + int(np.sum(c)))
            n_detected.append(self.flagger.n_detected)

        return np.array(n_total), np.array(n_detected)
