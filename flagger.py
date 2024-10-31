import numpy as np
import numpy.typing as npt

from scipy.optimize import Bounds, minimize  # type: ignore
from scipy.special import expit  # type: ignore
from scipy.stats import beta, entropy, norm  # type: ignore
from sklearn import svm  # type: ignore
from sklearn.exceptions import NotFittedError  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
from tmvbeta import TMVBeta
from typing import Union

# Custom types
anyfloat = Union[float, np.float64]
anyfloat_or_array = Union[anyfloat, npt.NDArray[np.float64]]


def logdet(A: npt.NDArray[np.float64]) -> np.float64:
    """
    Log-determinant. Convinient wrapper around numpy's `slogdet` that
    returns both absolute value and sign.
    """
    return np.linalg.slogdet(A)[1]


def get_equivalent_samples(chi_a: npt.NDArray[np.float64], chi_b: npt.NDArray[np.float64]) -> np.float64:
    """
    Generate a pair of samples such the corresponding maximum likelihood estimate equals
    the maximum a posteriori estimate for the sufficient statistics `chi_a` and `chi_b`.
    """
    s_a = np.exp(2 * chi_a)
    s_b = np.exp(2 * chi_b)
    p = (1 + s_a - s_b) / 2
    return p + np.array([1, -1]) * np.sqrt(p**2 - s_a)


class PosteriorModel:
    """Parameters of posterior distributions"""

    def __init__(self, M: int):
        self.M = M
        self.a = np.ones((2, M))
        self.b = np.ones((2, M))
        self.v = np.ones(2)
        self.psi = np.array([np.eye(M), np.eye(M)])
        self.chi_a = -np.ones((2, M))
        self.chi_b = -np.ones((2, M))


class BayesFlaggerBeta:
    """
    Flagger that uses the approximate Bayes method detailed in [...]
    """

    def __init__(self, K: int, M: int, rule: str = "detection") -> None:
        """
        Initialize Bayes flagger.

        K:    Number of samples to be flagged for review
        M:    Number of features (= length of feature vectors)
        rule: Flagging rule ("detection", "information", or "mixed").
              See above paper for details
        """
        # Parameters
        self.K = K
        self.phi = 1
        self.rule = rule

        # Model and per admin variables
        self.model = PosteriorModel(M)
        self.reset(False)

    def reset(self, reset_model: bool = True) -> None:
        """Reset model and per admin variables."""
        # Estimated model
        if reset_model:
            self.model = PosteriorModel(self.model.M)

        # Per admin variables
        self.N = 0
        self.X = np.empty((0, 0), np.float64)
        self.c = np.empty(0, bool)
        self.s = np.empty(0, bool)
        self.d = np.empty(0, bool)

        # Membership posterior
        self.pc = np.empty(0, np.float64)

        # Counter
        self.n_detected = 0

    def observe(self, X: npt.NDArray[np.float64]) -> None:
        """Observe `N` feature vectors of length `M` stacked into an N x M matrix `X`."""
        self.X = np.atleast_2d(X)
        N, M = self.X.shape

        # Check feature vector length
        if M != self.model.M:
            raise ValueError(f"Feature vector length is {M}, expected {self.model.M}")

        # Reset per admin variables
        self.N = N
        self.pc = (self.model.v[1] / self.model.v.sum()) * np.ones(N)
        self.c = np.full(N, False)
        self.s = np.full(N, False)
        self.d = np.full(N, False)

    def flag(self) -> None:
        """Flag observed samples according to set rule."""
        # Sample randomly if posterior probabilities are all equal
        if self.pc.min() == self.pc.max():
            self.s[np.random.choice(range(self.N), self.K, replace=False)] = True
        else:
            # Detection-greedy policy
            if self.rule == "detection":
                self.s[np.argpartition(self.pc, -self.K)[-self.K :]] = True
            # Information-greedy policy
            elif self.rule == "information":
                h = entropy(np.stack((self.pc, 1 - self.pc)))
                self.s[np.argpartition(h, -self.K)[-self.K :]] = True
            # Mixed policy
            elif self.rule == "mixed":
                K_map = int(np.ceil(self.phi * self.K))
                self.s[np.argpartition(self.pc, -K_map)[-K_map:]] = True
                if K_map < self.K:
                    K_mi = self.K - K_map
                    pc = np.zeros_like(self.pc)
                    pc[~self.s] = self.pc[~self.s]
                    h = entropy(np.stack((pc, 1 - pc)))
                    self.s[np.argpartition(h, -K_mi)[-K_mi:]] = True
            else:
                print(f"Flagging rule '{self.rule}' is not implemented.")

    def review(self, c: npt.NDArray[np.bool_]) -> None:
        """Incorporate review outcomes for flagged cases."""
        self.c = c
        self.d = self.s * self.c
        self.n_detected += int(np.sum(self.d))

        # Upate phi if mixed policy is used (see paper for details)
        if self.rule == "mixed":
            mu = np.sum(self.pc[self.s])
            sigma = np.sqrt(np.sum(self.pc[self.s] * (1 - self.pc[self.s])))
            self.phi = 1 if mu - np.sum(self.d) <= sigma else sigma / (mu - np.sum(self.d))

    def project_cov(self, C: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert covarinace matrix to correlation matrix."""
        D = np.diag(1 / np.sqrt(np.diag(C)))
        return D @ C @ D

    def update_pc(self, model: PosteriorModel) -> npt.NDArray[np.float64]:
        """Update posterior group probabilities based on given model."""
        # Sample log-probabilities
        logprob = np.array(
            [
                TMVBeta(a, b, self.project_cov(psi)).logpdf(self.X) + np.log(v)
                for (a, b, v, psi) in zip(model.a, model.b, model.v, model.psi)
            ]
        )

        # Posterior group probabilities
        pc = expit(np.diff(logprob, axis=0).flatten())

        # Review outcomes, if any
        pc[self.s] = self.c[self.s].astype(int)

        return pc

    def update_model(self, pc: npt.NDArray[np.float64]) -> PosteriorModel:
        """Upade model based on given group probabilities."""
        model_new = PosteriorModel(self.model.M)
        Z = [np.zeros_like(self.X), np.zeros_like(self.X)]

        # Update v
        model_new.v = self.model.v + np.array([np.sum(1 - pc), np.sum(pc)])

        # Update marginals and transform observations
        for m in range(self.model.M):
            model_new.chi_a[:, m] = self.model.chi_a[:, m] + np.array([(1 - pc), pc]) @ np.log(self.X[:, m])
            model_new.chi_b[:, m] = self.model.chi_b[:, m] + np.array([(1 - pc), pc]) @ np.log(1 - self.X[:, m])
            for i in range(2):
                x_equiv = get_equivalent_samples(
                    model_new.chi_a[i, m] / model_new.v[i], model_new.chi_b[i, m] / model_new.v[i]
                )
                model_new.a[i, m], model_new.b[i, m], _, _ = beta.fit(x_equiv, floc=0, fscale=1)
                Z[i][:, m] = np.clip(norm.ppf(beta.cdf(self.X[:, m], model_new.a[i, m], model_new.b[i, m])), -5, 5)

        # Update covariance
        model_new.psi = self.model.psi + np.array([Z[0].T @ np.diag(1 - pc) @ Z[0], Z[1].T @ np.diag(pc) @ Z[1]])

        return model_new

    def update_posterior(self, update_model: bool = False, verbose: bool = False) -> None:
        """
        Jointly updade group and model posterior using fixed-point iteration.
        Store updated model if `update_model` parameter is set, discard otherwise.
        """
        # Prior group probabilities
        pc = self.update_pc(self.model)

        # Try fixed-point iteration for 2 * self.N iterations
        diff: anyfloat = 0.0
        for it in range(2 * self.N):
            # Update pc
            pc_new = self.update_pc(self.update_model(pc))

            # Get (and print) difference
            diff = np.linalg.norm(pc_new - pc, np.inf)
            if verbose:
                print(f"{it}: {diff}")

            # Update variables
            pc = pc_new

            # Check convergence
            if diff < 1e-4:
                break

        # Check accuracy of solution
        if diff > 1e-4:
            print("Fixed-point iteration did not converge, minimizing squared error instead.")
            self.update_posterior_least_squares(update_model=update_model, verbose=verbose)
        # Update group probabilities and model
        else:
            self.pc = pc
            if update_model:
                self.model = self.update_model(pc)

    def update_posterior_least_squares(self, update_model: bool = False, verbose: bool = False) -> None:
        """
        Jointly updade group and model posterior using least squares.
        Store updated model if `update_model` parameter is set, discard otherwise.
        """

        def objective(pc_no_review: npt.NDArray[np.float64]) -> np.float64:
            """
            Square error cost function for posterior group probabilities
            of unreviewed samples. Reviewed samples have prob = {0,1}.
            """
            # Combine posterior probabilities with review outcomes
            pc = np.zeros(self.N)
            pc[~self.s] = pc_no_review
            pc[self.s] = np.clip(self.c[self.s].astype(int), 0, 1)

            # Update pc
            pc_new = self.update_pc(self.update_model(pc))

            # Get (and print) difference
            diff = np.linalg.norm(pc_no_review - pc_new[~self.s])
            if verbose:
                print(diff)

            return diff

        # On second run, use current posteriors as starting point for optimization
        if update_model:
            pc0 = self.pc[~self.s]
        # On first run, use priors as starting point for optimization
        else:
            pc0 = (self.model.v[1] / self.model.v.sum()) * np.ones(np.sum(~self.s))

        # Solve least squares problem using Nelder-Mead (slow, but robsut)
        sol = minimize(objective, pc0, bounds=Bounds(0, 1), method="Nelder-Mead", options={"adaptive": True})

        # Print message if optimization failed
        if not sol.success:
            print(sol.message)

        # Combine posterior probabilities with review outcomes
        pc = np.zeros(self.N)
        pc[self.s] = self.c[self.s].astype(int)
        pc[~self.s] = np.clip(sol.x, 0, 1)

        # Store results, if requested
        self.pc = pc
        if update_model:
            self.model = self.update_model(pc)


class SVMFlagger:
    """
    Flagger based on an SVM trained on reviewed samples only. Used to establish 
    baseline performance in [...]
    """
    def __init__(self, K: int, M: int) -> None:
        """
        Initialize SVM flagger.

        K:  Number of samples to be flagged for review
        M:  Number of features (= length of feature vectors)
        """
        # Parameters
        self.M = M
        self.K = K

        # Classifier and per admin variables
        self.clf = make_pipeline(StandardScaler(), svm.SVC())
        self.reset(False)

    def reset(self, reset_classifier: bool = True) -> None:
        """Reset classifier and per admin variables."""
        # Classifier
        if reset_classifier:
            self.clf = make_pipeline(StandardScaler(), svm.SVC())

        # Reviewed samples with labels
        self.X0 = np.empty((0, self.M))
        self.X1 = np.empty((0, self.M))
        self.y0 = np.empty(0)
        self.y1 = np.empty(0)

        # Per admin variables
        self.N = 0
        self.X = np.empty((0, 0), np.float64)
        self.s = np.empty(0, bool)
        self.d0 = np.empty(0, bool)
        self.d1 = np.empty(0, bool)

        # Decision function
        self.df = np.empty(0, np.float64)

        # Counters
        self.n_detected = 0

    def observe(self, X: npt.NDArray[np.float64]) -> None:
        """Observe `N` feature vectors of length `M` stacked into an N x M matrix `X`."""
        self.X = np.atleast_2d(X)
        N, M = self.X.shape

        # Check feature vector length
        if M != self.M:
            raise ValueError(f"Feature vector length is {M}, expected {self.M}")

        # Reset per admin variables
        self.N = N
        self.s = np.full(N, False)
        self.d0 = np.full(N, False)
        self.d1 = np.full(N, False)

    def flag(self) -> None:
        """Flag observed samples most likely to belong to critical group."""
        if self.df.min() == self.df.max():
            self.s[np.random.choice(np.arange(self.N), self.K, replace=False)] = True
        else:
            self.s[np.argpartition(self.df, -self.K)[-self.K :]] = True

    def review(self, c: npt.NDArray[np.bool_]) -> None:
        """
        Incorporate review outcomes for flagged cases, and add corresponding
        samples and labels to training data set.
        """
        self.d0 = self.s * (~c)
        self.d1 = self.s * c
        self.n_detected += int(np.sum(self.d1))

        self.X0 = np.concatenate((self.X0, self.X[self.d0]))
        self.X1 = np.concatenate((self.X1, self.X[self.d1]))
        self.y0 = np.concatenate((self.y0, np.zeros(int(np.sum(self.d0)))))
        self.y1 = np.concatenate((self.y1, np.ones(int(np.sum(self.d1)))))

    def update_df(self) -> None:
        """Calculate decision function for current sample."""
        try:
            check_is_fitted(self.clf)
            self.df = self.clf.decision_function(self.X)
        except NotFittedError:
            self.df = np.zeros(self.N)

    def update_svm(self) -> None:
        """Update classifier using training data observed so far."""
        # Check that samples from both classes have been observed
        if np.minimum(self.y0.size, self.y1.size) > 0:
            X = np.concatenate((self.X1, self.X0))
            y = np.concatenate((self.y1, self.y0))
            self.clf = make_pipeline(StandardScaler(), svm.SVC())
            self.clf.fit(X, y)
