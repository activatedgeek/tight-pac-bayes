
import numpy as np
from scipy import optimize


def _bernoulli_inv_laplace(a, q):
    return np.expm1(-a * q) / np.expm1(-a)


def pac_bayes_bound(divergence, train_error, n, gamma, epsilon=0.05):
    """ This function computes a simple pac-bayes bound for generalization.

    Parameters
    ----------
    divergence: The KL-divergence between the posterior and the prior.
    train_error: The training error of the posterior.
    n: The number of samples.
    gamma: A hyperparameter trading off between the KL-divergence and the training error.
    epsilon: The probability with which the returned bound holds.

    Returns
    -------
    A 1 - epsilon probability upper bound on the testing error.
    """
    gamma = np.exp(gamma)

    q = train_error + (divergence - np.log(epsilon)) / gamma
    return _bernoulli_inv_laplace(gamma / n, q)


def pac_bayes_bound_opt(divergence, train_error, n, alpha=1e-4, epsilon=0.05):
    """ This function computes a pac-bayes bound for generalization optimizing
        over the gamma parameter and applying the correct union bound.

    Parameters
    ----------
    divergence: The KL-divergence between the posterior and the prior.
    train_error: The training error of the posterior.
    n: The number of samples.
    alpha: A hyperparameter trading off the optimization parameters. This value
        usually has very little impact on the result.
    epsilon: The probability with which the returned bound holds.

    Returns
    -------
    A 1 - epsilon probability upper bound on the testing error.
    """
    inv_log_alpha = 1 / np.log1p(alpha)
    log_eps = np.log(epsilon)

    def bound(g):
        exp_g = np.exp(g)
        q = train_error + ((1 + alpha) / exp_g) * (divergence -
                                                   log_eps + 2 * np.log(2 + g * inv_log_alpha))
        return _bernoulli_inv_laplace(exp_g / n, q)

    result = optimize.minimize_scalar(
        bound,
        (0, 4 + np.log(n))
    )

    return result.fun


def compute_convexity_bound(train_error, div, sample_size, epsilon=0.05):
    def bound(lam):
        output = train_error / (1 - 0.5 * lam)
        num = div + np.log(1. / epsilon) + np.log(2. * np.sqrt(sample_size))
        denom = sample_size * lam * (1 - 0.5 * lam)
        output += num / denom
        return output

    # result = optimize.minimize_scalar(bound, (0., 2.))
    # lams = np.linspace(0.0 + 1.e-10, 1. - 1.e-10, num=1000)
    # results = np.zeros(len(lams))
    # for i in range(len(lams)):
    #     results[i] = bound(lams[i])
    # result = np.min(results)
    a = 2. * sample_size * train_error
    b = div + np.log(1. / epsilon) + np.log(2. * np.sqrt(sample_size))
    lam = 2. / ((np.sqrt(a / b + 1.)) + 1.)
    result = bound(lam)

    return result


def compute_mcallester_bound(train_error, div, sample_size, epsilon=0.05):
    # num = div - np.log(epsilon) + (5. / 2.) * np.log(sample_size) + 8.  # Theta inf
    num = div - np.log(epsilon) + np.log(sample_size)  # Theta finite
    bound = train_error
    bound += np.sqrt(num / (2. * sample_size - 1.))
    return bound


def compute_catoni_bound(train_error, divergence, sample_size, epsilon=0.05):
    log_eps = np.log(epsilon)

    def bound_fn(g):
        # g = np.log1p(np.exp(g))
        g = g
        q = train_error
        q += (1. / g) * (divergence - log_eps)
        return _bernoulli_inv_laplace(g / sample_size, q)

    # result = optimize.minimize_scalar(bound_fn, (0., 100.))
    result = optimize.minimize_scalar(bound_fn, (1., 100.))
    return result.fun
