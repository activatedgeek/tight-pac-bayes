import numpy as np
from scipy import optimize
from scipy.special import logsumexp
from functools import partial

try:
    # For speed use the random_intel RNG if available.
    import numpy.random_intel as random
except ImportError:
    import numpy.random as random

_standard_gaussian_entropy = 0.5 * np.log(2 * np.pi) + 0.5


def _ensure_sample(n_or_samples):
    n_or_samples = np.asarray(n_or_samples)

    if len(n_or_samples.squeeze().shape) == 0:
        return random.randn(n_or_samples)
    else:
        return n_or_samples


def normal_kl_divergence(log_density, mu=0, sigma=1,
                         n_or_samples=5000,
                         return_variance=False):
    """ Computes the KL-divergence of the given log-density with respect
    to a random normal variate with given mean and standard deviation.

    The computation is carried out by monte-carlo sampling of the base
    normal measure.

    Parameters
    ----------
    log_density: The log-density of the probability for which to compute
        the KL-divergence.
    mu: The mean or means of the normal for which to obtain the random
        variables.
    sigma: The stdandard deviation of the base measure.
    n_or_samples: The number of monte-carlo samples to use, or an array
        containing a sample of random normal variates.
    return_variance: If true, also compute the variance of the KL-divergence
        estimate.

    Returns
    -------
    kl_divergence: A scalar or an array containing the kl-divergence
        of the given random variables.
    """
    randn = _ensure_sample(n_or_samples)

    if not np.alltrue(sigma > 0):
        raise ValueError('Sigma must be positive!')

    params = np.broadcast(np.atleast_1d(mu), np.atleast_1d(sigma))
    result = np.empty(params.shape)
    result_var = np.empty(params.shape)

    idx = params.index

    for m, s in params:
        ld = log_density(randn * s + m)
        cross_entropy = -np.mean(ld)
        result[idx] = cross_entropy - _standard_gaussian_entropy - np.log(s)
        if return_variance:
            result_var[idx] = np.var(ld) / len(randn)
        idx = params.index

    if return_variance:
        return np.squeeze(result), np.squeeze(result_var)
    else:
        return np.squeeze(result)


def normal_mixture_log_density(x, mu, sigma=1, weights=1):
    """Computes a normal mixture log-density at the given points.

    Note that by default, the weights are set to 1, so that the
    mixture is not a probability distribution.

    Parameters
    ----------
    x: The point or points at which to evaluate the mixture.
    mu: The location of the mixture components.
    sigma: The scale of the mixture components.
    weights: The weights of the mixture components.

    Returns
    -------
    An array of the same shape as x containing the log-density.
    """
    x = np.asarray(x)
    if x.dtype != np.float32 or x.dtype != np.float64:
        x = x.astype(np.float64)

    result = np.empty_like(x)

    x_flat = np.reshape(x.flat, [-1, 1])

    mu, sigma = np.broadcast_arrays(mu, sigma)
    dist = np.square((x_flat - mu[np.newaxis, ...]) / sigma[np.newaxis, ...])
    const = -0.5 * np.log(2 * np.pi) - np.log(sigma)

    component_log_density = -0.5 * dist + const[np.newaxis, ...]

    result.flat = logsumexp(component_log_density, b=weights, axis=tuple(range(1, len(dist.shape))))

    return result


def divergence_gains(x, scale_posterior=1.0, scale_prior=1.0,
                     rescale_posterior=True,
                     scale_prior_by_x=True,
                     n_or_samples=5000,
                     return_std=True):
    """ Compute the divergence gains of the mixture distribution compared
    to the the simple atomic distribution.

    Parameters
    ----------
    x: The value of the non-zero parameters.
    scale_posterior: The scale of the posterior distribution.
    scale_prior: The scale of the prior distribution.
    rescale_posterior: Whether to scale the posterior scale by the
        natural scale of the values in x.
    scale_prior_by_x: Whether to scale the prior scale by the natural
        scale of the values in x.
    n_or_samples: The number of samples to use in Monte-Carlo estimation,
        or an array of random normal variates to use.
    return_std: Whether to additionally return the standard deviation
        of the Monte Carlo estimate.

    Returns
    -------
    A number representing the gain in divergence in nats.
    """
    x_values, counts = np.unique(x, return_counts=True)

    if scale_prior_by_x:
        # TRY WITH max-min
        scale_prior_x = np.diff(x_values)[0]
        scale_prior *= scale_prior_x

    if rescale_posterior:
        scale_posterior_x = np.max(np.abs(x_values)) - np.min(np.abs(x_values))
        scale_posterior *= scale_posterior_x

    # if 0.0 not in x_values:
    #     x_values = np.concatenate((x_values, [0]))
    #     counts = np.concatenate((counts, [0]))

    # log_density = partial(normal_mixture_log_density, mu=x_values, sigma=scale_posterior)

    # kl_div = normal_kl_divergence(log_density, x_values, scale_prior,
    #                               n_or_samples, return_variance=return_std)
    log_density = partial(normal_mixture_log_density, mu=x_values, sigma=scale_prior)

    kl_div = normal_kl_divergence(log_density, x_values, scale_posterior,
                                  n_or_samples, return_variance=return_std)
    if return_std:
        return np.vdot(counts, kl_div[0]), np.sqrt(np.vdot(counts, kl_div[1]))
    else:
        return np.vdot(counts, kl_div)


def divergence_gains_opt(x, scale_posterior, rescale_posterior=True,n_or_samples=5000):
    """ Computes the divergence gains of the mixture distribution compared
    to the simple atomic distribution, while optimizing over the scale of the
    prior distribution. Note that this is an incorrect bound and will require
    a union bound argument to correct.
    """
    randn = _ensure_sample(n_or_samples)

    def fn(scale_prior):
        return divergence_gains(
            x, scale_posterior, np.exp(scale_prior),
            rescale_posterior=rescale_posterior,
            scale_prior_by_x=True,
            n_or_samples=randn,
            return_std=False)

    opt_result = optimize.minimize_scalar(
        fn, bounds=[-6, 1], method='bounded')

    return divergence_gains(
        x, scale_posterior,
        scale_prior=np.exp(opt_result.x), n_or_samples=n_or_samples)




def get_gains(weights, scale_posterior, rescale_posterior, scale_prior=None, n_or_samples=5000):
    """ Approximation of the KL divergence in bits
    """

    if scale_prior is None:
        gain_nats, std_nats = divergence_gains_opt(
            weights,
            scale_posterior,
            rescale_posterior=rescale_posterior,
            n_or_samples=n_or_samples)
    else:

        gain_nats, std_nats = divergence_gains_opt(
            weights,
            rescale_posterior=rescale_posterior,
            scale_prior=scale_prior,
            scale_posterior=scale_posterior)

    ## NOTE: 32 bits for prior scale optimization.
    gain_bits = -(gain_nats / np.log(2) + 32)
    std_bits = std_nats / np.log(2)

    return gain_bits, std_bits
    
    
