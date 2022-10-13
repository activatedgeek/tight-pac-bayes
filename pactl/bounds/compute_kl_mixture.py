import numpy as np
import logging
from scipy import optimize
from scipy import integrate
from scipy.special import logsumexp
from scipy.stats import norm


def get_gains(theta_hat, posterior_scale, sample_size=int(1.0e3)):
    # gain_nats, _ = optimize_kl_div(theta_hat, posterior_scale, sample_size)
    gain_nats, _ = optimize_kl_div_ad(theta_hat, posterior_scale, sample_size)
    if gain_nats > 0:
        logging.error(f'KL ended up {gain_nats}')
    gain_nats = np.clip(gain_nats, a_min=-np.inf, a_max=0.)
    # NOTE: last term is penalty for prior grid search
    gain_bits = (gain_nats / np.log(2) + np.log2(int(5*1.0e3)))
    return gain_bits


def optimize_kl_div(theta_hat, posterior_scale, sample_size=int(1.0e3)):

    def fn(scale_prior):
        return compute_kl_div_mc(
            theta_hat, posterior_scale, np.exp(scale_prior), sample_size=sample_size
        )

    opt_result = optimize.minimize_scalar(
        fn, bounds=[-6, 1], method='bounded')
    kl_best = compute_kl_div_mc(
        theta_hat, posterior_scale, np.exp(opt_result.x), sample_size=sample_size
    )
    return kl_best, np.exp(opt_result.x)


def optimize_kl_div_ad(theta_hat, posterior_scale, sample_size=int(1.0e3)):
    # priors = np.linspace(1.0e-3, 3.0, num=int(1.0e3))
    priors = np.linspace(-6, 0.5, num=int(5*1.0e3))

    kls = np.zeros(len(priors))
    for i in range(len(priors)):
        kls[i] = compute_kl_div_mc(
            theta_hat, posterior_scale, np.exp(priors[i]), sample_size=sample_size
        )
    loc = np.argmin(kls)
    return kls[loc], np.exp(priors[loc])


def compute_fast_kl_div_quad(theta_hat, posterior_scale, prior_scale):
    mu, counts = np.unique(theta_hat, return_counts=True)
    kls = compute_kl_quad(mu, posterior_scale=posterior_scale, prior_scale=prior_scale)
    output = np.sum(kls * counts)
    return output


def compute_kl_quad(centroids, posterior_scale, prior_scale):
    kls = np.zeros(len(centroids))
    for k in range(len(centroids)):
        first = -0.5 * (1 + np.log(2. * np.pi * posterior_scale ** 2.))
        second, _ = integrate.quad(
            calculate_second, a=-np.inf, b=np.inf,
            args=(centroids[k], centroids, prior_scale, posterior_scale))
        kls[k] = first - second
    return kls


def calculate_second(theta, center, centroids, prior_scale, posterior_scale):
    p_mix = np.sum(norm.pdf(centroids, loc=theta, scale=prior_scale))
    p = norm.pdf(theta, loc=center, scale=posterior_scale)
    output = np.log(p_mix + 1.e-10) * p
    return output


def compute_kl_div_mc(theta_hat, posterior_scale, priors, sample_size):
    mu, counts = np.unique(theta_hat, return_counts=True)
    mu1 = np.expand_dims(mu, axis=[0])
    theta = np.random.multivariate_normal(
        mean=mu, cov=np.eye(mu.shape[0]) * posterior_scale**2.0, size=sample_size
    )

    log_mix = compute_broad_mix(theta, counts, mu1, priors)
    log_p = compute_broad_gauss(theta, counts, mu1, posterior_scale)
    kl = log_p - log_mix
    return kl


def compute_broad_gauss(theta, counts, mu, scale):
    scale2 = scale**2.0
    cons = -0.5 * np.log(2 * np.pi) - np.log(scale)
    log_p = cons - 0.5 * (1 / scale2) * (theta - mu) ** 2.0
    log_p = np.mean(log_p @ counts)
    return log_p


def compute_broad_mix(theta, counts, mu, scale):
    scale2 = scale**2.0
    diff = (np.expand_dims(theta, -1) - np.expand_dims(mu, -2)) ** 2.0
    cons = -0.5 * np.log(2 * np.pi) - np.log(scale)
    log_p = cons - 0.5 * (1 / scale2) * diff
    log_mix = logsumexp(log_p, axis=-1)
    return np.mean(log_mix @ counts)


def compute_sample_mix(theta, counts, mu, scale):
    scale2 = scale**2.0
    diff = (np.expand_dims(theta, -1) - np.expand_dims(mu, -2)) ** 2.0
    cons = -0.5 * np.log(2 * np.pi) - np.log(scale)
    log_p = cons - 0.5 * (1 / scale2) * diff
    out = logsumexp(log_p, axis=-1)
    return np.sum(out)


def compute_kl_div_quad(theta_hat, posterior_scale, prior_scale):
    mu, counts = np.unique(theta_hat, return_counts=True)
    kls = np.zeros(len(mu))
    for i in range(len(mu)):
        kl, _ = integrate.quad(
            compute_diff_quad,
            a=-np.inf,
            b=np.inf,
            args=(mu[i], mu, posterior_scale, prior_scale),
        )
        kls[i] = kl
    return np.vdot(kls, counts)


def get_kl_mix(theta, theta_hat, mu, posterior_scale, prior_scale):
    sample_size = theta.shape[0]
    results = np.zeros(sample_size)
    for s in range(sample_size):
        results[s] = compute_kl_mix(
            theta[s], theta_hat, mu, posterior_scale, prior_scale
        )
    return np.mean(results)


def compute_kl_mix(theta, theta_hat, mu, posterior_scale, prior_scale):
    dim = theta.shape[0]
    manual = 0.0
    for i in range(dim):
        manual += compute_diff(theta[i], theta_hat[i], mu, posterior_scale, prior_scale)
    return manual


def compute_diff(theta, theta_hat, mu, posterior_scale, prior_scale):
    output = evaluate_log_gaussian(theta, theta_hat, posterior_scale)
    output -= evalute_log_density_gaussian_mix(theta, mu, prior_scale)
    return output


def compute_kl_mix_quad(theta_hat, mu, posterior_scale, prior_scale):
    output = 0.0
    for i in range(theta_hat.shape[0]):
        kl, _ = integrate.quad(
            compute_diff_quad,
            a=-np.inf,
            b=np.inf,
            args=(theta_hat[i], mu, posterior_scale, prior_scale),
        )
        output += kl
    return output


def compute_diff_quad(theta, theta_hat, mu, posterior_scale, prior_scale):
    # aux = norm.logpdf(theta, loc=theta_hat, scale=posterior_scale)
    output = evaluate_log_gaussian(theta, theta_hat, posterior_scale)
    output -= evalute_log_density_gaussian_mix(theta, mu, prior_scale)
    output *= norm.pdf(theta, loc=theta_hat, scale=posterior_scale)
    return output


def evalute_log_density_gaussian_mix(theta, mu, scale):
    scale2 = scale**2.0
    cons = -0.5 * np.log(2 * np.pi) - np.log(scale)
    log_p = cons - 0.5 * (1 / scale2) * (mu - theta) ** 2.0
    return logsumexp(log_p)


def evaluate_log_gaussian(theta, theta_hat, scale):
    scale2 = scale**2.0
    cons = -0.5 * np.log(2.0 * np.pi) - np.log(scale)
    log_p = cons - 0.5 * (1 / scale2) * (theta_hat - theta) ** 2.0
    return log_p
