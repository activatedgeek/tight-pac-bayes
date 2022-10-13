import unittest
from functools import partial
import numpy as np
from scipy.stats import norm
from scipy import integrate
from pactl.bounds.compute_kl_mixture import evaluate_log_gaussian
from pactl.bounds.compute_kl_mixture import evalute_log_density_gaussian_mix
from pactl.bounds.compute_kl_mixture import get_kl_mix
from pactl.bounds.compute_kl_mixture import compute_kl_mix_quad
from pactl.bounds.compute_kl_mixture import compute_kl_div_quad
from pactl.bounds.compute_kl_mixture import compute_kl_div_mc
from pactl.bounds.compute_kl_mixture import compute_broad_mix
from pactl.bounds.compute_kl_mixture import compute_sample_mix
from pactl.bounds.compute_kl_mixture import compute_kl_quad
from pactl.bounds.compute_kl_mixture import compute_broad_gauss
from pactl.bounds.compute_kl_mixture import compute_fast_kl_div_quad
from pactl.bounds.kl_mixture import normal_mixture_log_density
from pactl.bounds.kl_mixture import normal_kl_divergence


class TestKL(unittest.TestCase):

    def test_fast_kl_div_quad(self):
        test_tol = 1.e-0
        theta_hat = np.array([-4., 2., 2., 4., -4., -1.2, -1.2, -1.2])
        mu, counts = np.unique(theta_hat, return_counts=True)
        posterior_scale = np.array(1.2)
        prior_scale = np.array(0.5)
        sample_size = int(1.e4)
        kl_q = compute_fast_kl_div_quad(theta_hat, posterior_scale, prior_scale)
        check = calculate_element(mu, posterior_scale, prior_scale)
        check = np.sum(check * counts)
        kl_mc = compute_kl_div_mc(theta_hat, posterior_scale, prior_scale, sample_size)
        diff = np.linalg.norm(kl_mc - kl_q) / np.linalg.norm(kl_q)
        print('TEST: FAST KL DIV Quad')
        print(f'CHECK:  {check:+1.5e}')
        print(f'KL Q:   {kl_q:+1.5e}')
        print(f'KL MC:  {kl_mc:+1.5e}')
        print(f'Diff:   {diff:+1.5e}')
        self.assertTrue(expr=diff < test_tol)

    def test_element(self):
        test_tol = 1.e-0
        posterior_scale = np.array(1.2)
        prior_scale = np.array(0.5)
        mu = np.array([-5.8, -4., -1.2, 0.5, 1., 3.8, 6.1])
        sample_size = int(1.e3)
        aux = calculate_element(mu, posterior_scale, prior_scale)
        check = np.sum(aux)
        kl_mix = compute_kl_div_mc(mu, posterior_scale, prior_scale, sample_size)
        kl_q = compute_kl_quad(mu, posterior_scale, prior_scale)
        kl_q = np.sum(kl_q)
        diff = np.linalg.norm(check - kl_mix)
        print('TEST: Atomic KL')
        print(f'CHECK: {check:+1.5e}')
        print(f'KL MC: {kl_mix:+1.5e}')
        print(f'KL Q:  {kl_q:+1.5e}')
        print(f'Diff:  {diff:+1.5e}')
        self.assertTrue(expr=diff < test_tol)

    def test_div_mc(self):
        test_tol = 1.e-0
        posterior_scale = np.array(1.2)
        theta_hat = np.array([-4., 2., 2., 4., -4.])
        priors = np.array([1.])
        sample_size = int(1.e3)
        kl = compute_kl_div_quad(theta_hat, posterior_scale, priors[0])
        theta = np.random.multivariate_normal(
            mean=theta_hat,
            cov=np.eye(len(theta_hat)) * posterior_scale ** 2.,
            size=sample_size)
        mu = np.unique(theta_hat)
        kl2 = get_kl_mix(theta, theta_hat, mu, posterior_scale, priors[0])
        kl_mc = compute_kl_div_mc(theta_hat, posterior_scale, priors, sample_size)
        diff = np.linalg.norm(kl_mc - kl)
        print('TEST: KL div MC')
        print(f'KL Q:  {kl:1.5e}')
        print(f'KL MC: {kl_mc:1.5e}')
        print(f'KL 2:  {kl2:1.5e}')
        print(f'Diff:  {diff:+1.5e}')
        self.assertTrue(expr=diff < test_tol)

    def test_broad_mix(self):
        test_tol = 1.e-0
        posterior_scale = np.array(1.2)
        theta_hat = np.array([-4., 2., 2., 4., -4.])
        priors = np.array([1.])
        sample_size = int(1.e2)
        mu, counts = np.unique(theta_hat, return_counts=True)
        theta = np.random.multivariate_normal(
            mean=mu,
            cov=np.eye(mu.shape[0]) * posterior_scale ** 2.,
            size=sample_size)
        aux = compute_broad_mix(theta, counts, np.expand_dims(mu, axis=0), priors[0])
        theta = np.random.multivariate_normal(
            mean=theta_hat,
            cov=np.eye(theta_hat.shape[0]) * posterior_scale ** 2.,
            size=sample_size)
        check = calculate_kl_mix(theta, mu, priors[0])
        diff = np.linalg.norm(aux - check)
        print('TEST: Mixture Broadcasting')
        print(f'AUX:   {aux:1.5e}')
        print(f'CHECK: {check:1.5e}')
        print(f'Diff:  {diff:+1.5e}')
        self.assertTrue(expr=diff < test_tol)

    def test_mix(self):
        test_tol = 1.e-0
        theta_hat = np.array([-4., 2., 2., 4., -4.])
        priors = np.array([1.])
        mu, counts = np.unique(theta_hat, return_counts=True)
        theta = theta_hat + np.random.normal(size=len(theta_hat))
        check = calculate_mix(theta, mu, priors[0])
        aux = compute_sample_mix(theta, counts, mu, priors[0])
        diff = np.linalg.norm(aux - check)
        print('TEST: Mixture fn')
        print(f'AUX:   {aux:1.5e}')
        print(f'CHECK: {check:1.5e}')
        print(f'Diff:  {diff:+1.5e}')
        self.assertTrue(expr=diff < test_tol)

    def test_broad_gauss(self):
        test_tol = 1.e-0
        posterior_scale = np.array(1.2)
        theta_hat = np.array([-4., 2., 2., 4., -4.])
        posterior_scale = np.array(1.)
        sample_size = int(1.e3)
        mu, counts = np.unique(theta_hat, return_counts=True)
        theta = np.random.multivariate_normal(
            mean=mu,
            cov=np.eye(mu.shape[0]) * posterior_scale ** 2.,
            size=sample_size)
        aux = compute_broad_gauss(theta, counts, mu, posterior_scale)
        theta = np.random.multivariate_normal(
            mean=theta_hat,
            cov=np.eye(theta_hat.shape[0]) * posterior_scale ** 2.,
            size=sample_size)
        check = evaluate_log_gaussian(theta, theta_hat, posterior_scale)
        check = np.mean(np.sum(check, axis=-1))
        diff = np.linalg.norm(aux - check)
        print('TEST: Gaussian Broadcasting')
        print(f'AUX:   {aux:1.5e}')
        print(f'CHECK: {check:1.5e}')
        print(f'Diff:  {diff:+1.5e}')
        self.assertTrue(expr=diff < test_tol)

    def test_normal_div(self):
        test_tol = 1.e-0
        mu = np.array([-4., 2., 4.])
        # prior_scale = np.array(0.5)
        # posterior_scale = np.array(1.25)
        prior_scale = np.array(0.25)
        posterior_scale = np.array(1.2)
        log_density = partial(normal_mixture_log_density, mu=mu, sigma=prior_scale)
        zhou = normal_kl_divergence(log_density, mu, posterior_scale)
        print(zhou)
        zhou = np.sum(zhou)
        kl = compute_kl_div_quad(mu, posterior_scale, prior_scale)
        diff = np.linalg.norm(zhou - kl)
        print('TEST: KL div')
        print(f'kl:   {kl:+1.5e}')
        print(f'Zhou: {zhou:+1.5e}')
        print(f'Diff: {diff:+1.5e}')
        self.assertTrue(expr=diff < test_tol)

    def test_mixture_log(self):
        test_tol = 1.e-0
        theta = np.array(2.)
        mu = np.array([-4., 2., 4.])
        prior_scale = np.array(0.25)
        manual = calculate_log_kl_mixture(theta, mu, prior_scale)
        logp = evalute_log_density_gaussian_mix(theta, mu, prior_scale)
        zhou = normal_mixture_log_density(theta, mu, sigma=prior_scale)
        diff = np.linalg.norm(manual - logp)
        print('TEST: Mixture log density')
        print(f'Manual: {manual:+1.5e}')
        print(f'Logp:   {logp:+1.5e}')
        print(f'Zhou:   {zhou:+1.5e}')
        print(f'Diff:   {diff:+1.5e}')
        self.assertTrue(expr=diff < test_tol)

    def test_quadrature(self):
        test_tol = 1.e-0
        seed = 21
        mu = np.array([-4., 2., 4.])
        prior_scale = np.array(1.)
        posterior_scale = np.array(1.)
        sample_size = int(1.e2)
        np.random.seed(seed=seed)
        theta_hat = np.array([-4., 2., 2., 4., -4.])
        dim = theta_hat.shape[0]
        theta = np.random.multivariate_normal(
            mean=theta_hat,
            cov=np.eye(dim) * posterior_scale ** 2.,
            size=sample_size)
        manual = get_kl_mix_test(theta, theta_hat, mu, posterior_scale, prior_scale)
        print('TEST: Quadrature')
        print(f'Samples {manual:+1.3e}')
        kl = compute_kl_mix_quad(theta_hat, mu, posterior_scale, prior_scale)
        kl_fast = compute_kl_div_quad(theta_hat, posterior_scale, prior_scale)
        print(f'KL      {kl:+1.3e}')
        print(f'KL Fast {kl_fast:+1.3e}')
        diff = np.linalg.norm(manual - kl)
        print(f'Diff:   {diff:+1.5e}')
        self.assertTrue(expr=diff < test_tol)
        diff = np.linalg.norm(kl_fast - kl)
        print(f'Diff:   {diff:+1.5e}')
        self.assertTrue(expr=diff < test_tol)

    def test_log_density(self):
        test_tol = 1.e-10
        seed = 21
        mu = np.array([-4., 2., 4.])
        prior_scale = np.array(1.)
        posterior_scale = np.array(1.)
        sample_size = int(1.e2)
        np.random.seed(seed=seed)
        theta_hat = np.array([-4., 2., 2., 4., -4.])
        dim = theta_hat.shape[0]
        theta = np.random.multivariate_normal(
            mean=theta_hat,
            cov=np.eye(dim) * posterior_scale ** 2.,
            size=sample_size)
        print('TEST: KL mix')
        manual = get_kl_mix_test(theta, theta_hat, mu, posterior_scale, prior_scale)
        print(f'Manual {manual:+1.3e}')
        kl = get_kl_mix(theta, theta_hat, mu, posterior_scale, prior_scale)
        print(f'KL     {kl:+1.3e}')
        diff = np.linalg.norm(manual - kl)
        self.assertTrue(expr=diff < test_tol)


def calculate_element(centroids, posterior_scale, prior_scale):
    kls = np.zeros(len(centroids))
    for k in range(len(centroids)):
        first = -0.5 * (1 + np.log(2. * np.pi * posterior_scale ** 2.))
        second, _ = integrate.quad(
                calculate_second, a=-np.inf, b=np.inf,
                args=(centroids[k], centroids, prior_scale, posterior_scale))
        kls[k] = first - second
    return kls


def calculate_second(theta, center, centroids, prior_scale, posterior_scale):
    p_mix = 0.
    for k in range(len(centroids)):
        p_mix += norm.pdf(theta, centroids[k], prior_scale)
    p = norm.pdf(theta, loc=center, scale=posterior_scale)
    return np.log(p_mix + 1.e-10) * p


def calculate_log_kl_mixture(theta, mu, prior_scale):
    output = 0.
    for i in range(len(mu)):
        output += norm.pdf(theta, loc=mu[i], scale=prior_scale)
    return np.log(output)


def calculate_kl_mix(theta, mu, prior_scale):
    sample_size, dim = theta.shape
    results = np.zeros(sample_size)
    for s in range(sample_size):
        results[s] = calculate_mix(theta[s, :], mu, prior_scale)
    return np.mean(results)


def calculate_mix(theta, mu, prior_scale):
    dim = len(theta)
    manual = 0.
    for i in range(dim):
        manual += evalute_log_density_gaussian_mix(theta[i], mu, prior_scale)
    return manual


def get_kl_mix_test(theta, theta_hat, mu, posterior_scale, prior_scale):
    sample_size, dim = theta.shape
    results = np.zeros(sample_size)
    for s in range(sample_size):
        manual = 0.
        for i in range(dim):
            manual += evaluate_log_gaussian(theta[s, i], theta_hat[i], posterior_scale)
            manual -= evalute_log_density_gaussian_mix(theta[s, i], mu, prior_scale)
        results[s] = manual
    return np.mean(results)


if __name__ == '__main__':
    unittest.main()
