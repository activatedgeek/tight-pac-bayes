import numpy as np
from pactl.bounds.compute_kl_mixture import get_kl_mix
from pactl.bounds.compute_kl_mixture import compute_kl_div_quad
from pactl.bounds.compute_kl_mixture import optimize_kl_div
from pactl.bounds.compute_kl_mixture import get_gains
from pactl.bounds.compute_kl_mixture import get_gains_opt
from pactl.bounds.kl_mixture import divergence_gains
from pactl.bounds.kl_mixture import divergence_gains_opt
from pactl.bounds.kl_mixture import get_gains as get_gains_zhou
# from pactl.bounds.compute_kl_mixture import compute_kl_div_mc
# from pactl.bounds.compute_kl_mixture import compute_kl_mix_quad

seed = 21
np.random.seed(seed=seed)
sample_size = int(1.e3)
quantized_vec = np.load(file='./tests/quant_3.npy')
centroids = np.load(file='./tests/centroids_3.npy')
# quantized_vec = np.load(file='./tests/quant_10.npy')
# centroids = np.load(file='./tests/centroids_10.npy')
# posterior_scale = np.array(1.)
# prior_scale = np.array(1.)
posterior_scale = 0.05 * (np.max(np.abs(centroids)) - np.min(np.abs(centroids)))
prior_scale = np.array(1.5)
print(f'Posterior scale: {posterior_scale:1.3e}')

print('=' * 50)
div = divergence_gains(
    quantized_vec,
    scale_posterior=posterior_scale,
    scale_prior=prior_scale,
    rescale_posterior=False, scale_prior_by_x=False,
    return_std=False)
print(f'Div {div:2.5e}')

# theta = np.random.multivariate_normal(
#     mean=quantized_vec,
#     cov=np.eye(quantized_vec.shape[0]) * posterior_scale ** 2.,
#     size=sample_size)
# kl_mc = get_kl_mix(theta, quantized_vec, centroids, posterior_scale, prior_scale)
# print(f'KL MC {kl_mc:2.5e}')
# 
# diff = np.linalg.norm(kl_mc - div)
# print(f'Diff: {diff:2.5e} | passed test {np.abs(diff) < 1.e-1}')

kl = compute_kl_div_quad(quantized_vec, posterior_scale, prior_scale)
print(f'KL Quad {kl:2.5e}')

print('=' * 50)
div_opt = divergence_gains_opt(quantized_vec, posterior_scale, False)
print(f'Div Opt: {div_opt[0]:2.5e}')

zhou_gains = get_gains_zhou(quantized_vec, posterior_scale, rescale_posterior=True)[0]
print(f'Zhou gains: {zhou_gains:2.5e}')

print('=' * 50)
kl, pp = optimize_kl_div(quantized_vec, posterior_scale, sample_size=sample_size)
print(f'Min KL: {kl:2.5e} | min prior {pp:2.5e}')

gain_bits = get_gains(quantized_vec, posterior_scale, sample_size=sample_size)
print(f'Bits back: {gain_bits:2.5e}')

gain_bits_opt = get_gains_opt(quantized_vec, posterior_scale, sample_size=sample_size)
print(f'Bits back Opt: {gain_bits_opt:2.5e}')

priors = np.linspace(1.e-3, 3., num=int(1.e2))
kls = np.zeros(len(priors))
for i in range(len(priors)):
    kls[i] = compute_kl_div_quad(quantized_vec, posterior_scale, priors[i])
#     # kls[i] = get_kl_mix(theta, quantized_vec, centroids, posterior_scale, priors[i])
#     kls[i] = compute_kl_div_mc(quantized_vec, posterior_scale, priors[i],
#                                sample_size=sample_size)
loc = np.argmin(kls)
print(f'Min KL: {kls[loc]:2.5e} | min prior {priors[loc]:2.5e}')
