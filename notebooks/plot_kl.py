import time
import numpy as np
from matplotlib import pyplot as plt
from pactl.bounds.compute_kl_mixture import optimize_kl_div
from pactl.bounds.compute_kl_mixture import get_gains
from pactl.bounds.compute_kl_mixture import compute_kl_div_mc
# from pactl.bounds.compute_kl_mixture import compute_fast_kl_div_quad
# from pactl.bounds.compute_kl_mixture import optimize_kl_div_opt

do_plot = True
seed = 21
np.random.seed(seed=seed)
sample_size = int(1.e3)
# quantized_vec = np.load(file='./tests/quant_3.npy')
# centroids = np.load(file='./tests/centroids_3.npy')
# quantized_vec = np.load(file='./tests/quant_10.npy')
# centroids = np.load(file='./tests/centroids_10.npy')
quantized_vec = np.load(file='./tests/quant.npy')
centroids = np.unique(quantized_vec, return_counts=False)
print(len(quantized_vec) * np.log(centroids.shape[0]))

if do_plot:
    uq, counts = np.unique(quantized_vec, return_counts=True)
    plt.style.use('ggplot')
    plt.figure(dpi=250)
    plt.bar(uq, counts)
    plt.xlabel('centroids')
    plt.ylabel('counts')
    plt.legend()
    plt.show()

posterior_scale = 0.5 * np.std(quantized_vec)
# posterior_scale = 0.1 * np.std(quantized_vec)
# posterior_scale = 0.05 * (np.max(np.abs(centroids)) - np.min(np.abs(centroids)))
# posterior_scale = 0.01
print(f'Posterior scale: {posterior_scale:1.3e}')

print('=' * 50)
kl, pp = optimize_kl_div(quantized_vec, posterior_scale, sample_size=sample_size)
print(f'Min KL: {kl:2.5e} | min prior {pp:2.5e}')

# gain_bits = get_gains(quantized_vec, posterior_scale, sample_size=sample_size)
# print(f'Bits back: {gain_bits:2.5e}')


t0 = time.time()
priors = np.linspace(1.0e-3, 3.0, num=int(1.0e3))
# priors = np.linspace(1.0e-2, 2.0, num=int(1.0e3))
# priors = np.linspace(1.0e-1, 2.0, num=int(1.0e2))
# priors = np.linspace(1.0, 2.0, num=int(1.0e2))  # Shows problematic low ranges
# priors = np.linspace(1.0e-3, 1.0e-2, num=int(1.0e1))
kls = np.zeros(len(priors))
# kls_f = np.zeros(len(priors))
for i in range(len(priors)):
    kls[i] = compute_kl_div_mc(
        quantized_vec, posterior_scale, priors[i], sample_size=sample_size
    )
    # kls_f[i] = compute_fast_kl_div_quad(
    #     theta_hat=quantized_vec,
    #     posterior_scale=posterior_scale,
    #     prior_scale=priors[i],
    # )
loc = np.argmin(kls)
# loc_f = np.argmin(kls_f)
t1 = time.time()
print(f'{t1 - t0} sec')
if do_plot:
    plt.figure(dpi=250)
    plt.plot(priors, kls, label='MC')
    # plt.plot(priors, kls_f, label='Quad')
    plt.title(f'Posterior {posterior_scale:1.3e} | Prior {priors[loc]:1.3e}')
    plt.ylim([kls[loc], kls[-1] + np.abs(kls[-1]) * 0.1])
    plt.legend()
    plt.show()

# kl, pp = optimize_kl_div_opt(quantized_vec, posterior_scale, sample_size=sample_size)
# print('Optimized')
# print(f'Min KL: {kl:2.5e} | min prior {pp:2.5e}')
print(f'Min KL: {kls[loc]:2.5e} | min prior {priors[loc]:2.5e}')
# print(f'Min KL: {kls_f[loc_f]:2.5e} | min prior {priors[loc_f]:2.5e}')
