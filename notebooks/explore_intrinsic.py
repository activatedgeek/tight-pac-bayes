import numpy as np
from matplotlib import pyplot as plt
from pactl.bounds.quantize_fns import get_kmeans_symbols_and_codebook
from pactl.bounds.quantize_fns import get_random_symbols_and_codebook
from qtorch import FloatingPoint

par = np.load(file='cifar_par.npy')
levels = 2 ** 2 + 1
print(np.max(par))
print(np.mean(par))
print(np.median(par))
print(np.min(par))

_, centroids = get_random_symbols_and_codebook(par, levels, np.float16)
_, centroids = get_kmeans_symbols_and_codebook(par, levels, np.float16)
# bit_8 = FloatingPoint(exp=3, man=4)
# aux = quantizer(forward_number=bit_8, forward_rounding='nearest')

plt.style.use('ggplot')
plt.figure(dpi=250)
plt.hist(par, bins=100, label='intrinsic values')
for i in range(levels - 1):
    plt.axvline(x=centroids[i], color='blue')
plt.axvline(x=centroids[-1], color='blue', label='centroids')
plt.legend()
plt.show()
