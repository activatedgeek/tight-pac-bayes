import numpy as np
from matplotlib import pyplot as plt

dds = np.load('dds.npy')
means = np.load('means.npy')

plt.style.use('ggplot')
plt.figure(dpi=150)
plt.scatter(dds, means)
plt.plot(dds, means)
plt.xlabel('intrinsic dim')
# plt.ylabel('mean abs error')
plt.ylabel('condition number')
plt.yscale('log')
plt.show()
