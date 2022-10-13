import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(font_scale=2.5, style='whitegrid')
fractions = np.array([0.05 + (0.1) * i for i in range(10)])
prior = 100 - np.array([100., 90., 80., 62., 57., 47., 37., 33., 28., 22.])
posterior = 100. - np.array([65., 60., 55., 50., 40., 35., 33., 29., 23., 20.])

plt.figure(dpi=150, figsize=(9, 7))
plt.plot(fractions, prior, lw=9, alpha=0.75, label="Prior", c="#d95f02")
# plt.scatter(fractions, prior, s=300, c="#d95f02")
plt.plot(fractions, posterior, lw=9, alpha=0.75, label="Posterior", c="#7570be")
# plt.scatter(fractions, posterior, s=300, c="#7570be")
plt.fill_between(fractions, prior, color='#d95f02', alpha=0.3)
plt.fill_between(fractions, posterior, prior, color='#7570be', alpha=0.3)
plt.axvline(x=0.8, ymin=0., ymax=100., color="#a6761d", lw=5)
plt.ylabel("Acc bound (%)")
plt.xlabel("Fraction of data used for prior")
plt.legend()
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig("dd_explain.pdf")
plt.show()
