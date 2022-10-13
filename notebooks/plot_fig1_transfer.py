import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(font_scale=2.5, style='whitegrid')
fractions = ["MNIST", "FMNIST", "CIFAR10", "CIFAR100"]
colors = ["#66a61e", "#e7298a", "#7570b3", "#d95f02", "#1b9e77"]
bound = np.array([11.6, 32.8, 58.2, 94.6])
transfer_bound = np.array([9.0, 28.2, 35.1, 81.3])
per = ((bound - transfer_bound) / bound) * 100

plt.figure(dpi=150, figsize=(10, 7))
plt.bar(fractions, per, color=colors)
plt.ylabel("Transfer Err. Bound \nDecrease (%)")
plt.tight_layout()
plt.savefig("transfer.pdf")
plt.show()
