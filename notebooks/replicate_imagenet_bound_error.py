import numpy as np
from pactl.bounds.get_pac_bounds import compute_catoni_bound
from pactl.bounds.get_pac_bounds import compute_mcallester_bound
from pactl.bounds.get_pac_bounds import pac_bayes_bound_opt


train_size = 1281167
train_accuracy = 0.22
divergence = (np.log2(20) * 1e5) * np.log(2)
# train_accuracy = 0.62
# divergence = (np.log2(20) * 600_000) * np.log(2)
# Argument to try 600 K intrinsic dim (or 650 K)
# Upperbound H(x) \leq \log_2(K)

# Numbers below reflect Zhou et al 2019.
# 4.2 M params in MobileNet with 67% pruning: 1.386e6
# 65% train accuracy
# 2.10 entropy computed to match paper numbers
# This is like close to 5 clusters! But they say they used 15
# train_accuracy = 0.65
# Comp size with 67% prunning and 15 clusters
# (np.log2(15) * 1.386e6) * np.log(2)
# 3702784 (452 KiB)
# Effective size 350 KiB (2,867,200 bits)
# divergence = 2_867_200 * np.log(2)

print(f'Divergence: {divergence:1.5e}')
pac_bayes = pac_bayes_bound_opt(
    divergence=divergence, train_error=1. - train_accuracy, n=train_size)

catoni = compute_catoni_bound(
    train_error=1. - train_accuracy, divergence=divergence, sample_size=train_size)

mcallester = compute_mcallester_bound(
    train_error=1. - train_accuracy, div=divergence, sample_size=train_size)

print(f'PAC-Bayes:  {pac_bayes:1.5e}')
print(f'Catoni:     {catoni:1.5e}')
print(f'McAllester: {mcallester:1.5e}')
