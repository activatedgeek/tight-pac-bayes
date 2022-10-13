import numpy as np
from pactl.bounds.get_pac_bounds import compute_catoni_bound

clusters_n = 2 ** 4
# clusters_n = 25000
acc = 0.92
prob_0 = 0.90
param_size = 257732
sample_size = int(5.e4)
probas = [(1 - prob_0) / clusters_n for _ in range(clusters_n)]
probas += [prob_0]
H = -np.sum(probas * np.log(probas))
message_len = np.ceil(param_size * H)
bound = compute_catoni_bound(
    train_error=1 - acc, divergence=message_len, sample_size=sample_size)
message_len_bits = message_len / np.log(2)
print(f"Mess (bits):  {message_len_bits:1.3e}")
print(f"Mess:         {message_len:1.3e}")
print(f"Entropy:      {H:1.3e}")
print(f"Train err:    {1 - acc:1.3e}")
print(f"Bound:        {bound:1.3e}")
