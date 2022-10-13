import numpy as np
# import seaborn as sns
from matplotlib import rc as rc
from matplotlib import pyplot as plt
from pactl.bounds.quantize_fns import get_random_symbols_and_codebook
from pactl.bounds.quantize_fns import get_kmeans_symbols_and_codebook

filepath = 'v3vec.npy'
# filepath = 'v3600vec.npy'
# filepath = 'cifar10vec.npy'
vec = np.load(filepath)
print("*+" * 50)
print("Statistics")
print("*+" * 50)
print(f"Max:  {np.max(vec):+1.3e}")
print(f"q75:  {np.quantile(vec, q=0.75):+1.3e}")
print(f"q50:  {np.quantile(vec, q=0.5):+1.3e}")
print(f"q25:  {np.quantile(vec, q=0.25):+1.3e}")
print(f"Min:  {np.min(vec):+1.3e}")
print(f"\nMean: {np.mean(vec):+1.3e}")
print(f"Std:  {np.std(vec):+1.3e}")
print(f"Size: {len(vec):+1.3e}")
print("*+" * 50)
# bins_num, ymax, xmin, xmax = 100, 500, -6, 6
bins_num, ymax, xmin, xmax = 1_000, 1_000, -6, 6
levels = 50
random_choices = (get_random_symbols_and_codebook, get_kmeans_symbols_and_codebook)
# get_random = random_choices[0]
get_random = random_choices[1]
symbols, codebook = get_random(vec, levels, codebook_dtype=np.float16)

sorted_codebook = np.sort(codebook)
new_codebook = []
for i in range(0, 7, 2):
    new_codebook.append(sorted_codebook[i])
    new_codebook.append(sorted_codebook[-i])

new_codebook.append(sorted_codebook[25])
new_codebook.append(sorted_codebook[20])
new_codebook.append(sorted_codebook[30])
codebook = np.array(new_codebook)

# sns.set(font_scale=2.5, style='whitegrid')
font = {"size": 25}
rc('font', **font)

plt.figure(dpi=150, figsize=(9, 7))
plt.hist(vec, bins=bins_num, color="#bf5b17")
plt.axvline(codebook[0], color='#386cb0', label="quant centroids", lw=3)
for i in range(1, len(codebook)):
    plt.axvline(codebook[i], color='#386cb0', lw=3)
plt.ylim(0, ymax)
plt.xlim(xmin, xmax)
# plt.xlim(-10, 10)
plt.xlabel("Random Projection Weights")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.savefig("centroids.pdf")
plt.show()
