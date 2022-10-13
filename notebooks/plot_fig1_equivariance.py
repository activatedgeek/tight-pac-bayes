from matplotlib import pyplot as plt
import seaborn as sns
from palettable.cartocolors.diverging import Temps_2
import dill
from matplotlib import rc

# rc('text', usetex=True)
# rc('text.latex', preamble=[r'\usepackage{sansmath}', r'\sansmath']) #r'\usepackage{DejaVuSans}'
sns.set(font_scale=2.5, style='whitegrid')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans']})
rc('xtick.major', pad=12)
rc('ytick.major', pad=12)
rc('grid', linewidth=1.3)
expt_name = 'rot_mnist'
with open(f'./logs/equivariant/{expt_name}/results.df', 'rb') as f:
    df = dill.load(f)
df = df[df['d'] > 100]
df.loc[df['expt'].str.contains(''), 'model'] = 'WRN'
df.loc[df['expt'].str.contains('rot'), 'model'] = 'WRN (equivariant)'
fig, ax = plt.subplots(figsize=(9, 7))
sns.lineplot(data=df, ax=ax, x='d', y='raw_err_bound_100', hue='model', legend=False,
             alpha=.5, lw=9, color='lightcoral', zorder=0, palette=Temps_2.mpl_colors)
sns.scatterplot(data=df, ax=ax, x='d', y='raw_err_bound_100', hue='model',
                marker='o', s=500, zorder=1, palette=Temps_2.mpl_colors)
handles, labels = ax.get_legend_handles_labels()
for h in handles:
    h.set_sizes([500])
    h.set_alpha(.9)
ax.legend(handles=handles, labels=labels, title='', fontsize=20)
plt.xlabel('Subspace Dimension')
plt.ylabel('Err. Bound (%)')
plt.ylim(60, 105)
fig.savefig('./rotmnist_equiv2.pdf', bbox_inches='tight')
plt.show()
