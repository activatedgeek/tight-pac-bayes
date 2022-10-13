import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd
import dill

folder = 'projector_compare_fromscratch'
# read pandas dataframe
with open(f'{folder}/results.df', 'rb') as f:
    df = dill.load(f)
df['projector'] = df['projector'].astype(str)
projectors = ['FiLMLazyRandom','RoundedDoubleKron','CombinedRDKronFiLM',]
new_names = ['FiLM','Kronecker Product', 'FiLM + Kron']
for name,proj in zip(new_names,projectors):
    df.loc[df['projector'].str.contains(proj),'P Matrix'] = name
df['GPU Hours'] = df['time']/3600.
print(df.head)


from matplotlib import rc
# rc('text', usetex=True)
# rc('text.latex', preamble=[r'\usepackage{sansmath}', r'\sansmath']) #r'\usepackage{DejaVuSans}'
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('xtick.major', pad=12)
rc('ytick.major', pad=12)
rc('grid', linewidth=1.3)

sns.set(font_scale=2.5, style='whitegrid')
fig, ax = plt.subplots(1,1,figsize=(10,8))

sns.lineplot(ax=ax,data=df,x='d',y='Train_Acc',hue='P Matrix',legend=False,alpha=.5,lw=4)
sns.scatterplot(ax=ax,data=df,x='d',y='Train_Acc',hue='P Matrix',alpha=.5,marker='o',s=400)
ax.set_title(f'Projection Methods', fontsize=30)
ax.set_xlabel(f'Subspace dimension', fontsize=30)

fig.tight_layout()
# fig.savefig('id.pdf', bbox_inches='tight')
fig.savefig(f'projector_comparison_scratch.png',bbox_inches='tight')
fig.savefig(f'projector_comparison_scratch.pdf',bbox_inches='tight')
#fig.savefig(f'projector_comparison.pdf',bbox_inches='tight')