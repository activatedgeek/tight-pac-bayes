import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd
import dill

# folder = '../transfer_learning/shuffled'
# # read pandas dataframe
# with open(f'{folder}/results.df', 'rb') as f:
#     df = dill.load(f)

from matplotlib import rc
# rc('text', usetex=True)
# rc('text.latex', preamble=[r'\usepackage{sansmath}', r'\sansmath']) #r'\usepackage{DejaVuSans}'
rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
rc('xtick.major', pad=12)
rc('ytick.major', pad=12)
rc('grid', linewidth=1.3)




sns.set(font_scale=2.5, style='whitegrid')
fig, ax = plt.subplots(1,2,figsize=(20,8))

for i,expt_name in enumerate(['rot_mnist','nonrot_mnist']):
    try:
        with open(f'../equivariant/{expt_name}/results.df', 'rb') as f:
            df = dill.load(f)
    except FileNotFoundError:
        continue

    df.loc[df['expt'].str.contains(''),'model']='WRN'
    df.loc[df['expt'].str.contains('rot'),'model']='C8WRN'
    #dfa = df[df['expt'].str.contains(expt_name)]
    sns.lineplot(ax=ax[i],data=df,x='d',y='raw_err_bound_100',hue='model',legend=False,alpha=.5,lw=4)
    sns.scatterplot(ax=ax[i],data=df,x='d',y='raw_err_bound_100',hue='model',alpha=.5,marker='o',s=400,legend=False if (i>0) else 'brief')
    ax[i].set_xlabel('d')
    # set title to exptname
    ax[i].set_title(expt_name)
    ax[i].set_ylim(0,105)

# set log scale
#ax[1].set_yscale('log')
#ax[1].set_title(f'Runtime', fontsize=30)
fig.tight_layout()
# fig.savefig('id.pdf', bbox_inches='tight')
fig.savefig(f'equivariant.png',bbox_inches='tight')