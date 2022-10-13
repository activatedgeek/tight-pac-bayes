import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd
import dill

folder = 'network_comparison'
# read pandas dataframe
with open(f'{folder}/results.df', 'rb') as f:
    df = dill.load(f)
if 'model' in df.columns:
    df.loc[df['model'].str.contains('resnet20'),'model'] = 'resnet20'
    df.loc[df['model'].str.contains('layer13s'),'model'] = 'layer13s'
print(df.columns)
#df['d'] = df['d'].astype(float)
#df['Pretrained'] = name

    
sns.set(font_scale=2, style='whitegrid')
fig, ax = plt.subplots(figsize=(9,7))
#ax.set(xscale='log')
# sns.lineplot(ax=ax,data=df,x='base_width',y='acc_bound',hue='model',legend=False,alpha=.5)
# sns.lineplot(ax=ax,data=df,x='base_width',y='train_acc',hue='model',legend=False,alpha=.5)
# #ax.set(xscale='log')
# sns.scatterplot(ax=ax,data=df,x='base_width',y='acc_bound',hue='model',alpha=.5,marker='o',s=400)

sns.lineplot(ax=ax,data=df,x='base_width',y='train_acc',hue='model',legend=False,alpha=.5)
sns.scatterplot(ax=ax,data=df,x='base_width',y='train_acc',hue='model',alpha=.5,marker='o',s=400)
sns.scatterplot(ax=ax,data=df,x='base_width',y='quantized_train_acc',hue='model',alpha=.5,marker='o',s=400)



#plt.legend(loc='lower right')
#plt.xscale('log')
# save figure
#fig.title(f'Intrinsic Dim finetune on {ds}')
# add title
ax.set_title(f'From scratch', fontsize=20)
fig.tight_layout()
# fig.savefig('id.pdf', bbox_inches='tight')
fig.savefig(f'from_scratch.png',bbox_inches='tight')