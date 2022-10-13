import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd
import dill

folder = '../transfer_learning/finetune_cifar10_100_effnetb0'
with open(f'{folder}/results.df', 'rb') as f:
    df = dill.load(f)
    df['kl'] = df['prefix_message_len']/(1024*8)
    print(df[['d','raw_err_bound_100','kl']])