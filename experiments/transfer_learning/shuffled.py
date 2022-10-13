import torch
from torch.utils.data import DataLoader
from torch.optim import SGD,Adam
from oil.utils.utils import LoaderTo, cosLr, islice, export
from oil.tuning.study import train_trial
#from oil.datasetup.datasets import split_dataset,CIFAR100,CIFAR10
from pactl.data import get_dataset,get_data_dir
#from oil.architectures.img_classifiers import layer13s
from oil.utils.parallel import try_multigpu_parallelize
from oil.tuning.args import argupdated_config
from oil.model_trainers.classifier import Classifier
from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from pactl.nn.projectors import LazyRandom,IDModule,RoundedKron, FixedNumpySeed, FixedPytorchSeed,RoundedDoubleKron
from pactl.nn.projectors import FiLMLazyRandom,CombinedRDKronFiLM
from oil.datasetup.datasets import augLayers,EasyIMGDataset
#import torchvision.datasets as datasets
import timm
from timm.data.transforms import RandomResizedCropAndInterpolation
import torchvision.transforms as transforms
import copy
from oil.tuning.study import Study
from oil.tuning.args import argupdated_config
import warnings
from pactl.nn import resnet20,layer13s
from pactl.nn.small_cnn import Expression
import pactl
import pandas as pd
#from pactl.bounds.get_bound_from_checkpoint import evaluate_idmodel,auto_eval
from oil.datasetup.augLayers import RandomTranslate,RandomHorizontalFlip
from pactl.bounds.get_bound_from_chk_v2 import evaluate_idmodel#,auto_eval
import numpy as np

# make an mlp
class MLP(nn.Module):
    def __init__(self,num_classes=10,nlayers=3,base_width=150):
        super().__init__()
        layers = []
        width = 3*32*32
        self.layers = nn.Sequential(
            nn.Linear(width,base_width),
            nn.ReLU(),
            nn.Linear(base_width,base_width),
            nn.ReLU(),
            nn.Linear(base_width,base_width),
            nn.ReLU(),
        )
        self.fc = nn.Linear(base_width,num_classes)
    def forward(self,x):
        return self.fc(self.layers(x.reshape(x.shape[0],-1)))

def makeTrainer(*,dataset='cifar10',num_epochs=2,data_dir=None,projector=LazyRandom,
                bs=128,lr=1e-3,optim=Adam,device='cuda',trainer=Classifier,expt='',
                net_config={'base_width':20},opt_config={},d=8000,seed=137,
                trainer_config={'log_dir':None,'log_args':{'minPeriod':0,'timeFrac':1/3}}):

    # Prep the datasets splits, model, and dataloaders
    trainset, testset=  get_dataset(dataset,root=get_data_dir(data_dir),aug=False)
    # why do we have to set these again?
    trainset.class_weights=None
    trainset.ignored_index=-1
    testset.class_weights=None
    testset.ignored_index=-1
    datasets = {'train':trainset,'test':testset}

    device = torch.device(device)
    with FixedPytorchSeed(seed):
        if 'mlp' in expt:
            model = MLP(num_classes=10,nlayers=3,base_width=150).to(device)
        elif 'cnn' in expt:
            model = layer13s(num_classes=trainset.num_classes,**net_config)
        else:
            raise ValueError(f'unknown experiment {expt}')
        if 'shuffled_pixels' in expt:
            perm = torch.randperm(np.prod(trainset[0][0].shape[1:]))
            pixel_shuffle = Expression(lambda x: x.reshape(*x.shape[:2],-1)[:,:,perm].reshape(*x.shape))
            model = nn.Sequential(pixel_shuffle,model).to(device)
        model = IDModule(model,projector,d).to(device)
        if 'shuffled_labels' in expt:
            trainset.targets = [trainset.targets[i] for i in torch.randperm(len(trainset))]
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],1+len(dataloaders['train'])//10)
    opt_constr = partial(optim, lr=lr, **opt_config)
    return trainer(model,dataloaders,opt_constr,**trainer_config)

import os
import pandas as pd
import time

def Trial(cfg,i=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if i is not None:
            orig_suffix = cfg.setdefault('trainer_config',{}).get('log_suffix','')
            cfg['trainer_config']['log_suffix'] = os.path.join(orig_suffix,f'trial{i}/')
        trainer = makeTrainer(**cfg)
        
        num_params = sum([param.numel() for param in trainer.model.trainable_initparams])
        
        print(f'{cfg["expt"]} {num_params}')
        start = time.time()
        trainer.train(cfg['num_epochs'])
        df_out  = trainer.ckpt['outcome']
        df_out['time'] = time.time()-start
        df_out['params']=num_params
        
        
        try:
            df2 = pd.Series(evaluate_idmodel(trainer.model,trainer.dataloaders['train'],
            trainer.dataloaders['test'],lr=3e-3,epochs=50,use_kmeans=False,levels=7))
            if isinstance(df_out,pd.DataFrame):
                df_out = df_out.iloc[0]
            df_out = df_out.append(df2)
            df_out = pd.DataFrame(df_out).T
            print(df_out)
        except Exception as e:
            print('failed to evaluate due to ',e)
        # combine the two
        torch.cuda.empty_cache()
        del trainer
        return cfg,df_out


import dill
if __name__=='__main__':
        cfg_spec = argupdated_config(makeTrainer.__kwdefaults__)
        cfg_spec = copy.deepcopy(cfg_spec)
        cfg_spec.update({
            'd':[500,1000,2000,4000,8000,16000,32000],
            'study_name':'shuffled','num_epochs':100,'lr':3e-3,
            'dataset':'cifar10', 'expt':['cnn','cnn_shuffled_pixels','cnn_shuffled_labels','mlp','mlp_shuffled_labels'],
            'projector':CombinedRDKronFiLM,
            'net_config':{'base_width':23},
        })
        name = cfg_spec.pop('study_name')
        thestudy = Study(Trial,cfg_spec,study_name=name,
                base_log_dir=cfg_spec['trainer_config'].get('log_dir',None))
        thestudy.run(ordered=True)
        thestudy.results_df().to_csv(f'{name}.csv')
        print(thestudy.covariates())
        print(thestudy.outcomes)
