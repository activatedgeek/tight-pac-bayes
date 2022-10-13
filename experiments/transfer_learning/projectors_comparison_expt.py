import torch
from torch.utils.data import DataLoader
from torch.optim import SGD,Adam
from oil.utils.utils import LoaderTo, cosLr, islice, export
from oil.tuning.study import train_trial
#from oil.datasetup.datasets import split_dataset,CIFAR100,CIFAR10
from pactl.data import get_dataset,get_data_dir
from oil.architectures.img_classifiers import layer13s
from oil.utils.parallel import try_multigpu_parallelize
from oil.tuning.args import argupdated_config
from oil.model_trainers.classifier import Classifier
from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from pactl.nn.projectors import LazyRandom,IDModule,RoundedKron,RoundedDoubleKron,SparseOperator,FastfoodOperator
from pactl.nn.projectors import CombinedRDKronFiLM
from oil.datasetup.datasets import augLayers,EasyIMGDataset
#import torchvision.datasets as datasets
import timm
from timm.data.transforms import RandomResizedCropAndInterpolation
import torchvision.transforms as transforms
import copy
from oil.tuning.study import Study
from oil.tuning.args import argupdated_config
import warnings
from functools import partial
from pactl.bounds.get_bound_from_chk_v2 import evaluate_idmodel
import os
import pandas as pd

def pretrained_model_surgery(model,num_targets,modelname='resnet34'):

    # make all parameters frozen except for the last layer and BN parameters
    
    # initialize fc with old fc weights
    if 'resnet' in modelname:
        new_fc = torch.nn.Linear(model.fc.in_features,num_targets)
        new_fc.weight.data = model.fc.weight.data[:num_targets]
        new_fc.bias.data = model.fc.bias.data[:num_targets]
        model.fc = new_fc
    else:
        new_fc = torch.nn.Linear(model.classifier.in_features,num_targets)
        new_fc.weight.data = model.classifier.weight.data[:num_targets]
        new_fc.bias.data = model.classifier.bias.data[:num_targets]
        model.classifier = new_fc
    return model


def makeTrainer(*,dataset='cifar10',num_epochs=40,modelname='resnet34',data_dir=None,
                bs=50,lr=1e-3,optim=SGD,device='cuda',trainer=Classifier,projector=LazyRandom,
                net_config={},opt_config={},d=1000,pretrained=True,
                trainer_config={'log_dir':None,'log_args':{'minPeriod':0,'timeFrac':1/3}}):

    # Prep the datasets splits, model, and dataloaders
    resize = RandomResizedCropAndInterpolation(224,(1,1),(1,1))
    trainset, testset=  get_dataset(dataset,root=get_data_dir(data_dir),extra_transform=resize,aug=False)
    # why do we have to set these again?
    trainset.class_weights=None
    trainset.ignored_index=-1
    testset.class_weights=None
    testset.ignored_index=-1
    datasets = {'train':trainset,'test':testset}
    
    device = torch.device(device)
    model = timm.create_model(modelname, pretrained=pretrained)
    model = pretrained_model_surgery(model,trainset.num_classes,modelname)
    model = IDModule(model,projector,d).to(device)
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],1+len(dataloaders['train'])//10)
    # Add some extra defaults if SGD is chosen
    if optim==SGD: opt_config={**{'momentum':.9,'weight_decay':1e-4,'nesterov':True},**opt_config}
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)
    return trainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)

import time

def Trial(cfg,i=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if i is not None:
            orig_suffix = cfg.setdefault('trainer_config',{}).get('log_suffix','')
            cfg['trainer_config']['log_suffix'] = os.path.join(orig_suffix,f'trial{i}/')
        trainer = makeTrainer(**cfg)
        num_params = sum([param.numel() for param in trainer.model.trainable_initparams])
        start = time.time()
        trainer.train(cfg['num_epochs'])
        with open(f"./{cfg['modelname']}_{cfg['d']}_{cfg['dataset']}_{i}.pkl",'wb') as f:
            dill.dump(trainer.model,f)
        df_out  = trainer.ckpt['outcome']
        df_out['time'] = time.time()-start
        df_out['params']=num_params
        try:
            df2 = pd.Series(evaluate_idmodel(trainer.model,trainer.dataloaders['train'],
            trainer.dataloaders['test'],lr=3e-3,epochs=40,use_kmeans=False,levels=7))
            if isinstance(df_out,pd.DataFrame):
                df_out = df_out.iloc[0]
            df_out = df_out.append(df2)
            df_out = pd.DataFrame(df_out).T
            print(df_out)
        except Exception as e:
            print('failed to evaluate due to ',e)
        ##combine the two
        torch.cuda.empty_cache()
        del trainer
        return cfg,df_out

import dill
if __name__=='__main__':
    with warnings.catch_warnings():
        cfg_spec = argupdated_config(makeTrainer.__kwdefaults__)
        warnings.simplefilter("ignore")
        #simpleTrial(argupdated_config(makeTrainer.__kwdefaults__))
        cfg_spec2 = copy.deepcopy(cfg_spec)
        cfg_spec2.update({
            'projector':CombinedRDKronFiLM,
            'd':[250,500,1000,2000,3000,4000],'num_epochs':60,'lr':.03,
            'dataset':'svhn', 'modelname':'efficientnet_b0','pretrained':True,
        })
        
        name = 'finetune_svhn_effnetb0'
        thestudy = Study(Trial,cfg_spec2,study_name=name,
                base_log_dir=cfg_spec['trainer_config'].get('log_dir',None))
        thestudy.run(ordered=True)
        # cfg_spec2 = copy.deepcopy(cfg_spec)
        # cfg_spec2.update({
        #     'projector':CombinedRDKronFiLM,
        #     'd':[1000,2000,4000,6000,8000],'num_epochs':60,'lr':.03,
        #     'dataset':'cifar100', 'modelname':'efficientnet_b0','pretrained':True,
        # })
        # thestudy.run(new_config_spec=cfg_spec2,ordered=True)
        print(thestudy.covariates())
        print(thestudy.outcomes)
