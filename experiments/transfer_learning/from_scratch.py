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
import pactl
import pandas as pd
#from pactl.bounds.get_bound_from_checkpoint import evaluate_idmodel,auto_eval
from oil.datasetup.augLayers import RandomTranslate,RandomHorizontalFlip
from pactl.bounds.get_bound_from_chk_v2 import evaluate_idmodel#,auto_eval

def makeTrainer(*,dataset='cifar10',num_epochs=2,model=layer13s,data_dir=None,projector=LazyRandom,
                bs=50,lr=.1,optim=SGD,device='cuda',trainer=Classifier,aug=True,
                net_config={'base_width':8},opt_config={},d=8000,
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
    model = model(num_classes=trainset.num_classes,**net_config)
    if aug: model = torch.nn.Sequential(RandomTranslate(4),RandomHorizontalFlip(),model)
    model = IDModule(model,projector,d).to(device)
    
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],1+len(dataloaders['train'])//10)
    # Add some extra defaults if SGD is chosen
    if optim==SGD: opt_config={**{'momentum':.9,'weight_decay':1e-4,'nesterov':True},**opt_config}
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)
    return trainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)

# import dill
# if __name__=='__main__':
#     with warnings.catch_warnings():
#         cfg = argupdated_config(makeTrainer.__kwdefaults__,namespace = pactl.nn)
#         warnings.simplefilter("ignore")
#         print(pd.Series(cfg).T)
#         trainer = makeTrainer(**cfg)
#         # trainer.train(cfg['num_epochs'])
#         # # save the model
#         # with open(f"./{cfg['model'].__name__}_{cfg['d']}_{cfg['dataset']}.pkl",'wb') as f:
#         #     dill.dump(trainer.model,f)
#             #torch.save(trainer.model,f,pickle_module=dill)
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
        start = time.time()
        trainer.train(cfg['num_epochs'])
        df_out  = trainer.ckpt['outcome']
        df_out['time'] = time.time()-start
        num_params = sum(p.numel() for p in trainer.model._forward_net[0].parameters() if p.requires_grad)
        df_out['params']=num_params
        
        try:
            df2 = pd.Series(evaluate_idmodel(trainer.model,trainer.dataloaders['train'],
            trainer.dataloaders['test'],lr=3e-3,epochs=30,use_kmeans=False,levels=7))
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
            'd':[250,500,1000,2000,4000,8000,16000,32000],
            'study_name':'projector_compare_fromscratch','num_epochs':50,'lr':.1,
            'dataset':'cifar10', 'model':layer13s,
            'projector':[RoundedDoubleKron,FiLMLazyRandom,CombinedRDKronFiLM],
            'net_config':{'base_width':8},'aug':True,
        })
        name = cfg_spec.pop('study_name')
        thestudy = Study(Trial,cfg_spec,study_name=name,
                base_log_dir=cfg_spec['trainer_config'].get('log_dir',None))
        thestudy.run(ordered=True)
        thestudy.results_df().to_csv(f'{name}.csv')
        print(thestudy.covariates())
        print(thestudy.outcomes)
