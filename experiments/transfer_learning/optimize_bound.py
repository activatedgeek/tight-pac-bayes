import torch
from torch.utils.data import DataLoader
from torch.optim import SGD,Adam
from oil.utils.utils import LoaderTo, cosLr, islice, export
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset,CIFAR100,CIFAR10
from oil.architectures.img_classifiers import layer13s
from oil.utils.parallel import try_multigpu_parallelize
from oil.tuning.args import argupdated_config
from oil.model_trainers.classifier import Classifier
from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from pactl.nn.projectors import LazyRandom,IDModule,RoundedKron
from oil.datasetup.datasets import augLayers,EasyIMGDataset
import torchvision.datasets as datasets
import timm
from timm.data.transforms import RandomResizedCropAndInterpolation
import torchvision.transforms as transforms
import copy
from oil.tuning.study import Study
from oil.tuning.args import argupdated_config
import warnings
from pactl.bounds.get_bound_from_checkpoint import evaluate_idmodel
import pandas as pd
import numpy as np
import math

def pretrained_model_surgery(model,num_targets,bnfc_only=False):
    # make all parameters frozen except for the last layer and BN parameters
    if bnfc_only:
        for name,param in model.named_parameters():
            param.requires_grad = False
            if 'bn' in name: param.requires_grad = True
            if 'fc' in name: param.requires_grad=True
    new_fc = torch.nn.Linear(model.fc.in_features,num_targets)
    # initialize fc with old fc weights
    new_fc.weight.data = model.fc.weight.data[:num_targets]
    new_fc.bias.data = model.fc.bias.data[:num_targets]
    model.fc = new_fc
    return model


def makeTrainer(*,dataset=CIFAR10,num_epochs=40,modelname='resnet34',
                bs=50,lr=.03,aug=False,optim=SGD,device='cuda',trainer=Classifier,
                split={'train':1.},net_config={},opt_config={},d=1000,bnfc_only=True,pretrained=True,
                trainer_config={'log_dir':None,'log_args':{'minPeriod':0,'timeFrac':1/3}},save=False):

    # Prep the datasets splits, model, and dataloaders
    basic_transform = dataset(f'~/datasets/{dataset}/').default_transform()
    transform = transforms.Compose([RandomResizedCropAndInterpolation(224,(1,1),(1,1)),basic_transform])
    datasets = split_dataset(dataset(f'~/datasets/{dataset}/',transform=transform),splits=split)
    datasets['test'] = dataset(f'~/datasets/{dataset}/', train=False,transform=transform)
    
    device = torch.device(device)
    model = timm.create_model(modelname, pretrained=pretrained)
    model = pretrained_model_surgery(model,dataset.num_targets,bnfc_only)
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = IDModule(model,LazyRandom,d).to(device)
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if aug: model = torch.nn.Sequential(datasets['train'].default_aug_layers(),model)
    #model,bs = try_multigpu_parallelize(model,bs)

    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],1+len(dataloaders['train'])//10)
    # Add some extra defaults if SGD is chosen
    if optim==SGD: opt_config={**{'momentum':.9,'weight_decay':1e-4,'nesterov':True},**opt_config}
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)
    return trainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)

def get_trained_result(cfg,d,extra_bits=0):
    cfg['d'] = d
    trainer = makeTrainer(**cfg)
    trainer.train(cfg['num_epochs'])
    results = evaluate_idmodel(trainer.model,trainer.dataloaders['train'],
        trainer.dataloaders['test'],extra_bits=extra_bits)
    results['d'] = d
    print(results)
    torch.cuda.empty_cache()
    del trainer
    return results

def ternary_search(f,bounds,nevals):
    """ https://mecha-mind.medium.com/ternary-search-convex-optimization-f379e60363b7"""
    a,b = bounds
    for _ in range(nevals//2):
        c1 = a+(b-a)/3
        c2 = a+2*(b-a)/3
        f1 = f(c1)
        f2 = f(c2)
        if f1> f2:
            a = c1
        elif f1 <= f2:
            b = c2
        if abs(a-b) <= delta:
            break
    return (a+b)/2


invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2
def gss(f,bounds,nevals):
    """Golden-section search. https://en.wikipedia.org/wiki/Golden-section_search"""
    a,b = bounds
    (a, b) = (min(a, b), max(a, b))
    h = b - a
    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)
    for k in range(nevals-2):
        if yc < yd:  # yc > yd to find the maximum
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)
    if yc < yd:
        return (a+d)/2
    else:
        return (c+d)/2

class IDBoundSearch(object):
    def __init__(self,bounds=(100,10000),evals=8):
        self.bounds = (int(np.log2(bounds[0])),int(np.log2(bounds[1])))
        self.evals = evals
    def __call__(self,cfg,i=None):
        if i is not None:
            orig_suffix = cfg.setdefault('trainer_config',{}).get('log_suffix','')
            cfg['trainer_config']['log_suffix'] = os.path.join(orig_suffix,f'trial{i}/')
        
        # results = [get_trained_result(cfg,d=b,extra_bits=self.evals-1) for b in self.bounds]
        # lowerval = results[0]['acc_bound']
        # upperval = results[1]['acc_bound']
        #bounds = self.bounds
        results = []
        def f(logd):
            result = get_trained_result(cfg,d=int(np.round(2**logd)),extra_bits=self.evals-1)
            results.append(result)
            return 1-result['acc_bound']
        # perform ternary search to find the maximum accuracy
        optimal_d = gss(f,self.bounds,self.evals)
        return pd.DataFrame(results)
  
if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = argupdated_config(makeTrainer.__kwdefaults__)
        trial = IDBoundSearch((100,10000),evals=8)
        result_df = trial(cfg)
        result_df.to_csv('ID_BS2.csv')