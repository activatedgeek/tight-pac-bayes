import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
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
from pactl.nn import IntrinsicDenseNet, LazyIntrinsicDenseNet
from oil.datasetup.datasets import augLayers,EasyIMGDataset
import torchvision.datasets as datasets
import timm
from timm.data.transforms import RandomResizedCropAndInterpolation
import torchvision.transforms as transforms
import copy
from oil.tuning.study import Study
from oil.tuning.args import argupdated_config


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


def makeTrainer(*,dataset=CIFAR10,num_epochs=50,modelname='resnet34',
                bs=50,lr=.03,aug=True,optim=SGD,device='cuda',trainer=Classifier,
                split={'train':1.},net_config={},opt_config={},d=290,bnfc_only=True,pretrained=True,
                trainer_config={'log_dir':None},save=False):

    # Prep the datasets splits, model, and dataloaders
    basic_transform = dataset(f'~/datasets/{dataset}/').default_transform()
    transform = transforms.Compose([RandomResizedCropAndInterpolation(224,(1,1),(1,1)),basic_transform])
    datasets = split_dataset(dataset(f'~/datasets/{dataset}/',transform=transform),splits=split)
    datasets['test'] = dataset(f'~/datasets/{dataset}/', train=False,transform=transform)
    
    device = torch.device(device)
    model = timm.create_model(modelname, pretrained=pretrained)
    model = pretrained_model_surgery(model,dataset.num_targets,bnfc_only)
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = LazyIntrinsicDenseNet(model,d).to(device)
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

simpleTrial = train_trial(makeTrainer)
if __name__=='__main__':
    #simpleTrial(argupdated_config(makeTrainer.__kwdefaults__))
    cfg_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
    cfg_spec.update({
        'd':[250,500,1000,2000,4000,8000,16000],'study_name':'id_study_nopretrained','num_epochs':100,'lr':.03,
        'aug':True,'dataset':[CIFAR10,CIFAR100], 'modelname':'resnet34','pretrained':True,
    })
    name = cfg_spec.pop('study_name')
    thestudy = Study(simpleTrial,cfg_spec,study_name=name,
            base_log_dir=cfg_spec['trainer_config'].get('log_dir',None))
    thestudy.run(ordered=False)
    print(thestudy.covariates())
    print(thestudy.outcomes)