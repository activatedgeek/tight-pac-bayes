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
from pactl.nn.projectors import LazyRandom,IDModule,RoundedKron, FixedNumpySeed, FixedPytorchSeed
from oil.datasetup.datasets import augLayers,EasyIMGDataset
#import torchvision.datasets as datasets
import timm
from timm.data.transforms import RandomResizedCropAndInterpolation
import torchvision.transforms as transforms
import copy
from oil.tuning.study import Study
from oil.tuning.args import argupdated_config
import warnings

def pretrained_model_surgery(model,num_inputs,num_targets,bnfc_only=False):
    # make all parameters frozen except for the last layer and BN parameters
    if bnfc_only:
        for name,param in model.named_parameters():
            param.requires_grad = False
            if 'bn' in name: param.requires_grad = True
            if 'fc' in name: param.requires_grad=True

    if num_inputs ==1:
        cout,cin,kx,ky = model.conv1.weight.shape
        stride,padding = model.conv1.stride,model.conv1.padding
        conv = nn.Conv2d(num_inputs,cout,kernel_size=(kx,ky),stride=stride,padding=padding,bias=False)
        #conv.weight.data = torch.mean(model.conv1.weight.data,dim=1,keepdim=True)
        model.conv1 = conv

    new_fc = torch.nn.Linear(model.fc.in_features,num_targets)
    # initialize fc with old fc weights
    new_fc.weight.data = model.fc.weight.data[:num_targets]
    new_fc.bias.data = model.fc.bias.data[:num_targets]
    model.fc = new_fc
    return model


def makeTrainer(*,dataset='cifar10',num_epochs=40,modelname='resnet34',data_dir=None,
                bs=50,lr=1e-3,optim=Adam,device='cuda',trainer=Classifier,
                split={'train':1.},net_config={},opt_config={},d=1000,bnfc_only=True,pretrained=True,
                trainer_config={'log_dir':None,'log_args':{'minPeriod':0,'timeFrac':1/3}},save=False):

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
    model = pretrained_model_surgery(model,trainset.num_inputs,trainset.num_classes,bnfc_only)
    model = IDModule(model,LazyRandom,d).to(device)
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],1+len(dataloaders['train'])//10)
    # Add some extra defaults if SGD is chosen
    if optim==SGD: opt_config={**{'momentum':.9,'weight_decay':1e-4,'nesterov':True},**opt_config}
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)
    return trainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)

import dill
if __name__=='__main__':
    with warnings.catch_warnings():
        cfg = argupdated_config(makeTrainer.__kwdefaults__)
        warnings.simplefilter("ignore")
        trainer = makeTrainer(**cfg)
        trainer.train(cfg['num_epochs'])
        # save the model
        with open(f"./{cfg['modelname']}_{cfg['d']}_{cfg['dataset']}.pkl",'wb') as f:
            dill.dump(trainer.model,f)
            #torch.save(trainer.model,f,pickle_module=dill)
