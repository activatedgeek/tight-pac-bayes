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


from PIL import Image
from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, \
    verify_str_arg
from torchvision.datasets.vision import VisionDataset
import torchvision.datasets as ds
import os
#Rot mnist dataset
class MnistRotDataset(VisionDataset):
    """ Official RotMNIST dataset."""
    ignored_index = -100
    class_weights = None
    balanced = True
    stratify = True
    means = (0.130,)
    stds = (0.297,)
    num_targets=10
    resources = ["http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"]
    training_file = 'mnist_all_rotation_normalized_float_train_valid.amat'
    test_file = 'mnist_all_rotation_normalized_float_test.amat'
    def __init__(self,root, train=True, transform=None,download=True):
        if transform is None:
            normalize = transforms.Normalize(self.means, self.stds)
            transform = transforms.Compose([transforms.ToTensor(),normalize])
        super().__init__(root,transform=transform)
        self.train = train
        if download:
            self.download()
        if train:
            file=os.path.join(self.raw_folder, self.training_file)
        else:
            file=os.path.join(self.raw_folder, self.test_file)
        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        c,h,w = image.shape
        rgb_img = torch.zeros(3,h,w)
        return rgb_img+image, label
    def _check_exists(self):
        return (os.path.exists(os.path.join(self.raw_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.raw_folder,
                                            self.test_file)))
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')
    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder,exist_ok=True)
        os.makedirs(self.processed_folder,exist_ok=True)

        # download files
        for url in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=None)
        print('Downloaded!')

    def __len__(self):
        return len(self.labels)

import torch

from pactl.nn.e2cnn import Wide_ResNet

from oil.datasetup.datasets import EasyIMGDataset
@export
class MNIST(EasyIMGDataset,ds.MNIST):
    means = (0.1307*255,)
    stds = (0.3081*255,)
    num_targets=10

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        c,h,w = img.shape
        rgb_img = torch.zeros(3,h,w)
        return rgb_img+img, target

@export
class RotMNIST(MNIST):
    """ Unofficial RotMNIST that has the same size as MNIST (60k)"""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        N = len(self)
        with FixedNumpySeed(182):
            angles = torch.rand(N)*2*np.pi
        with torch.no_grad():
            affineMatrices = torch.zeros(N,2,3)
            affineMatrices[:,0,0] = angles.cos()
            affineMatrices[:,1,1] = angles.cos()
            affineMatrices[:,0,1] = angles.sin()
            affineMatrices[:,1,0] = -angles.sin()
            self.data = self.data.unsqueeze(1).float()
            flowgrid = F.affine_grid(affineMatrices, size = self.data.size())
            self.data = F.grid_sample(self.data, flowgrid).squeeze(1)

def makeTrainer(*,num_epochs=2,data_dir=None,projector=LazyRandom,
                bs=128,lr=1e-3,optim=Adam,device='cuda',trainer=Classifier,expt='',
                opt_config={},d=8000,seed=137,dataset=MNIST,
                trainer_config={'log_dir':None,'log_args':{'minPeriod':0,'timeFrac':1/3}}):

    # Prep the datasets splits, model, and dataloaders
    #trainset, testset=  get_dataset(dataset,root=get_data_dir(data_dir),aug=False)
    # why do we have to set these again?
    # trainset = MnistRotDataset(root=get_data_dir(data_dir),train=True)
    # testset = MnistRotDataset(root=get_data_dir(data_dir),train=False)
    trainset = dataset(root=get_data_dir(data_dir),train=True)
    testset = dataset(root=get_data_dir(data_dir),train=False)
    datasets = {'train':trainset,'test':testset}

    device = torch.device(device)
    with FixedPytorchSeed(seed):
        if 'rot' in expt:
            model = Wide_ResNet(10, 4, 0.0, initial_stride=2, N=8, f=False, r=3, num_classes=trainset.num_targets).to(device)
        else:
            model = Wide_ResNet(10, 4.67, 0.0, initial_stride=2, N=1, f=False, r=0, num_classes=trainset.num_targets).to(device)
        model = IDModule(model,projector,d).to(device)

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
        start = time.time()
        trainer.train(cfg['num_epochs'])
        df_out  = trainer.ckpt['outcome']
        df_out['time'] = time.time()-start
        df_out['params']=num_params
        
        try:
            df2 = pd.Series(evaluate_idmodel(trainer.model,trainer.dataloaders['train'],
            trainer.dataloaders['test'],lr=3e-3,epochs=30,use_kmeans=False,levels=7,misc_extra_bits=6))
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
        model = Wide_ResNet(10, 4, 0.0, initial_stride=2, N=8, f=False, r=3, num_classes=10)
        num_params1 = sum([param.numel() for param in model.parameters()])
        model = Wide_ResNet(10, 4.67, 0.0, initial_stride=2, N=1, f=False, r=0, num_classes=10)
        num_params2 = sum([param.numel() for param in model.parameters()])
        print(num_params1,num_params2)
        # cfg_spec = argupdated_config(makeTrainer.__kwdefaults__)
        # cfg_spec = copy.deepcopy(cfg_spec)
        # cfg_spec.update({
        #     'd':[500,1000,2000,3000,4000,6000,8000,16000],
        #     'num_epochs':60,'lr':3e-3, 'expt':['','rot'],
        #     'dataset':[MNIST],
        #     'projector':CombinedRDKronFiLM,
        # })
        
        # name = 'equivariant_all5_nonrot'
        # thestudy = Study(Trial,cfg_spec,study_name=name,
        #         base_log_dir=cfg_spec['trainer_config'].get('log_dir',None))
        # thestudy.run(ordered=True)
        # thestudy.results_df().to_csv(f'{name}.csv')
        # print(thestudy.covariates())
        # print(thestudy.outcomes)
