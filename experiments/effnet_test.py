from pactl.nn.projectors import FiLMLazyRandom, IDModule
import torch
import timm

def pretrained_model_surgery(model,num_targets):
    # make all parameters frozen except for the last layer and BN parameters
    new_fc = torch.nn.Linear(model.classifier.in_features,num_targets)
    # initialize fc with old fc weights
    new_fc.weight.data = model.classifier.weight.data[:num_targets]
    new_fc.bias.data = model.classifier.bias.data[:num_targets]
    model.classifier = new_fc
    return model


modelname='efficientnet_b0'
model = timm.create_model(modelname, pretrained=True)
model = pretrained_model_surgery(model,10)
model = IDModule(model,FiLMLazyRandom,1000)#.to(device)