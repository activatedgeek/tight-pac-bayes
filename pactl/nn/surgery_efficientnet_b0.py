import torch
from timm.models import register_model, create_model


def pretrained_model_surgery(model, num_classes=1000):
    new_fc = torch.nn.Linear(model.classifier.in_features, num_classes)
    # initialize fc with old fc weights
    new_fc.weight.data = model.classifier.weight.data[:num_classes]
    new_fc.bias.data = model.classifier.bias.data[:num_classes]
    model.classifier = new_fc
    return model


@register_model
def surgery_efficientnet_b0(num_classes=1000, **net_cfg):
    net_cfg = {**net_cfg, 'num_classes': 1000,
               'model_name': 'efficientnet_b0', 'pretrained': True}
    return pretrained_model_surgery(create_model(**net_cfg), num_classes)


@register_model
def EfficientNet_b1(num_classes=1000, **net_cfg):
    net_cfg = {**net_cfg, 'num_classes': 1000,
               'model_name': 'efficientnet_b1', 'pretrained': True}
    return pretrained_model_surgery(create_model(**net_cfg), num_classes)
