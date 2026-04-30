# modify on top of https://github.com/xavihart/Diff-PGD

from dataset import get_normalize_layer
import torchvision
from transformers import BeitForImageClassification
import torch


def get_archs(arch, dataset='imagenet'):
    if dataset == 'imagenet':
        if arch == 'resnet50':
            model = torchvision.models.resnet50(weights="DEFAULT")
        
        elif arch == 'resnet152':
            model = torchvision.models.resnet152(weights="DEFAULT")
            
        elif arch == 'wrn50':
            model = torchvision.models.wide_resnet50_2(weights="DEFAULT")

        elif arch == 'vit':
            model = torchvision.models.vit_b_16(weights='DEFAULT')

        elif arch == 'swin_b':
            model = torchvision.models.swin_b(weights='DEFAULT')

        elif arch == 'convnext_b':
            model = torchvision.models.convnext_base(weights='DEFAULT')
    
    normalize_layer = get_normalize_layer(dataset)
    
    return torch.nn.Sequential(normalize_layer, model)

    
