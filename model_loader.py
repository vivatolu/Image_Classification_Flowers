import pandas as pd
import numpy as np

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict





def load_model(args, model_name='alexnet', hidden_layer=1024, train=True): #model_name has a default value of 'alexnet' and hidden layer has a default value of 1024 if no value is provided
    if args.arch:
        model_name = args.arch

    if args.hidden_units:
        hidden_layer = args.hidden_units

    in_features_dict = {
        'vgg13': [25088, 4096],
        'alexnet': [9216, 4096],
        'resnet50': [2048],
        'vgg16': [25088, 4096],
    }

    if model_name not in in_features_dict.keys():
        model_name = 'alexnet' #Default to Alexnet if provided arch is invalid

    model = getattr(models, model_name)(pretrained=True)
    linear_layer_name = list(model.named_children())[-1][0]

    in_features = in_features_dict[model_name]

    for params in model.parameters():
        params.requires_grad = False

    if train: #Handle for Train and test cases
        if len(in_features) == 1:
            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(in_features[0], hidden_layer)),
                ('relu1', nn.ReLU()),
                ('dropout1', nn.Dropout(p=0.3)),
                ('fc2', nn.Linear(hidden_layer, 102)),
                ('output', nn.LogSoftmax(dim=1))

            ]))
        elif len(in_features) == 2:
            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(in_features[0], in_features[1])),
                ('relu1', nn.ReLU()),
                ('dropout1', nn.Dropout(p=0.3)),
                ('fc2', nn.Linear(in_features[1], hidden_layer)),
                ('relu2', nn.ReLU()),
                ('dropout2', nn.Dropout(p=0.3)),
                ('fc3', nn.Linear(hidden_layer, 102)),
                ('output', nn.LogSoftmax(dim=1))

            ]))

        #model.classifier = classifier
        setattr(model, linear_layer_name, classifier)

    return model, model_name, (in_features, [hidden_layer])
