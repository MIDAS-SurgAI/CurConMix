import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import random

class TripletModel(nn.Module):
    def __init__(self, CFG, model_name, pretrained=True):
        super().__init__()
        self.CFG = CFG
        if pretrained:
            print(f'Using default pre-trained weights for initialized fine-tuning model')
        self.model = timm.create_model(model_name, pretrained=True)

        if hasattr(self.model, 'fc'):
            n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, 'head'):  
            n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise AttributeError("The model does not have 'fc' or 'head' attributes.")
        self.head = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        feature = self.model(x)
        x = self.head(feature)
   
        return x, feature


class supcon_Model(nn.Module):
    def __init__(self, CFG, model_name, pretrained=True):
        super().__init__()
        self.CFG = CFG
        self.model = timm.create_model(model_name, pretrained=pretrained) 

        if hasattr(self.model, 'fc'):
            n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, 'head'):
            n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise AttributeError("The model does not have 'fc' or 'head' attributes.")
 
        self.head = torch.nn.Sequential(
             torch.nn.Linear(n_features, 2048),
             nn.BatchNorm1d(2048),
             nn.ReLU(inplace=True),
             torch.nn.Linear(2048, 128),)

    def forward(self, x):
        feature = self.model(x)
        x = self.head(feature)
   
        return x

class SupConHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SupConHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, output_dim),
        )

    def forward(self, x):
        return self.head(x)

class FeatureSupConModel(nn.Module):
    def __init__(self, CFG, model_name, pretrained=True):
        super(FeatureSupConModel, self).__init__()
        self.CFG = CFG
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if hasattr(self.model, 'fc'):
            n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, 'head'):
            n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise AttributeError("The model does not have 'fc' or 'head' attributes.")

        self.head = SupConHead(n_features, 128)

    def forward(self, x):
        feature = self.model(x)
        x = self.head(feature)
        return x
