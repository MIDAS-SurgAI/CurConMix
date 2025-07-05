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

        # Check if 'fc' exists in the model, if so replace it with nn.Identity()
        if hasattr(self.model, 'fc'):
            n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, 'head'):  # If 'fc' doesn't exist, check for 'head'
            n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise AttributeError("The model does not have 'fc' or 'head' attributes.")

        # Add a new linear head for classification
        self.head = nn.Linear(n_features, CFG.target_size)

            
    def forward(self, x):
        feature = self.model(x)
        x = self.head(feature)
   
        return x, feature

class TripletModel_with_MLP(nn.Module):
    def __init__(self, CFG, model_name, pretrained=True):
        super(TripletModel_with_MLP, self).__init__()
        self.CFG = CFG
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # Remove the last layer of the backbone model
        if hasattr(self.model, 'fc'):
            n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, 'head'):
            n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise AttributeError("The model does not have 'fc' or 'head' attributes.")

        # Initialize the SupConHead as in pre-training
        self.head = SupConHead(n_features, 128)

        # Add a classification head for fine-tuning
        self.classification_head = nn.Linear(128, CFG.target_size)

    def forward(self, x):
        feature = self.model(x)
        x = self.head(feature)
        logits = self.classification_head(x)
        return logits, x


class Model_feature_extractor(nn.Module):
    def __init__(self, CFG, model_name, pretrained=True):
        super().__init__()
        self.CFG = CFG
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # Check if 'fc' exists in the model, if so replace it with nn.Identity()
        if hasattr(self.model, 'fc'):
            n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, 'head'):  # If 'fc' doesn't exist, check for 'head'
            n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise AttributeError("The model does not have 'fc' or 'head' attributes.")
      
    def forward(self, x):
        feature = self.model(x)
 
   
        return feature


class supcon_Model(nn.Module):
    def __init__(self, CFG, model_name, pretrained=True):
        super().__init__()
        self.CFG = CFG
        self.model = timm.create_model(model_name, pretrained=pretrained) 
      

        # Check if 'fc' exists in the model, if so replace it with nn.Identity()
        if hasattr(self.model, 'fc'):
            n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, 'head'):  # If 'fc' doesn't exist, check for 'head'
            n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise AttributeError("The model does not have 'fc' or 'head' attributes.")
 
        # Linear head for the classification task
     
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
        # 백본 모델의 마지막 레이어 제거
        if hasattr(self.model, 'fc'):
            n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, 'head'):
            n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            raise AttributeError("The model does not have 'fc' or 'head' attributes.")

        # 헤드 초기화
        self.head = SupConHead(n_features, 128)

    def forward(self, x):
        feature = self.model(x)
        x = self.head(feature)
        return x
