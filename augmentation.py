import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import cv2
import numpy as np
import os
def read_fn(x):
    return x
#==================================================================================================
def get_transforms(*, data, CFG):
    """
    get_transforms functions to return the augmentations during training
    """

    if data == "train":
        aug_list = [A.Resize(CFG.width, CFG.height, p=1),]
        if CFG.cutout:
            aug_list.append(A.CoarseDropout( p=0.5, max_holes=20, max_height=40, max_width=40, min_holes=10, min_height=20, min_width=20))
        if CFG.base_aug:
            aug_list.append(A.HorizontalFlip(p=0.5))
            aug_list.append(A.ShiftScaleRotate(p=0.5))
            aug_list.append(A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.5))
        aug_list.append(A.GaussNoise(p=0.5, var_limit=(10.0, 50.0)))
        aug_list.append(A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ))
        aug_list.append(ToTensorV2())
        return A.Compose(aug_list)

    elif data == "valid":
        aug_list = [A.Resize(CFG.width, CFG.height, p=1),]
        aug_list.append(A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]))
        aug_list.append(ToTensorV2())
        return A.Compose(aug_list)
    
def get_transforms_with_custom_augmentations(*, data, CFG):
    """
    get_transforms functions to return the augmentations during training
    """
    resize = A.Resize(CFG.width, CFG.height, p=1)
    normalize = A.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    if data == "train":
        aug_list = [resize]

        # Color augmentation with default settings
        color_aug = A.Compose([
            A.RandomBrightnessContrast(p=CFG.p),
            A.HueSaturationValue(p=CFG.p)
        ])
        
        # Multi-crop augmentation
        multi_crop_aug = A.Compose([
            A.RandomResizedCrop(CFG.width, CFG.height, scale=(0.05, 0.4), p=CFG.p),  # local crop
            A.RandomResizedCrop(CFG.width, CFG.height, scale=(0.5, 1.0), p=CFG.p)   # global crop
        ])
        
        # Geometric augmentation with default settings
        geometric_aug = A.Compose([
            A.ShiftScaleRotate(p=CFG.p),
            A.Affine(shear=CFG.M_std * 10, p=CFG.p)
        ])
        
        # Augmentation 리스트에 추가
        aug_list.extend([
            color_aug,
            multi_crop_aug,
            geometric_aug,
            normalize,      
            ToTensorV2()
        ])
        
        return A.Compose(aug_list)