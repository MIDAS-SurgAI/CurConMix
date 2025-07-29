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
    