import time
import numpy as np
from torch.cuda import amp
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from functools import wraps, partial

import tqdm
from augmentation import get_transforms
from dataset import *
from models import *
import torch.nn.utils as nn_utils

# Helper functions
class AverageMeter(object):
    def __init__(self):
        """
        Initialize AverageMeter attributes.
        """
        self.reset()

    def reset(self):
        """
        Reset the meter to its initial state.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.

        Parameters:
        - val (float): Current value to be added to the running sum.
        - n (int): Number of occurrences of the value (default is 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(
    train_loader, model, CFG, criterion, optimizer, epoch, scheduler, device, scaler
):
    """
    Training loop function: loops over the dataloader.

    Parameters:
    - train_loader (DataLoader): DataLoader for training data.
    - model (nn.Module): PyTorch model to be trained.
    - CFG (Namespace): Configuration object containing hyperparameters.
    - criterion (nn.Module): Loss function for training.
    - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    - epoch (int): Current epoch number.
    - scheduler: Learning rate scheduler.
    - device (torch.device): Device (GPU or CPU) on which the training is performed.
    - scaler (torch.cuda.amp.GradScaler): PyTorch AMP scaler for mixed precision training.

    Returns:
    float: Average loss per epoch.
    """
    losses = AverageMeter()
    global_step = 0
    print('criterion', criterion)
    model.train()
    
    for step, data in enumerate(train_loader):
        images, labels = data
        batch_size = labels.size(0)
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        with amp.autocast():
            y_preds, feature = model(images)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            global_step += 1
            scaler.update()

    return losses.avg

def valid_fn(valid_loader, model, CFG, criterion, device):
    """
    Run validation over the validation set.

    Args:
        valid_loader (DataLoader): Validation data loader.
        model (nn.Module): Model to evaluate.
        CFG (object): Configuration object.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use.

    Returns:
        tuple: Average loss and predicted probabilities (numpy array).
    """
    losses = AverageMeter()
    model.eval()
    preds = []

    for step, data in enumerate(valid_loader):
        images, labels = data
        batch_size = labels.size(0)
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            y_preds, _ = model(images)

        loss = criterion(y_preds[:, :100], labels[:, :100])
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to("cpu").numpy())

    predictions = np.concatenate(preds)

    if CFG.gradient_accumulation_steps > 1:
        loss = loss / CFG.gradient_accumulation_steps

    return losses.avg, predictions


def inference_fn(valid_loader, model, device, CFG):
    """
    Run inference on the validation set.

    Args:
        valid_loader (DataLoader): DataLoader for inference.
        model (nn.Module): Model for prediction.
        device (torch.device): Device to use.
        CFG (object): Configuration object.

    Returns:
        np.ndarray: Predicted probabilities.
    """
    model.eval()
    preds = []

    for images in valid_loader:
        images = images.to(device)

        with torch.no_grad():
            y_preds, _ = model(images)

        preds.append(y_preds.sigmoid().cpu().numpy())

    predictions = np.concatenate(preds)
    return predictions

def apply_self_distillation(fold, train_folds, CFG):
    """
    Apply self-distillation to the student model.

    Parameters:
    - fold: Current fold index.
    - train_folds: Training folds DataFrame.
    - CFG: Configuration object.

    Returns:
    pd.DataFrame: Updated training folds DataFrame after applying self-distillation.
    """
    # Read soft labels
    teacher_name = CFG.teacher_exp
    target_size = CFG.target_size
    # teacher_name = teacher_name.replace('student', 'teacher')
    if "challenge" in CFG.split_selector:
        soft_labels_path = os.path.join(
        CFG.output_dir,
        f"softlabels/sl_{CFG.model_name[:8]}_{target_size}_{teacher_name}.csv",
    )
    else:
        soft_labels_path = os.path.join(
        CFG.output_dir,
        f"softlabels/sl_f{fold}_{CFG.model_name[:8]}_{target_size}_{teacher_name}.csv",
    )
    train_softs = pd.read_csv(soft_labels_path)

    # Get the index of triplet 0 and soft label 0
    tri0_idx = train_folds.columns.get_loc("tri0")
    sl_pred0_idx = train_softs.columns.get_loc("0")

    # Reorder train soft labels to match the train labels order
    train_softs = train_softs.merge(train_folds[["image_id"]], on="image_id", how="right")

    # Apply self-distillation: Default SD=1
    tri_range = slice(tri0_idx, tri0_idx + target_size)
    sl_range = slice(sl_pred0_idx, sl_pred0_idx + target_size)
    train_folds.iloc[:, tri_range] = (
        train_folds.iloc[:, tri_range].values * (1 - CFG.SD)
        + train_softs.iloc[:, sl_range].values * CFG.SD
    )
    print("Soft-labels loaded successfully!")

    # Apply label smoothing
    if CFG.smooth:
        train_folds.iloc[:, tri_range] = (
            train_folds.iloc[:, tri_range] * (1.0 - CFG.ls) + 0.5 * CFG.ls
        )
    return train_folds
#======================================


#=================================================================================================
def get_dataframes(folds, fold):
    """
    Split the provided DataFrame into train and validation sets based on the given fold index.

    Parameters:
    - folds (pd.DataFrame): DataFrame containing the data with a "fold" column for splitting.
    - fold (int): Fold index used for validation set, while the rest are used for training.

    Returns:
    - train_folds (pd.DataFrame): DataFrame for the training set.
    - valid_folds (pd.DataFrame): DataFrame for the validation set.
    - temp (pd.DataFrame): Temporary DataFrame for metric computation.

    """
    # Get train and valid indexes
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    # Get train dataset
    train_folds = folds.loc[trn_idx].reset_index(drop=True)

    # Get valid dataset
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    # Temporary df to compute the metric
    temp = folds.loc[val_idx].reset_index(drop=True)

     # Print the number of samples in train and valid datasets
    print(f"Number of training samples: {len(train_folds)}")
    print(f"Number of validation samples: {len(valid_folds)}")
    
    return train_folds, valid_folds, temp

#=================================================================================================

def train_supcon(
    train_loader, model, CFG, criterion, optimizer, epoch, scheduler, device, scaler
):
    """
    Run supervised contrastive training for one epoch.

    Args:
        train_loader (DataLoader): Training data loader.
        model (nn.Module): Model to train.
        CFG (object): Configuration object.
        criterion (nn.Module): SupCon loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epoch (int): Current epoch.
        scheduler: Learning rate scheduler.
        device (torch.device): Device to use.
        scaler (torch.cuda.amp.GradScaler): AMP gradient scaler.

    Returns:
        float: Average training loss for the epoch.
    """
    losses = AverageMeter()
    model.train()

    for step, data in enumerate(train_loader):
        images, labels = data
        images_1, images_2 = images
        label1, label2, label3 = labels
        batch_size = label1.size(0)

        images = torch.cat((images_1, images_2), dim=0).to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        label3 = label3.to(device)
        labels_list = [label1, label2, label3]
        num = 2  # Using triplet labels
        labels = labels_list[num].to(device)

        optimizer.zero_grad()

        with amp.autocast():
            features = model(images)
            f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
            f1 = torch.nn.functional.normalize(f1, dim=1)
            f2 = torch.nn.functional.normalize(f2, dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg


def train_feature_batch_supcon(
    train_loader, model, CFG, criterion, optimizer, epoch, scheduler, device, scaler
):
    """
    Run training for one epoch using feature-level contrastive learning.

    Args:
        train_loader (DataLoader): Training data loader.
        model (nn.Module): Model to train.
        CFG (object): Configuration object.
        criterion (nn.Module): Feature-level SupCon loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epoch (int): Current epoch.
        scheduler: Learning rate scheduler.
        device (torch.device): Device to use.
        scaler (torch.cuda.amp.GradScaler): AMP gradient scaler.

    Returns:
        float: Average training loss for the epoch.
    """
    losses = AverageMeter()
    model.train()

    for step, data in enumerate(train_loader):
        images, labels, contrast_features, contrast_labels = data
        images = images.to(device)
        contrast_features = contrast_features.to(device)
        labels = labels.to(device)
        contrast_labels = contrast_labels.to(device)

        batch_size = images.size(0)
        optimizer.zero_grad()

        with amp.autocast():
            features = model(images)
            features = torch.nn.functional.normalize(features, dim=1)

            contrast_features_flat = contrast_features.view(-1, contrast_features.shape[-1])
            contrast_features_flat = model.head(contrast_features_flat)
            contrast_features_flat = contrast_features_flat.view(contrast_features.shape[0], contrast_features.shape[1], -1)
            contrast_features_flat = torch.nn.functional.normalize(contrast_features_flat, dim=2)

            loss = criterion(features, contrast_features_flat, labels, contrast_labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

    return losses.avg

################################################################################################################################################################
def compute_features(model, dataloader, device):
    model.eval()
    features = []
    labels_list = []
    with torch.no_grad():
        for images, file_name in tqdm.tqdm(dataloader, desc="Extracting Features", unit="batch"):
            images = images.to(device)
            feature = model.model(images)
            features.append(feature.cpu())
    features = torch.cat(features)
    return features.numpy()

def compute_cosine_similarity_matrix(features):
    features_normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    cos_sim_matrix = np.dot(features_normalized, features_normalized.T)
    print(f'features.shape:{features.shape}, cosine_matrix.shape: {cos_sim_matrix.shape}')
    return cos_sim_matrix
################################################################################################################################################################

def train_mixup(
    train_loader, model, CFG, criterion, optimizer, epoch, scheduler, device, scaler
):
    """
    Run one epoch of mixup training.

    Args:
        train_loader (DataLoader): Training data loader.
        model (nn.Module): Model to train.
        CFG (object): Configuration object.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epoch (int): Current epoch.
        scheduler: Learning rate scheduler.
        device (torch.device): Device to use.
        scaler (torch.cuda.amp.GradScaler): AMP gradient scaler.

    Returns:
        float: Average training loss for the epoch.
    """
    losses = AverageMeter()
    global_step = 0
    model.train()

    for step, data in enumerate(train_loader):
        images, labels = data
        batch_size = labels.size(0)

        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        mixed_images, label_a, label_b, lamb = mixup_data(images, labels, alpha=CFG.alpha, device=device)

        with amp.autocast():
            y_preds, _ = model(mixed_images)
            loss = mixup_criterion(criterion, y_preds, label_a, label_b, lamb)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            global_step += 1
            scaler.update()

    return losses.avg

def mixup_data(x, y, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
