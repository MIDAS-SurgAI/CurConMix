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
    # m = nn.Sigmoid()
    # Start variables
    losses = AverageMeter()
    global_step = 0
    print('criterion', criterion)
    # Switch to train mode
    model.train()
    if CFG.extract_feature:
        criterion, mse_loss = criterion


   
    for step, data in enumerate(train_loader):

        # Get the batch of images and labels
        if CFG.extract_feature:
            images, labels, y_feature = data
            y_feature = y_feature.to(device)
        else:
            images, labels = data

        batch_size = labels.size(0)

        # Start the optimizer
        optimizer.zero_grad()

        # Send the images and labels to the GPU
        images = images.to(device)
        labels = labels.to(device)

        # Apply mixed precision
        with amp.autocast():

            # Get the predictions
            y_preds, feature = model(images)
        
            # Compute the loss on multitask or triplets only

            loss = criterion(y_preds, labels)
  
            if CFG.extract_feature:
                loss_mse=mse_loss(feature,y_feature )
              
                loss = loss+loss_mse
        # print('loss: ',loss.item(), 'step:', step)

            
        
        # Update the loss
        losses.update(loss.item(), batch_size)
 
        # Backward pass
        scaler.scale(loss).backward()

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            # Perform optimization step only after accumulating gradients for a specified number of steps
            scaler.step(optimizer)
            global_step += 1
            scaler.update()


    return losses.avg

def valid_fn(valid_loader, model, CFG, criterion, device):
    """
    Validation loop over the validation DataLoader.

    Parameters:
    - valid_loader (DataLoader): DataLoader for validation data.
    - model (Module): PyTorch model to be evaluated.
    - CFG (object): Configuration object containing hyperparameters.
    - criterion (Module): Loss function for validation.
    - device (object): Device (GPU or CPU) on which the validation is performed.

    Returns:
    tuple: Average loss, predictions (numpy array).

    """
    losses = AverageMeter()

    # Switch to evaluation mode
    model.eval()
    

    # Start a list to store the predictions
    preds = []
   
    # Loop over the DataLoader
    for step, data in enumerate(valid_loader):
        # Get the images and labels
        images, labels = data
        batch_size = labels.size(0)

        # Send images and labels to GPU
        images, labels = images.to(device), labels.to(device)

        # Eval mode
        with torch.no_grad():
            # Run the model on the validation set
            y_preds, _ = model(images)

        # Compute the validation loss on the triplets only
        loss = criterion(y_preds[:, :100], labels[:, :100])

        # Update the loss
        losses.update(loss.item(), batch_size)

        # Update predictions
        preds.append(y_preds.sigmoid().to("cpu").numpy())

    # Concatenate predictions
    predictions = np.concatenate(preds)

    if CFG.gradient_accumulation_steps > 1:
        loss = loss / CFG.gradient_accumulation_steps
  

    return losses.avg, predictions


def inference_fn(
    valid_loader,
    model,
    device,CFG
):
    """
    Inference loop over the validation DataLoader.

    Parameters:
    - valid_loader (DataLoader): DataLoader for inference data.
    - model (Module): PyTorch model for making predictions.
    - device (object): Device (GPU or CPU) on which the inference is performed.

    Returns:
    np.ndarray: Predictions as a NumPy array.

    """
    model.eval()



    # Start a list to store the predictions
    preds = []
    # Loop over the DataLoader
    for step, images in enumerate(valid_loader):
        # Measure data loading time
        images = images.to(device)

        # Compute predictions and extract features
        with torch.no_grad():
            y_preds, feature = model(images)

        # Store the predictions and features
        preds.append(y_preds.sigmoid().to("cpu").numpy())

    # Concatenate predictions and features
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

def get_dataloaders(train_folds, valid_folds, CFG):
    """
    Get PyTorch dataloaders for training and validation datasets.

    Parameters:
    - train_folds (pd.DataFrame): DataFrame containing training data.
    - valid_folds (pd.DataFrame): DataFrame containing validation data.
    - CFG (object): Configuration object containing hyperparameters.

    Returns:
    - DataLoader: PyTorch DataLoader for the training dataset.
    - DataLoader: PyTorch DataLoader for the validation dataset.

    """

    # PyTorch datasets
    # Apply train augmentations

    train_dataset = TrainDataset(
        train_folds, CFG, transform=get_transforms(data="train", CFG=CFG)
        )

    # Apply validation augmentations
   
        
    valid_dataset = TrainDataset(
        valid_folds, CFG, transform=get_transforms(data="valid", CFG=CFG)
    )

    # PyTorch train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=CFG.shuffle,
        num_workers=CFG.nworkers,
        pin_memory=True,
        drop_last=True,
    )

    # PyTorch valid dataloader
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.nworkers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, valid_loader


def get_dataloaders_train(train_folds, CFG):
    """
    Get PyTorch dataloaders for training and validation datasets.

    Parameters:
    - train_folds (pd.DataFrame): DataFrame containing training data.
    - valid_folds (pd.DataFrame): DataFrame containing validation data.
    - CFG (object): Configuration object containing hyperparameters.

    Returns:
    - DataLoader: PyTorch DataLoader for the training dataset.
    - DataLoader: PyTorch DataLoader for the validation dataset.

    """

    # PyTorch datasets
    # Apply train augmentations

    train_dataset = TrainDataset(
        train_folds, CFG, transform=get_transforms(data="valid", CFG=CFG)
        )

    # Apply validation augmentations
   
        
    # PyTorch train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.nworkers,
        pin_memory=True,
        drop_last=False,
    )


    return train_loader

def get_dataloaders_train_using_custom_aug(train_folds, transform, CFG):
    """
    Get PyTorch dataloaders for training and validation datasets.

    Parameters:
    - train_folds (pd.DataFrame): DataFrame containing training data.
    - valid_folds (pd.DataFrame): DataFrame containing validation data.
    - CFG (object): Configuration object containing hyperparameters.

    Returns:
    - DataLoader: PyTorch DataLoader for the training dataset.
    - DataLoader: PyTorch DataLoader for the validation dataset.

    """

    # PyTorch datasets
    # Apply train augmentations
    print('transform: ', transform)
    train_dataset = TrainDataset(
        train_folds, CFG, transform=transform
        )

    # Apply validation augmentations
   
        
    # PyTorch train dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.nworkers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader

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


def get_dataframes_split(folds):
    """
    Split the provided DataFrame into train and validation sets based on the given fold index.

    Parameters:
    - folds (pd.DataFrame): DataFrame containing the data with a "fold" column for splitting.

    Returns:
    - train_folds (pd.DataFrame): DataFrame for the training set.
    - valid_folds (pd.DataFrame): DataFrame for the validation set.
    - temp (pd.DataFrame): Temporary DataFrame for metric computation.

    """
    # Get train and valid indexes
    trn_idx = folds[folds["fold"] == 'train'].index
    val_idx = folds[folds["fold"] == 'val'].index
    test_idx = folds[folds["fold"] == 'test'].index

    # Get train dataset
    train_folds = folds.loc[trn_idx].reset_index(drop=True)

    # Get valid dataset
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    #Get test dataset
    test_folds = folds.loc[test_idx].reset_index(drop=True)

    # # Temporary df to compute the metric
    # temp = folds.loc[val_idx].reset_index(drop=True)

    return train_folds, valid_folds, test_folds
#=================================================================================================
#=================================================================================================

def train_supcon(
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
    # m = nn.Sigmoid()
    # Start variables
    losses = AverageMeter()
    global_step = 0

    # Switch to train mode
    model.train()

   

    for step, data in enumerate(train_loader):
     
        # Get the batch of images and labels
        images, labels = data
        images_1, images_2 = images
        label1, label2, label3 = labels
        batch_size = label1.size(0)

        images = torch.cat((images_1, images_2), dim=0).to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        label3 = label3.to(device)
        labels_list = [label1, label2, label3]
        
        num = 2 ## 마지막 label 사용 => label3: IVT로 설정해서 사용함 
        labels = labels_list[num].to(device)

        # Start the optimizer
        optimizer.zero_grad()

        # Apply mixed precision
        with amp.autocast():

            # Get the predictions
            features = model(images)
            f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)

            f1 = torch.nn.functional.normalize(f1, dim=1)
            f2 = torch.nn.functional.normalize(f2, dim=1)

            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            loss = criterion(features, labels)
 
        # Update the loss
        losses.update(loss.item(), batch_size)
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # if (step + 1) % CFG.gradient_accumulation_steps == 0:
        #     # Perform optimization step only after accumulating gradients for a specified number of steps
        #     scaler.step(optimizer)
        #     global_step += 1
        #     scaler.update()


    return losses.avg

def train_feature_batch_supcon(
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
    # Start variables
    losses = AverageMeter()
    global_step = 0

    # Switch to train mode
    model.train()
    
    for step, data in enumerate(train_loader):
        # Unpack the data from the DataLoader
        images, labels, contrast_features, contrast_labels = data

        # Move data to the specified device
        images = images.to(device)
        contrast_features = contrast_features.to(device)
        labels = labels.to(device)
        contrast_labels = contrast_labels.to(device)

        batch_size = images.size(0)

        # Start the optimizer
        optimizer.zero_grad()

        # Apply mixed precision
        with amp.autocast():
            # Get the features from the model
            features = model(images)  # [batch_size, feature_dim]

            # Normalize the features
            features = torch.nn.functional.normalize(features, dim=1)                
            contrast_features_flat = contrast_features.view(-1, contrast_features.shape[-1])

            contrast_features_flat = model.head(contrast_features_flat)
            contrast_features_flat = contrast_features_flat.view(contrast_features.shape[0], contrast_features.shape[1], -1)
            # Normalize the contrast_features
            contrast_features_flat = torch.nn.functional.normalize(contrast_features_flat, dim=2)


            # Compute the loss
            loss = criterion(features, contrast_features_flat, labels, contrast_labels)

        # Update the loss
        losses.update(loss.item(), batch_size)

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Optionally update the learning rate scheduler
        if scheduler is not None:
            scheduler.step()

    return losses.avg

################################################################################################################################################################
def compute_features(model, dataloader, device):
    """
    모델의 backbone을 사용하여 전체 데이터셋에 대한 feature를 계산합니다.
    """
    model.eval()
    features = []
    labels_list = []
    with torch.no_grad():
        for images, file_name in tqdm.tqdm(dataloader, desc="Extracting Features", unit="batch"):
            images = images.to(device)
            # 모델의 backbone을 통해 feature를 추출합니다.
            feature = model.model(images)
            features.append(feature.cpu())
    features = torch.cat(features)
    return features.numpy()

def compute_cosine_similarity_matrix(features):
    """
    주어진 features에 대한 cosine similarity matrix를 계산합니다.
    """
    features_normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
    cos_sim_matrix = np.dot(features_normalized, features_normalized.T)
    print(f'features.shape:{features.shape}, cosine_matrix.shape: {cos_sim_matrix.shape}')
    return cos_sim_matrix
################################################################################################################################################################
def train_curriculum_supcon(
    train_loader, model, CFG, criterion, optimizer, epoch, scheduler, device, scaler
):
    losses = AverageMeter()
    global_step = 0

    # 모델을 학습 모드로 전환
    model.train()
    prev_num = -1

    for step, data in enumerate(train_loader):
        # image_pairs_list: [(image, mixup_image1, mixup_image2), ...]
        image_pairs_list, labels = data  
        
        # Curriculum learning setting: epoch에 따라 num 설정
        if CFG.epochs == 6:
            num = min(epoch // 2, len(image_pairs_list) - 1) ## epoch 가 2번씩 반복 || CFG.epochs = 6 일 때 
        elif CFG.epochs == 4 or CFG.epochs == 3: 
            num = min(epoch, len(image_pairs_list) - 1) ## epoch 가 1번씩 반복 || CFG.epochs= 4 일 때        
        
        # num = min(epoch // 2, len(image_pairs_list) - 1) ## epoch 가 2번씩 반복 || CFG.epochs = 6 일 때 
        # num = min(epoch, len(image_pairs_list) - 1) ## epoch 가 1번씩 반복 || CFG.epochs= 4 일 때  

        if num != prev_num:
            print(f'Curriculum Learning Label 변경: {prev_num} => {num}')
            prev_num = num 
        
        # 이미지 및 레이블 가져오기
        images_1, mixup_image1, mixup_image2 = image_pairs_list[num]
        images_1 = images_1.to(device)
        mixup_image1 = mixup_image1.to(device)
        mixup_image2 = mixup_image2.to(device)

        labels = labels[num].to(device)
        batch_size = images_1.size(0)

        # Mixup 연산 수행
        lam = np.random.beta(CFG.alpha, CFG.alpha)
        mixed_images = lam * mixup_image1 + (1 - lam) * mixup_image2

        # Concatenate 원본 이미지와 Mixup 이미지
        images = torch.cat((images_1, mixed_images), dim=0)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # Mixed precision 적용
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

def train_Naive_Curriculum_Supcon(
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
    # m = nn.Sigmoid()
    # Start variables
    losses = AverageMeter()
    global_step = 0

    # Switch to train mode
    model.train()
    
    for step, data in enumerate(train_loader):
        # Get the batch of images and labels
        images, labels = data
        images_1, images_2 = images
        label1, label2, label3 = labels
        batch_size = label1.size(0)

        images = torch.cat((images_1, images_2), dim=0).to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        label3 = label3.to(device)
        labels_list = [label1, label2, label3]
        num = 2 if epoch >= len(labels_list) else epoch
        # print(f'using {CFG.label_order[num]} label')
        labels = labels_list[num].to(device)
        # Start the optimizer
        optimizer.zero_grad()

        # Apply mixed precision
        with amp.autocast():

            # Get the predictions
            features = model(images)
            f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
            f1 = torch.nn.functional.normalize(f1, dim=1)
            f2 = torch.nn.functional.normalize(f2, dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features, labels)
 
        # Update the loss
        losses.update(loss.item(), batch_size)
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f'Using {CFG.label_order[num]} label')
    return losses.avg

def check_for_nan_or_inf(tensor, tensor_name=""):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"Tensor {tensor_name} contains NaN or Inf values.")

def inspect_gradients(model):
    """
    Print the gradient norms for each parameter in the model.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < 1e-6:
                print(f"Warning: Vanishing gradient detected in {name}, norm: {grad_norm}")
            elif grad_norm > 1e2:  # Adjust this threshold based on your model
                print(f"Warning: Exploding gradient detected in {name}, norm: {grad_norm}")
            else:
                print(f"Layer: {name}, Gradient Norm: {grad_norm}")
        else:
            print(f"Layer: {name} has no gradients (might be frozen or not involved in the loss calculation)")
    assert print('stop')

def train_mixup(
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
    # m = nn.Sigmoid()
    # Start variables
    losses = AverageMeter()
    global_step = 0
    # Switch to train mode
    model.train()
    for step, data in enumerate(train_loader):
        # Get the batch of images and labels
        images, labels = data
      
        ## Creating a mixup sample 
        batch_size = labels.size(0)

        # Start the optimizer
        optimizer.zero_grad()

        # Send the images and labels to the GPU
        images = images.to(device)
        labels = labels.to(device)

        mixed_images, label_a, label_b, lamb = mixup_data(images, labels, alpha=CFG.alpha,device=CFG.device)

        # Apply mixed precision
        with amp.autocast():
            # Get the predictions
            y_preds, feature = model(mixed_images)
            # Mixup Loss 
            loss = mixup_criterion(criterion, y_preds, label_a, label_b, lamb)        

        # Update the loss
        losses.update(loss.item(), batch_size)
 
        # Backward pass
        scaler.scale(loss).backward()
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            # Perform optimization step only after accumulating gradients for a specified number of steps
            scaler.step(optimizer)
            global_step += 1
            scaler.update()
    return losses.avg

def train_feature_mixup(
    train_loader, model, CFG, criterion, optimizer, epoch, scheduler, device, scaler
):
    # 손실 추적을 위한 변수 초기화
    losses = AverageMeter()
    global_step = 0
    # 모델을 학습 모드로 전환
    model.train()
    if CFG.extract_feature:
        criterion, mse_loss = criterion

    for step, data in enumerate(train_loader):
       
        images, labels = data
        batch_size = labels.size(0)

        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        # 백본을 통해 피처 추출
        features = model.model(images)

        # 피처에 믹스업 적용
        mixed_features, label_a, label_b, lam = mixup_data(
            features, labels, alpha=CFG.alpha, device=CFG.device
        )

        # 혼합 정밀도 적용
        with amp.autocast():
            # 믹스업된 피처를 분류 헤드에 통과
            y_preds = model.head(mixed_features)

            # 믹스업 손실 계산
            loss = mixup_criterion(criterion, y_preds, label_a, label_b, lam)

        # 손실 업데이트
        losses.update(loss.item(), batch_size)

        # 역전파
        scaler.scale(loss).backward()

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            # 그래디언트 누적 후 옵티마이저 스텝 실행
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

# Define beta distribution
def manifold_mixup_data(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam