
import os
import gc
import time
import pickle
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import neptune.new as neptune
from torch import amp
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import average_precision_score

# Local modules
from models import *
from utils import *
from helper import *
from dataset import *
from losses import *
from sampler import ImbalancedDatasetSampler
from augmentation import *
from preprocess import get_folds
from losses_weighted import *


# =========================
# Neptune Logging Helper
# =========================
def log_neptune(run, key, value, fold=None):
    if fold is not None:
        key = f"{key}{fold}"
    run[key].log(value)


# =========================
# Select Default Transforms
# =========================
def select_transforms(data, CFG):
    print("Using Default Augmentation Methods")
    return get_transforms(data=data, CFG=CFG)


# =========================
# Standard Cross-Validation Training
# =========================
def train_cross_val(CFG):
    start_time = time.time()
    seed_torch(CFG.seed)

    # Set up directories
    summary_dir = os.path.join(CFG.output_dir, 'summary_dir')
    log_folder_dir = os.path.join(CFG.output_dir, CFG.exp)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(log_folder_dir, exist_ok=True)

    if CFG.debug:
        CFG.epochs = 1
        CFG.neplog = False

    # Neptune initialization
    if CFG.neplog:
        run = neptune.init(project=CFG.neptune_project, api_token=CFG.neptune_api_token)
        for key, val in {
            "Model": CFG.model_name, "imsize": CFG.height, "LR": CFG.lr,
            "bs": CFG.batch_size, "Epochs": CFG.epochs, "SD": CFG.SD,
            "T_0": CFG.T_0, "min_lr": CFG.min_lr, "seed": str(CFG.seed),
            "split": CFG.challenge_split, "tsize": CFG.target_size,
            "smooth": CFG.smooth, "exp": CFG.exp
        }.items():
            run[key].log(val)

    folds = get_folds(CFG)
    transform = select_transforms(data="train", CFG=CFG)
    print_training_info(folds, CFG)

    valid_folds_temp = None

    for fold in range(CFG.start_n_fold, CFG.n_fold):
        if fold not in CFG.trn_fold:
            continue

        print(f"\033[92m{'-' * 8} Fold {fold + 1} / {CFG.n_fold}\033[0m")
        fold_log_path = os.path.join(log_folder_dir, f'{CFG.exp}_fold_{fold}.csv')

        with open(fold_log_path, 'w') as log_training:
            model = TripletModel(CFG, CFG.model_name, pretrained=CFG.pretrained).to(CFG.device)

            # Load SSL pretrained weights if specified
            if CFG.pretrained_ssl and CFG.pretrained_exp != 'myexp':
                weight_path = os.path.join(CFG.output_dir, f"checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.pretrained_exp}.pth")
                print('Using Pre-trained Checkpoint:', weight_path)
                ssl_state_dict = torch.load(weight_path, map_location=CFG.device)['model']
                model.load_state_dict(ssl_state_dict, strict=False)

            # Get train and validation data
            train_folds, valid_folds, valid_folds_temp = get_dataframes(folds, fold)
            train_dataset = TrainDataset(train_folds, CFG, transform=get_transforms(data="train", CFG=CFG), fold=fold)
            valid_dataset = TrainDataset(valid_folds, CFG, transform=get_transforms(data="valid", CFG=CFG), fold=fold)

            train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=CFG.shuffle, num_workers=CFG.nworkers, pin_memory=True, drop_last=True)
            valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.nworkers, pin_memory=False, drop_last=False)

            # Optimizer & Scheduler
            optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.epochs + 1, T_mult=1000, eta_min=CFG.min_lr)

            # Loss function
            criterion = nn.BCEWithLogitsLoss(reduction="sum").to(CFG.device) if CFG.loss == 'bce' else MultiLabelSoftMarginLoss(reduction="sum").to(CFG.device)
            scaler = amp.GradScaler()
            best_score = 0.0

            if CFG.mixup:
                print('Using Mixup')

            # Training loop
            for epoch in range(CFG.epochs):
                epoch_start = time.time()

                if CFG.mixup:
                    avg_loss = train_mixup(train_loader, model, CFG, criterion, optimizer, epoch, scheduler, CFG.device, scaler)
                else:
                    avg_loss = train_fn(train_loader, model, CFG, criterion, optimizer, epoch, scheduler, CFG.device, scaler)

                avg_val_loss, preds = valid_fn(valid_loader, model, CFG, criterion, CFG.device)
                scheduler.step()

                valid_folds_temp[[str(c) for c in range(CFG.target_size)]] = preds
                mAP_score = per_epoch_ivtmetrics(valid_folds_temp, CFG)

                log_training.write(f"Epoch: {epoch} Validation Loss: {avg_val_loss:.4f} mAP: {mAP_score:.4f}\n")

                if CFG.neplog:
                    log_neptune(run, "tloss", avg_loss, fold)
                    log_neptune(run, "val_loss", avg_val_loss, fold)
                    log_neptune(run, "cmAP", mAP_score, fold)

                if mAP_score > best_score:
                    best_score = mAP_score
                    save_path = os.path.join(CFG.output_dir, f"checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.pth")
                    torch.save({"model": model.state_dict(), "preds": preds}, save_path)

                print(raw_line.format(epoch, avg_loss, avg_val_loss, mAP_score, (time.time() - epoch_start) / 60))

            # Cleanup
            del model, train_loader, valid_loader
            torch.cuda.empty_cache()
            gc.collect()

    # Final cross-validation metric
    final_mAP = cholect45_ivtmetrics_mAP(valid_folds_temp, CFG)
    print(f"CV: Overall mAP: {final_mAP:.4f}")
    if CFG.neplog:
        run["CV"].log(final_mAP)

    print(f"Training time: {(time.time() - start_time) / 60:.2f} minutes")
#================================================================================================

#================================================================================================

def select_transforms(data, CFG):
    print('Using Default Augmentation Methods')
    return get_transforms(data='train', CFG=CFG)
    

def train_cross_val_SSL(CFG):
    start_time = time.time()
    seed_torch(CFG.seed)

    # Directory setup
    summary_dir = os.path.join(CFG.output_dir, 'summary_dir')
    log_folder_dir = os.path.join(CFG.output_dir, CFG.exp)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(log_folder_dir, exist_ok=True)

    summary_dir_total = os.path.join(log_folder_dir, f'{CFG.exp}_total.csv')
    log_total = open(summary_dir_total, 'w')

    if CFG.debug:
        CFG.epochs = 1
        CFG.neplog = False

    if CFG.neplog:
        run = neptune.init(project=CFG.neptune_project, api_token=CFG.neptune_api_token)
        for key, val in {
            "Model": CFG.model_name, "imsize": CFG.height, "LR": CFG.lr,
            "bs": CFG.batch_size, "Epochs": CFG.epochs, "SD": CFG.SD,
            "T_0": CFG.T_0, "min_lr": CFG.min_lr, "seed": str(CFG.seed),
            "split": CFG.challenge_split, "tsize": CFG.target_size,
            "smooth": CFG.smooth, "exp": CFG.exp
        }.items():
            run[key].log(val)

    folds = get_folds(CFG)
    transform = select_transforms(data="train", CFG=CFG)
    print_training_info(folds, CFG)

    label_key_list = CFG.label_order

    for fold in range(CFG.start_n_fold, CFG.n_fold):
        if fold not in CFG.trn_fold:
            continue

        print(f"\033[92m{'-' * 8} Fold {fold + 1} / {CFG.n_fold}\033[0m")
        fold_log_path = os.path.join(log_folder_dir, f'{CFG.exp}_fold_{fold}.csv')
        log_training = open(fold_log_path, 'w')

        # Model selection
        if CFG.feature_batch and CFG.method in ['supcon', 'curriculum_supcon']:
            model = FeatureSupConModel(CFG, CFG.model_name, pretrained=CFG.pretrained).to(CFG.device)
        else:
            model = supcon_Model(CFG, CFG.model_name, pretrained=CFG.pretrained).to(CFG.device)

        train_folds, _, _ = get_dataframes(folds, fold)

        # Dataset selection
        feature_file_name, matrix_file_name, label_sim_matrix = None, None, None
        if CFG.feature_batch:
            if CFG.Base384:
                base_path = f"E:/Surgical/384SwinT_fold{fold}_"
            elif CFG.tiny_model:
                base_path = f"E:/Surgical/Swin_Tiny224_fold{fold}_"
            elif CFG.base_model:
                base_path = f"C:/Users/kyuhw/Desktop/work/sd_temporal/baseline_train_mixup/SwinB_fold0_"
            else:
                base_path = f"C:/Users/kyuhw/Desktop/work/sd_temporal/baseline_train_mixup/fold{fold}_"

            feature_file_name = base_path + CFG.feature_file_name
            matrix_file_name = base_path.replace("features", "similarity") + CFG.cos_sim_matrix_file_name

            with open(feature_file_name, "rb") as f:
                feature_list = pickle.load(f)
            with open(matrix_file_name, "rb") as f:
                cos_sim_matrix = pickle.load(f)

            if CFG.label_sim:
                label_matrix_file_name = f"E:/Surgical/label_similarity_matrices_t_fold_{fold}.pkl"
                with open(label_matrix_file_name, "rb") as f:
                    label_sim_matrix = pickle.load(f)

            label_list = {
                'i': train_folds['instrument'].tolist(),
                't': train_folds['target'].tolist(),
                'v': train_folds['verb'].tolist(),
                'it': train_folds['inst_target'].tolist(),
                'iv': train_folds['inst_verb'].tolist(),
                'tv': train_folds['target_verb'].tolist(),
                'ivt': train_folds['triplet'].tolist()
            }

            if CFG.feature_mixup:
                dataset_cls = WeightedSupConFeatureMixupBatchDataset if CFG.label_sim else SupConFeatureMixupBatchDataset
            else:
                dataset_cls = SupConFeatureBatchDataset

            train_dataset = dataset_cls(train_folds, CFG, features=feature_list, labels=label_list, cos_sim_matrix=cos_sim_matrix, label_sim_matrix=label_sim_matrix if CFG.label_sim else None, transform=transform)
        else:
            train_dataset = Supcon_TrainDataset(train_folds, CFG, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.nworkers, pin_memory=True, drop_last=True)
        optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.epochs + 1, T_mult=1, eta_min=CFG.min_lr)

        if CFG.ssl_loss != 'supcon':
            criterion = SupConLoss(CFG, temperature=CFG.temp)
        elif CFG.feature_batch:
            print('Using Feature batch Supcon')
            criterion = FeatureBatchdSupConLoss(CFG, temperature=CFG.temp)
        else:
            print('Using Default Supcon Loss')
            criterion = SupConLoss(CFG, temperature=CFG.temp)

        scaler = amp.GradScaler()
        auxiliary_weight = CFG.auxiliary_weight

        print(header_ssl)
        for epoch in range(CFG.epochs):
            if (epoch + 1) % 5 == 0:
                auxiliary_weight -= 1
                print(f"Epoch {epoch + 1}: Decreased auxiliary_weight to {auxiliary_weight}")

            epoch_start = time.time()

            if CFG.method == 'supcon' and CFG.feature_batch:
                train_dataset.set_label_key('ivt')
                avg_loss = train_feature_batch_supcon(train_loader, model, CFG, criterion, optimizer, epoch, scheduler, CFG.device, scaler)

            elif CFG.method == 'supcon':
                avg_loss = train_supcon(train_loader, model, CFG, criterion, optimizer, epoch, scheduler, CFG.device, scaler)

            elif CFG.method == 'curriculum_supcon' and CFG.feature_batch:
                label_index = min(epoch // 1, len(label_key_list) - 1)
                train_dataset.set_label_key(label_key_list[label_index])
                if CFG.label_sim:
                    sim_path = f"E:/Surgical/label_similarity_matrices_{label_key_list[label_index]}_fold_{fold}.pkl"
                    train_dataset.update_label_sim_matrix(sim_path)
                avg_loss = train_feature_batch_supcon(train_loader, model, CFG, criterion, optimizer, epoch, scheduler, CFG.device, scaler)

            elif CFG.method == "curriculum_supcon":
                avg_loss = train_Naive_Curriculum_Supcon(train_loader, model, CFG, criterion, optimizer, epoch, scheduler, CFG.device, scaler)

            log_training.write(f"Epoch: {epoch} Training Loss: {avg_loss:.4f}\n")
            scheduler.step()
            cur_lr = scheduler.get_last_lr()

            if CFG.neplog:
                run[f"tloss{fold}"].log(avg_loss)
                run[f"cLR_{fold}"].log(cur_lr)

            print(f"epoch: {epoch}, avg_loss: {avg_loss:.4f}")
            print(raw_line_ssl.format(epoch, avg_loss, (time.time() - epoch_start) / 60))

            if epoch == CFG.epochs - 1:
                save_path = os.path.join(CFG.output_dir, f"checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.exp}_{epoch}.pth")
                torch.save({"model": model.state_dict()}, save_path)
                print(f"Model saved at epoch: {epoch}\n")

        del model, train_loader, train_folds
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Training time: {(time.time() - start_time) / 60:.2f} minutes")
#================================================================================================
