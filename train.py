# Empty dataframe to store oofs and metrics
import gc
import os
import time
import torch
import pandas as pd
import neptune.new as neptune
from sklearn.metrics import average_precision_score
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.cuda import amp
from sklearn.utils.class_weight import compute_class_weight

from augmentation import *
from models import *
from preprocess import get_folds
from utils import *
from helper import *
from dataset import *
from losses import *
from losses_weighted import *
import pickle 

def log_neptune(run, key, value, fold=None):
    if fold is not None:
        key = f"{key}{fold}"
    run[key].log(value)

def train_cross_val(CFG):
    start_time = time.time()
    seed_torch(seed=CFG.seed)
    summary_dir = os.path.join(CFG.output_dir, 'summary_dir')
    log_folder_dir = os.path.join(CFG.output_dir, CFG.exp)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(log_folder_dir, exist_ok=True)

    if CFG.debug:
        CFG.epochs = 1
        CFG.neplog = False

    if CFG.neplog:
        run = neptune.init(
            project=CFG.neptune_project,
            api_token=CFG.neptune_api_token,
        )
        log_params = {
            "Model": CFG.model_name, "imsize": CFG.height, "LR": CFG.lr, 
            "bs": CFG.batch_size, "Epochs": CFG.epochs, "SD": CFG.SD, 
            "T_0": CFG.T_0, "min_lr": CFG.min_lr, "seed": str(CFG.seed), 
            "tsize": CFG.target_size, 
            "smooth": CFG.smooth, "exp": CFG.exp
        }
        for key, value in log_params.items():
            run[key].log(value)

    folds = get_folds(CFG)
    print_training_info(folds, CFG)

    for fold in range(CFG.start_n_fold, CFG.n_fold):
        if fold not in CFG.trn_fold:
            continue
        print(f"\033[92m{'-' * 8} Fold {fold + 1} / {CFG.n_fold}\033[0m")
        summary_dir_fold = os.path.join(log_folder_dir, f'{CFG.exp}_fold_{fold}.csv')

        with open(summary_dir_fold, 'w') as log_training:
            model = TripletModel(CFG, CFG.model_name, pretrained=CFG.pretrained).to(CFG.device)
            if CFG.pretrained_ssl:
                weights_path = os.path.join(
                    CFG.output_dir,
                    f"checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.pretrained_exp}.pth"
                )
                print(f"Using Pre-trained Checkpoint: {weights_path}")
                ssl_checkpoint = torch.load(weights_path, map_location=CFG.device)
                ssl_state_dict = ssl_checkpoint['model']
                # Load weights with non-strict mode
                load_result = model.load_state_dict(ssl_state_dict, strict=False)
                print("Weight Loading Summary:")
                if load_result.missing_keys:
                    print("❌ Missing Keys (not loaded into model):")
                    for k in load_result.missing_keys:
                        print(f"   - {k}")
                else:
                    print("No missing keys.")

                if load_result.unexpected_keys:
                    print("⚠️ Unexpected Keys (in checkpoint but not in model):")
                    for k in load_result.unexpected_keys:
                        print(f"   - {k}")
                else:
                    print("No unexpected keys.")


            train_folds, valid_folds, valid_folds_temp = get_dataframes(folds, fold)
            if CFG.distill:
                train_folds = apply_self_distillation(fold, train_folds, CFG)
            else:
                print('Using default Train dataset')

            train_dataset = TrainDataset(train_folds, CFG, transform=get_transforms(data="train", CFG=CFG), fold=fold)
            valid_dataset = TrainDataset(valid_folds, CFG, transform=get_transforms(data="valid", CFG=CFG), fold=fold)

            train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=CFG.shuffle,
                                      num_workers=CFG.nworkers, pin_memory=True, drop_last=True)
            valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False,
                                      num_workers=CFG.nworkers, pin_memory=False, drop_last=False)

            # Define optimizer, scheduler, and loss function
            optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.epochs+1, T_mult=1000, eta_min=CFG.min_lr)
            if CFG.loss == 'bce':
                criterion = nn.BCEWithLogitsLoss(reduction="sum").to(CFG.device) 

            best_score = 0.0
            scaler = amp.GradScaler()
            print('start training')
            if CFG.mixup: 
                print('Using Mixup')
            
            for epoch in range(CFG.epochs):
                epoch_start = time.time()
                if CFG.mixup:
                    avg_loss = train_mixup(train_loader, model, CFG, criterion, optimizer, epoch, scheduler, CFG.device, scaler=scaler)
                else:
                    print('Default training without Mixup')
                    avg_loss = train_fn(train_loader, model, CFG, criterion, optimizer, epoch, scheduler, CFG.device, scaler=scaler)

                avg_val_loss, preds = valid_fn(valid_loader, model, CFG, criterion, CFG.device)
                scheduler.step()

                valid_folds_temp[[str(c) for c in range(CFG.target_size)]] = preds
                cholect45_epoch_CV = per_epoch_ivtmetrics(valid_folds_temp, CFG)

                log_training.write(f"Epoch: {epoch} Validation Loss: {avg_val_loss} CholecT45 mAP: {cholect45_epoch_CV} \n")

                if CFG.neplog:
                    log_neptune(run, "tloss", avg_loss, fold)
                    log_neptune(run, "val_loss", avg_val_loss, fold)
                    log_neptune(run, "cmAP", cholect45_epoch_CV, fold)

                if cholect45_epoch_CV > best_score:
                    best_score = cholect45_epoch_CV
                    save_checkpoint_path = os.path.join(CFG.output_dir, f"checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.pth")
                    torch.save({"model": model.state_dict(), "preds": preds}, save_checkpoint_path)
                print(raw_line.format(epoch, avg_loss, avg_val_loss, cholect45_epoch_CV, (time.time() - epoch_start) / 60 ** 1))
            del model, train_loader, valid_loader
            torch.cuda.empty_cache()
            
    cholect45_final_CV = cholect45_ivtmetrics_mAP(valid_folds_temp, CFG)
    print(f"CV: Overall mAP: {cholect45_final_CV:.4f}")

    if CFG.neplog:
        run["CV"].log(cholect45_final_CV)

    print(f"Training time: {(time.time() - start_time) / 60:.2f} minutes")
#================================================================================================

#================================================================================================

def select_transforms(data, CFG):
    print('Using Default Augmentation Methods')
    return get_transforms(data='train', CFG=CFG)
    
def train_cross_val_SSL(CFG):
    start_time = time.time()
    seed_torch(seed=CFG.seed)
    summary_dir = os.path.join(CFG.output_dir, 'summary_dir')
    os.makedirs(summary_dir, exist_ok=True)
    log_folder_dir = os.path.join(CFG.output_dir, CFG.exp )
    os.makedirs(log_folder_dir, exist_ok=True)
    summary_dir_total = os.path.join(log_folder_dir, f'{CFG.exp}_total.csv')
    log_total = open(summary_dir_total, 'w')
    # DEBUG: Faster iteration
    if CFG.debug:
        CFG.epochs = 1
        CFG.neplog = False

    if CFG.neplog:
        # Initiate logging
        run = neptune.init(
            project=CFG.neptune_project,
            api_token=CFG.neptune_api_token,
        )  # your credential

        run["Model"].log(CFG.model_name)
        run["imsize"].log(CFG.height)
        run["LR"].log(CFG.lr)
        run["bs"].log(CFG.batch_size)
        run["Epochs"].log(CFG.epochs)
        run["SD"].log(CFG.SD)
        run["T_0"].log(CFG.T_0)
        run["min_lr"].log(CFG.min_lr)
        run["seed"].log(str(CFG.seed))
        run["tsize"].log(CFG.target_size)
        run["smooth"].log(CFG.smooth)
        run["exp"].log(CFG.exp)

    # Start an empty dataframe to store the predictions
    oof_df = pd.DataFrame()

    # Create folders to save the checkpoints and predictions
    os.makedirs(os.path.join(CFG.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(CFG.output_dir, "oofs"), exist_ok=True)

    # Get the preprocessed train dataframe
    folds = get_folds(CFG)
    transform = select_transforms(data="train", CFG=CFG)

    print_training_info(folds, CFG)
    label_key_list = CFG.label_order
    # Loop over the folds
    for fold in range(CFG.start_n_fold, CFG.n_fold):
        # Skip some folds
        if fold in CFG.trn_fold:
            if CFG.feature_batch and (CFG.method == 'supcon' or CFG.method =='curriculum_supcon'):
                print("Using separated head Supcon model")
                model = FeatureSupConModel(CFG, CFG.model_name, pretrained=CFG.pretrained).to(CFG.device)
            elif CFG.method=='supcon' or CFG.method == 'curriculum_supcon':
                print('Using Default Supcon Model')
                model = supcon_Model(CFG, CFG.model_name, pretrained=CFG.pretrained).to(CFG.device)
            
            print("\033[92m" + f"{'-' * 8} Fold {fold + 1} / {CFG.n_fold}" + "\033[0m")
            summary_dir_fold = os.path.join(log_folder_dir, f'{CFG.exp}_fold_{fold}.csv')
            log_training = open(summary_dir_fold, 'w')
            train_folds, _, _ = get_dataframes(folds, fold)
            
            if (CFG.method=='supcon' or CFG.method=='curriculum_supcon') and CFG.feature_batch and CFG.feature_mixup: ## CurConMix Full Framework for feature-level contrastive learning
                print('Feature Batch and Feature Mixup during Pre-training')
                feature_file_name = f'C:/Users/kyuhw/Desktop/work/sd_temporal/baseline_train_mixup/fold{fold}_' + CFG.feature_file_name
                matrix_file_name = f'C:/Users/kyuhw/Desktop/work/sd_temporal/baseline_train_mixup/fold_{fold}_' + CFG.cos_sim_matrix_file_name
                # feature_file_name = f'E:/cholect45_features/fold{fold}_' + CFG.feature_file_name
                # matrix_file_name = f'E:/cholect45_features/fold_{fold}_' + CFG.cos_sim_matrix_file_name
                
                if CFG.Base384:
                    feature_file_name = f'E:/Surgical/384SwinT_fold{fold}_' + CFG.feature_file_name
                    matrix_file_name = f'E:/Surgical/384SwinT_fold_{fold}_' + CFG.cos_sim_matrix_file_name
                elif CFG.tiny_model:
                    feature_file_name = f'E:/Surgical/Swin_Tiny224_fold{fold}_' + CFG.feature_file_name
                    matrix_file_name = f'E:/Surgical/Swin_Tiny224_fold_{fold}_' + CFG.cos_sim_matrix_file_name
                elif CFG.base_model: 
                    feature_file_name = r'C:\Users\kyuhw\Desktop\work\sd_temporal\baseline_train_mixup\SwinB_fold0_trainset_features_threshold_10.pkl'
                    matrix_file_name = r'C:\Users\kyuhw\Desktop\work\sd_temporal\baseline_train_mixup\SwinB_fold_0_similarity_matrix.pkl'

                with open(feature_file_name, "rb") as f:
                    feature_list = pickle.load(f)
                with open(matrix_file_name, "rb") as f:
                    cos_sim_matrix = pickle.load(f)
                    
                print(f'Successfully Load {feature_file_name}, Successfully Load Matrix File {matrix_file_name}')
                print(f"Loaded features: {len(feature_list)} samples, Loaded Similarity Matrix: {cos_sim_matrix.shape}")
                label_list = {
                    'i': train_folds['instrument'].tolist(),
                    't': train_folds['target'].tolist(),
                    'v': train_folds['verb'].tolist(),
                    'it': train_folds['inst_target'].tolist(),
                    'iv': train_folds['inst_verb'].tolist(),
                    'tv': train_folds['target_verb'].tolist(),
                    'ivt': train_folds['triplet'].tolist()}
                train_dataset = SupConFeatureMixupBatchDataset(train_folds, CFG, features=feature_list,labels=label_list, cos_sim_matrix=cos_sim_matrix, transform=transform)

            else:
                print("Using default Supcon Dataset")
                train_dataset = Supcon_TrainDataset(train_folds, CFG, transform=transform)
            
            print('Using Defalut Dataloader')
            train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.nworkers, pin_memory=True, drop_last=True)
            optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 =CFG.epochs+1, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
            
            if CFG.ssl_loss == 'supcon':
                if CFG.feature_batch:
                    print('Using Feature batch Supcon')
                    criterion = FeatureBatchdSupConLoss(CFG, temperature=CFG.temp)
                else:
                    print('Using Default Supcon Loss')
                    criterion = SupConLoss(CFG, temperature=CFG.temp)
            else:
                raise ValueError(f"Unsupported ssl_loss: {CFG.ssl_loss}")

            # Mixed precision scaler
            scaler = amp.GradScaler()
            # Start training: Loop over epochs
            print(header_ssl)
            for epoch in range(CFG.epochs):
                epoch_start = time.time()

                if CFG.method == 'supcon':
                    print('using default supcon training')
                    avg_loss = train_supcon(
                    train_loader,
                    model,
                    CFG,
                    criterion,
                    optimizer,
                    epoch,
                    scheduler,
                    CFG.device,
                    scaler=scaler,
                )
                elif CFG.method == 'curriculum_supcon' and CFG.feature_batch:
                    print('curriculum supcon with sampling')
                    interval = 2 if CFG.epochs == 6 else 1 # Set curriculum interval based on total training epochs
                    label_index = min(epoch // interval, len(label_key_list) - 1) # Determine label index based on current epoch and cap it by label_key_list length
                    train_dataset.set_label_key(label_key_list[label_index])
                    
                    avg_loss = train_feature_batch_supcon(
                    train_loader,
                    model,
                    CFG,
                    criterion,
                    optimizer,
                    epoch,
                    scheduler,
                    CFG.device,
                    scaler=scaler,
                )

                log_output = f"Epoch: {epoch} Training Loss: {avg_loss} \n"
                log_training.write(log_output)
                scheduler.step()
                cur_lr = scheduler.get_last_lr()

                log_val_output = f"Epoch: {epoch}  Loss: {avg_loss}  \n"
                log_training.write(log_val_output)
               
                if CFG.neplog:
                    run[f"tloss{fold}"].log(avg_loss)
                    run[f"cLR_{fold}"].log(cur_lr)

                # Print loss/metric
                print(f'epoch: {epoch}, avg_loss: {avg_loss}')
                print(raw_line_ssl.format(epoch, avg_loss, (time.time() - epoch_start) / 60 ** 1))                       
                if epoch in [CFG.epochs - 1]: 
                    save_checkpoint_path = os.path.join(CFG.output_dir, f"checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.exp}_{epoch}.pth")
                    torch.save({"model": model.state_dict()}, save_checkpoint_path)
                    print(f"Model saved at epoch: {epoch}\n")
            # Free the GPU memory
            del model, train_loader, train_folds
            torch.cuda.empty_cache()
            gc.collect()
    print(f"Training time: {(time.time() - start_time) / 60}")

#================================================================================================
