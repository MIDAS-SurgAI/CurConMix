# helper functions
import os
import random
import numpy as np
import torch
# import ivtmetrics
from ivtmetrics.recognition import Recognition
import pandas as pd 
from tqdm import tqdm 


def seed_torch(seed=42):
    """
    Seed various random number generators to ensure reproducibility.

    Args:
        seed (int): Seed value to set for random number generators.

    Returns:
        None
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


### Table for printing results

header = f"""
 Epoch | {"Loss":^6} | {"Val Loss":^7} | {"mAP":^7} | {"CmAP":^7} | {"Time, m":^6}
"""
header_v2 = f"""
 Epoch | {"Loss":^6} | {"Val Loss":^7} | {"Train_mAP":^7} | {"CmAP":^7} | {"Time, m":^6}
"""
header_split = f"""
 Epoch | {"Loss":^6} | {"Val Loss":^7} | {"Train_mAP":^7} | {"Val_mAP":^7} | {"test_mAP":^7} | {"Time, m":^6}
"""
header_ssl = f"""
 Epoch | {"Loss":^6}  | {"Time, m":^6}
"""
raw_line_ssl = "{:6d} | {:6.2f} | {:6.2f}"

# raw_line = "{:6d} | {:7.3f} | {:7.3f} | {:7.3f} | {:7.3f} | {:6.2f}"
raw_line = "{:6d} | {:7.3f} | {:7.3f} | {:7.3f} | {:6.2f}"
raw_line_split = "{:6d} | {:7.3f} | {:7.3f} | {:7.3f} | {:7.3f} | {:7.3f} | {:6.2f}"
temp_header = f"""
 Epoch | {"Loss":^6} | {"Val Loss":^7} | {"mAP":^7} | {"CmAP":^7} | {"future_mAP":^7} | {"future_CmAP":^7} | {"Time, m":^6}
"""
temp_raw_line = "{:6d} | {:7.3f} | {:7.3f} | {:7.3f} | {:7.3f} | {:7.3f} | {:7.3f} | {:6.2f}"

def cholect45_ivtmetrics_mAP(df, CFG):
    """
    Compute the official CholecT45 mAP score.

    Takes a dataframe with ground truth triplets and predictions.

    Metric calculation:
    - Aggregate per video over each fold
    - Mean of 5 folds

    Parameters:
    - df (pd.DataFrame): DataFrame with ground truth triplets and predictions.
    - CFG (object): Configuration object containing hyperparameters.

    Returns:
    float: Mean mAP value over 5 folds.

    """
    # Get the indexes of the 1st triplet/prediction columns
    tri0_idx = int(df.columns.get_loc("tri0"))
    pred0_idx = int(df.columns.get_loc("0"))

    # Initiate empty list to store the folds mAP
    ivt = []
    classwise_ap_list = [] 

    # Loop over the 5 folds
    for fold in range(CFG.n_fold):
        # Initialize the ivt metric
        rec = Recognition(num_class=100)

        # Filter the fold and its corresponding videos
        fold_df = df[df["fold"] == fold]
        vids = fold_df.video.unique()

        # Loop over the videos
        for i, v in enumerate(vids):
            # Filter the video
            vid_df = fold_df[fold_df["video"] == v]

            rec.update(
                vid_df.iloc[:, tri0_idx : tri0_idx + 100].values,
                vid_df.iloc[:, pred0_idx : pred0_idx + 100].values,
            )

            rec.video_end()
   
        # Get the final mAP score for the fold
        result = rec.compute_video_AP('ivt', ignore_null=CFG.ignore_null)
        ivt.append(result['mAP'])
        classwise_ap_list.append(result['AP'])

    # Return the mean mAP value over 5 folds
    return np.mean(ivt)

def cholect45_ivtmetrics_mAP_all(df, CFG):
    """
    Compute official CholecT45 mAP for all components.

    Args:
        df (pd.DataFrame): DataFrame with ground truth and predictions.
        CFG (object): Configuration with fold and evaluation settings.

    Returns:
        mean_mAPs (dict): Mean mAP per component.
        std_mAPs (dict): Std of mAP per component.
        classwise_AP_dfs (dict): DataFrames with classwise AP per component.
    """
    tri0_idx = int(df.columns.get_loc("tri0"))
    pred0_idx = int(df.columns.get_loc("0"))

    components = ['i', 'v', 't', 'iv', 'it', 'ivt']
    mAPs = {comp: [] for comp in components}
    classwise_APs = {comp: [] for comp in components}

    for fold in tqdm(range(CFG.n_fold)):
        rec = Recognition(num_class=100, ignore_null=CFG.ignore_null)
        fold_df = df[df["fold"] == fold]
        vids = fold_df.video.unique()

        for v in vids:
            vid_df = fold_df[fold_df["video"] == v]
            rec.update(
                vid_df.iloc[:, tri0_idx : tri0_idx + 100].values,
                vid_df.iloc[:, pred0_idx : pred0_idx + 100].values,
            )
            rec.video_end()

        for comp in components:
            result = rec.compute_video_AP(comp)
            mAPs[comp].append(result['mAP'])
            classwise_APs[comp].append(result['AP'])

    mean_mAPs = {comp: np.mean(mAPs[comp]) for comp in components}
    std_mAPs = {comp: np.std(mAPs[comp]) for comp in components}
    mean_classwise_APs = {comp: np.mean(classwise_APs[comp], axis=0) for comp in components}

    classwise_AP_dfs = {}
    for comp in components:
        classwise_AP_df = pd.DataFrame({
            "class": [f'class_{i}' for i in range(len(mean_classwise_APs[comp]))],
            "AP" : mean_classwise_APs[comp]
        })
        classwise_AP_dfs[comp] = classwise_AP_df

    return mean_mAPs, std_mAPs, classwise_AP_dfs

def per_epoch_ivtmetrics(fold_df, CFG):
    """
    Compute per-epoch ivtmetrics.

    Parameters:
    - fold_df (pd.DataFrame): DataFrame with ground truth triplets and predictions for a fold.
    - CFG (object): Configuration object containing hyperparameters.

    Returns:
    float: mAP score for the given fold.

    Example:
    ```python
    epoch_mAP = per_epoch_ivtmetrics(fold_df, CFG)
    ```

    """
    # Get the indexes of the 1st triplet/prediction columns
    tri0_idx = int(fold_df.columns.get_loc("tri0"))
    pred0_idx = int(fold_df.columns.get_loc("0"))

    # Initialize the ivt metric
    rec = Recognition(num_class=100)

    # Get unique videos
    vids = fold_df.video.unique()

    # Loop over the videos
    for i, v in enumerate(vids):
        # Filter the video
        vid_df = fold_df[fold_df["video"] == v]

        rec.update(
            vid_df.iloc[:, tri0_idx : tri0_idx + 100].values,
            vid_df.iloc[:, pred0_idx : pred0_idx + 100].values,
        )
        rec.video_end()

    # Get the final mAP score
    mAP = rec.compute_video_AP("ivt", ignore_null=CFG.ignore_null)["mAP"]
    return mAP

def print_training_info(folds, CFG):
    
    # print GPU model
    print("\033[94mHardware used\033[0m")
    print(f"GPU: {torch.cuda.get_device_name(0)}, cpu cores: {os.cpu_count()}\n")
    
    # Experiment tag
    tag = (
        f"\033[92m{CFG.exp}\033[0m"
        if CFG.exp != "myexp"
        else f"\033[91mPlease tag your experiment; i.e: exp=teacher_multitask\033[0m\n"
    )
    # Create a formatted training info string
    training_info = (
        f"{'Model:':<20} {CFG.model_name}\n"
        f"{'Multitask:':<20} {False if CFG.target_size==100 else True}\n"
        f"{'Target size:':<20} {CFG.target_size}\n"
        f"{'Self-distillation:':<20} {CFG.distill}\n"
        f"{'N° images used is:':<20} {len(folds)}\n"
        f"{'Experiment:':<20} {tag}\n"
    )

    # Print the formatted training info
    print("\033[94mTraining parameters\033[0m")
    print(training_info)

    # Print the formatted training info

    hyperparameters_info = (
        f"{'Starting LR:':<20} {CFG.lr}\n"
        f"{'Minimum LR:':<20} {CFG.min_lr}\n"
        f"{'Epochs:':<20} {CFG.epochs}\n"
        f"{'Batch size:':<20} {CFG.batch_size}\n"
    )

    print("\033[94mHyperparameters\033[0m")
    print(hyperparameters_info)

    metrics_info = (
        f"{'mAP:':<20} Overall mAP per fold (no aggregation)\n"
        f"{'cmAP:':<20} Challenge official mAP (aggregation per video)\n"
    )

    print("\033[94mMetrics\033[0m")
    print(metrics_info)


    print("\033[94mTraining started\033[0m\n")



