import torch
import os
from torch.utils.data import DataLoader
import hydra
import warnings
import pandas as pd
import pickle
from models import TripletModel
from preprocess import get_folds
from augmentation import get_transforms
from dataset import TrainDataset
from helper import *
from global_var import config_name
os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")
from helper import *


def inference(CFG):
    """
    Run inference and generate soft-labels or final predictions.

    Args:
        CFG (OmegaConf): Configuration object.

    Returns:
        None

    Side Effects:
        - Saves predictions to CSV file if CFG.inference is True, in the 'predictions' directory.
        - Saves soft-labels to CSV file if CFG.inference is False, in the 'softlabels' directory.
    """

    # Get folds and video ids
    folds = get_folds(CFG)

    # Create soft labels folder
    softlabels_dir = os.path.join(CFG.output_dir, "softlabels")
    if not os.path.exists(softlabels_dir):
        os.mkdir(softlabels_dir)
        print("./softlabels directory created!")

    # Create predictions folder
    # predictions_dir = os.path.join(CFG.output_dir, "predictions")
    # if not os.path.exists(predictions_dir):
    #     os.mkdir(predictions_dir)
    predictions_dir = os.path.join(CFG.output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    print("./predictions directory created!")

    # Initialize an empty dataframe for predictions
    pred_df = pd.DataFrame()

    if CFG.inference:
        print("\033[94mStarting inference\033[0m")
    else:
        print("\033[94mGenerating soft-labels\033[0m")


    if "challenge" in CFG.split_selector:
        # Load model
        model = TripletModel(CFG, model_name=CFG.model_name, pretrained=False).to(
                CFG.device)
        weights_path = os.path.join(CFG.output_dir,f"checkpoints/{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.pth")
        model.load_state_dict(torch.load(weights_path)["model"])

        trn_idx = folds[folds["fold"] == 'train'].index
        test_idx = folds[folds["fold"] == 'test'].index
        inference_folds = (folds.loc[test_idx].reset_index(drop=True) if CFG.inference else folds.loc[trn_idx].reset_index(drop=True))
        # Pytorch dataset
        inference_dataset = TrainDataset(
            inference_folds,
            transform=get_transforms(CFG=CFG, data="valid"),
            inference=True,
            CFG=CFG,
        )

        # Pytorch dataloader
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.nworkers,
            pin_memory=False,
            drop_last=False,
        )
        # Inference loop
        preds, feature = inference_fn(inference_loader, model, CFG.device,CFG)
        # Load and save preds
        inference_folds_temp = inference_folds.copy()
        inference_folds[[str(c) for c in range(CFG.target_size)]] = preds
        # save path
        if CFG.inference:
            # Concatenate the folds predictions
            pred_df = pd.concat([pred_df, inference_folds])
            preds_save_path = os.path.join(CFG.output_dir, f"predictions/{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.csv")
            pred_df.to_csv(preds_save_path)
            print(f"Predictions saved at {preds_save_path}")
        else:
            # Save soft-labels
            os.makedirs(os.path.join(CFG.output_dir, 'softlabels'),exist_ok=True )
            save_path = os.path.join(
                CFG.output_dir,
                f"softlabels/sl_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.csv",
            )
            inference_folds.to_csv(save_path)
            print(f"Soft labels saved at {save_path}")
    else:

        # Process each fold
        for fold in range(CFG.start_n_fold, CFG.n_fold):
            # Load model
            model = TripletModel(CFG, model_name=CFG.model_name, pretrained=True).to(CFG.device)
            # Load the weights
            weights_path = os.path.join(CFG.output_dir, f"checkpoints/fold{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.pth")
            print('weights_path: ', weights_path)
            model.load_state_dict(torch.load(weights_path)["model"],strict=True)
            if CFG.inference:
                print(f"fold {fold}: Weights loaded successfully")
            else:
                print(f"fold {fold}: Weights loaded successfully")

            # Get train and valid indexes
            trn_idx = folds[folds["fold"] != fold].index
            vld_idx = folds[folds["fold"] == fold].index

            # Get train dataframe for training or validation set based on inference mode
            inference_folds = (folds.loc[vld_idx].reset_index(drop=True)if CFG.inference else folds.loc[trn_idx].reset_index(drop=True))
            # Pytorch dataset
            inference_dataset = TrainDataset(inference_folds, transform=get_transforms(CFG=CFG, data="valid"), inference=True, CFG=CFG)

            # Pytorch dataloader
            inference_loader = DataLoader(
                inference_dataset,
                batch_size=CFG.valid_batch_size,
                shuffle=False,
                num_workers=CFG.nworkers,
                pin_memory=False,
                drop_last=False,
            )
            preds = inference_fn(inference_loader, model, CFG.device, CFG)
            inference_folds[[str(c) for c in range(CFG.target_size)]] = preds  # Store predictions in the DataFrame

            # save path
            if CFG.inference:
                # Concatenate the folds predictions
                pred_df = pd.concat([pred_df, inference_folds])
            else:
                os.makedirs(os.path.join(CFG.output_dir, 'softlabels'),exist_ok=True )
                save_path = os.path.join(
                    CFG.output_dir,
                    f"softlabels/sl_f{fold}_{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.csv",
                )
                inference_folds.to_csv(save_path)

        print("\033[94mSaving...\033[0m")
        if CFG.inference:
            preds_save_path = os.path.join(
                CFG.output_dir,
                f"predictions/{CFG.model_name[:8]}_{CFG.target_size}_{CFG.exp}.csv",
            )
            pred_df.to_csv(preds_save_path)
            print(f"Predictions saved at {preds_save_path}")
        else:
            print(f"Soft labels saved at {save_path}")

# Run the code
@hydra.main(config_name=config_name)
def generate(CFG):
    """
    Main function to run the inference.

    Args:
        CFG (OmegaConf): Configuration object.

    Returns:
        None
    """
    inference(CFG)


if __name__ == "__main__":
    generate()
