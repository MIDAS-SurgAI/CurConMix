import pandas as pd
import os
import hydra
import warnings
from utils import cholect45_ivtmetrics_mAP, cholect45_ivtmetrics_mAP_all
from global_var import config_name


warnings.filterwarnings('ignore')


def evaluate(CFG):
    """
    Evaluate predictions using the CholecT45 metric for experiments.

    This function reads prediction files from a specified folder, computes the CholecT45 metric for each experiment,
    and optionally computes the ensemble metric if specified in the configuration.

    Args:
        CFG (OmegaConf): Configuration object.

    Returns:
        None
    """
    # Set target size to 100 to evaluate on the triplets only
    CFG.target_size = 100

    # Determine the folder of saved predictions (inference or out-of-folds)
    folder = "predictions" if CFG.inference else "oofs"

    # Get the available experiments in the specified folder
    prediction_dfs = os.listdir(os.path.join(CFG.output_dir, folder))

    # Loop over the experiments
    for pred_df in prediction_dfs:
        # Load the dataframe
        df = pd.read_csv(os.path.join(CFG.output_dir, folder, pred_df))

        # Parse the experiment tag
        #experiment = pred_df.split(".")[0].split('_')[-1]
        experiment = pred_df.split(".")[0].split('_')[1:]
        experiment_name = ('_').join(experiment)
    
        # Get the mAP score
        # score = cholect45_ivtmetrics_mAP(df, CFG)
        score, classwise_ap_df = cholect45_ivtmetrics_mAP(df, CFG)
        classwise_ap_df.to_csv(f'{experiment_name}_classwise_AP.csv', index=False)
        print(f"{experiment_name}: {round(score * 100, 2)}%")

    # Compute the ensemble of multiple experiments available in CFG.ensemble_models
    if CFG.ensemble:
        try:
            preds = None
            num_models = len(CFG.ensemble_models)
            for model in CFG.ensemble_models:
                # Load the model's predictions
                df = pd.read_csv(os.path.join(CFG.output_dir, folder, model))

                # Get the indexes of the 1st prediction columns
                pred0_idx = df.columns.get_loc("0")

                # Accumulate the predictions
                preds = preds + df.iloc[:, pred0_idx:pred0_idx + 100].values if preds is not None else df.iloc[:, pred0_idx:pred0_idx + 100].values

            df.iloc[:, pred0_idx:pred0_idx + 100] = preds
            if CFG.ensemble_avg:
                preds /= num_models
            # Compute the ensemble mAP metric
            score = cholect45_ivtmetrics_mAP(df, CFG)
            
            # Get experiment tags for ensemble models
            ensemble_experiments = [model.split(".")[0].split('_')[-1] for model in CFG.ensemble_models]
            print(f"Ensemble of {ensemble_experiments}: {round(score * 100, 2)}")
        except Exception as e:
            print("Ensemble didn't work: Please check the spelling or the path of your prediction csv files.")
            print(e)

def evaluate_all(CFG):
    CFG.target_size = 100

    folder = "predictions" if CFG.inference else "oofs"
    if CFG.evaluate_all:
        folder = 'Crucial_predictions' if CFG.inference else "oofs"
    prediction_dfs = os.listdir(os.path.join(CFG.output_dir, folder))

    for pred_df in prediction_dfs:
        df = pd.read_csv(os.path.join(CFG.output_dir, folder, pred_df))
        experiment = pred_df.split(".")[0].split('_')[1:]
        experiment_name = ('_').join(experiment)
        mean_mAPs, std_mAPs, classwise_AP_dfs = cholect45_ivtmetrics_mAP_all(df, CFG)

        for comp in classwise_AP_dfs:
            classwise_AP_df = classwise_AP_dfs[comp]
            classwise_AP_df.to_csv(f'{experiment_name}_{comp}_classwise_AP.csv', index=False)

        print(f"{experiment_name}의 결과:")
        for comp in mean_mAPs:
            print(f"{comp} mAP: {mean_mAPs[comp]*100:.2f}% (±{std_mAPs[comp]*100:.2f}%)")


# Run the code
@hydra.main(config_name=config_name)
def run(CFG):
    """
    Main function to run the evaluation.

    Args:
        CFG (OmegaConf): Configuration object.

    Returns:
        None
    """
    if CFG.evaluate_all:
        evaluate_all(CFG)
    else:
        evaluate(CFG)


if __name__ == "__main__":
    run()
