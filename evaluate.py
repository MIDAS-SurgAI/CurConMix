import pandas as pd
import os
import hydra
import warnings
from utils import cholect45_ivtmetrics_mAP, cholect45_ivtmetrics_mAP_challenge, cholect45_ivtmetrics_mAP_all
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
        if 'challenge' in experiment:
            # Get the mAP score
            print(len(df))
            score = cholect45_ivtmetrics_mAP_challenge(df, CFG)
        else:
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
    """
    실험에 대해 CholecT45 메트릭을 사용하여 예측을 평가합니다.

    매개변수:
        CFG (OmegaConf): 설정 객체.

    반환값:
        None
    """
    # 평가를 위해 target_size를 100으로 설정
    CFG.target_size = 100

    # 예측 결과가 저장된 폴더 결정 (inference 또는 oofs)
    folder = "predictions" if CFG.inference else "oofs"
    if CFG.evaluate_all:
        folder = 'Crucial_predictions' if CFG.inference else "oofs"
    # 지정된 폴더에서 사용 가능한 실험 결과 파일 가져오기
    prediction_dfs = os.listdir(os.path.join(CFG.output_dir, folder))

    # 각 실험에 대해 반복
    for pred_df in prediction_dfs:
        # 데이터프레임 로드
        df = pd.read_csv(os.path.join(CFG.output_dir, folder, pred_df))

        # 실험 이름 파싱
        experiment = pred_df.split(".")[0].split('_')[1:]
        experiment_name = ('_').join(experiment)


        # mAP 점수 계산
        mean_mAPs, std_mAPs, classwise_AP_dfs = cholect45_ivtmetrics_mAP_all(df, CFG)

        # 클래스별 AP를 CSV로 저장 (필요한 경우)
        for comp in classwise_AP_dfs:
            classwise_AP_df = classwise_AP_dfs[comp]
            classwise_AP_df.to_csv(f'{experiment_name}_{comp}_classwise_AP.csv', index=False)

        # 결과 출력
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
