from train import *
import hydra
import warnings

import os 
os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

config_name="config"
@hydra.main(config_name=config_name)
def train(CFG):
    if CFG.ssl:
        train_cross_val_SSL(CFG)
    else:
        train_cross_val(CFG)

if __name__ == "__main__":
    train()
