import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import timm

# Dataset class using torchvision + PIL
class SupConFeatureExtractorDataset(Dataset):
    def __init__(self, df, CFG, transform=None):
        self.df = df.reset_index(drop=True)
        self.CFG = CFG
        self.transform = transform
        self.file_names = df["image_path"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        file_path = os.path.join(self.CFG.parent_path, self.CFG.train_path, self.file_names[index])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image not found: {file_path}")
        
        image = Image.open(file_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


@torch.no_grad()
def extract_features_for_fold(CFG, folds):
    from helper import get_dataframes
    from models import supcon_Model

    device = CFG.device
    feature_dir = r"E:\cholect45_features"
    os.makedirs(feature_dir, exist_ok=True)

    # torchvision-style validation transform
    transform = transforms.Compose([
        transforms.Resize((CFG.height, CFG.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    for fold in CFG.trn_fold:
        print(f"\n🔄 Processing Fold {fold}")
        train_folds, _, _ = get_dataframes(folds, fold)

        dataset = SupConFeatureExtractorDataset(train_folds, CFG, transform=transform)
        loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False, 
                            num_workers=CFG.nworkers, pin_memory=True)

        model = supcon_Model(CFG, CFG.model_name, pretrained=True)
        model.head = nn.Identity()
        model.to(device)
        model.eval()

        all_features = []

        for images in tqdm(loader, desc=f"Extracting features (Fold {fold})"):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)
        print(f"✅ Feature shape for fold {fold}: {all_features.shape}")

        sim_matrix = cosine_similarity(all_features)
        print(f"✅ Similarity matrix shape for fold {fold}: {sim_matrix.shape}")

        if not CFG.feature_file_name.endswith(".pkl"):
            CFG.feature_file_name += ".pkl"
        if not CFG.cos_sim_matrix_file_name.endswith(".pkl"):
            CFG.cos_sim_matrix_file_name += ".pkl"

        feature_path = os.path.join(feature_dir, f"fold{fold}_{CFG.feature_file_name}")
        sim_path = os.path.join(feature_dir, f"fold_{fold}_{CFG.cos_sim_matrix_file_name}")

        with open(feature_path, "wb") as f:
            pickle.dump(all_features, f)
        with open(sim_path, "wb") as f:
            pickle.dump(sim_matrix, f)

        print(f"📁 Saved features to: {feature_path}")
        print(f"📁 Saved similarity matrix to: {sim_path}")


import yaml
from types import SimpleNamespace

def load_cfg_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)

# ================================
# Entry Point
# ================================
if __name__ == "__main__":
    import argparse
    from preprocess import get_folds

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()

    CFG = load_cfg_from_yaml(args.cfg)

    seed = CFG.seed if hasattr(CFG, 'seed') else 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    folds = get_folds(CFG)
    extract_features_for_fold(CFG, folds)
