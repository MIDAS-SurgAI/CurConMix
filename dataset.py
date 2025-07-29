import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
from torch.utils.data import Sampler
import random 
from collections import defaultdict, Counter
from functools import lru_cache
import ast
import pickle

# Fix number of threads used by opencv
cv2.setNumThreads(1)
class TrainDataset(Dataset):
    def __init__(self, df, CFG, transform=None,fold=None, inference=False):
        self.df = df
        self.CFG = CFG
        self.file_names = df["image_path"].values
        self.transform = transform
        self.inference = inference
        index_no = int(df.columns.get_loc(CFG.col0))
        self.labels = torch.FloatTensor(
            self.df.iloc[:, index_no : index_no + CFG.target_size].values.astype(
                np.float16
            )
        )
        self.feature=None
        self.fold=fold
   
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
   
        # Localize the image and targets
        file_name = self.file_names[index]
        target = self.labels[index]
        if self.feature!=None:
            feature = self.feature[index]

        # Read the image
        file_path = os.path.join(self.CFG.parent_path, self.CFG.train_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
  
        if self.inference:
            return image
        else:
            return image, target

    
class Supcon_TrainDataset(Dataset):
    def __init__(self, df, CFG, transform=None, inference=False):
        self.df = df
        self.CFG = CFG
        self.file_names = df["image_path"].values
        self.transform = transform
        self.inference = inference
        self.target = df["target"].values
        self.inst = df["instrument"].values
        self.verbs = df["verb"].values
        self.inst_target = df["inst_target"].values
        self.triplet = df["triplet"].values
        self.inst_verb = df["inst_verb"].values
        self.target_verb = df["target_verb"].values
        print(f'label_order: {self.CFG.label_order}')
        
    def load_image(self, file_path):

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Localize the image and targets
        file_name = self.file_names[index]
        target = self.target[index]
        inst_target = self.inst_target[index]
        triplet_label = self.triplet[index]
        verb_label = self.verbs[index]
        inst_verbs = self.inst_verb[index]
        target_verbs = self.target_verb[index]
        inst = self.inst[index]

        gt_labels = {'i': inst, 't': target, 'v': verb_label, 'it': inst_target, 'iv': inst_verbs, 'tv': target_verbs, 'ivt': triplet_label}
        label = ()
        for x in self.CFG.label_order:
            label += (gt_labels[x],)

        # Read the image
        file_path = os.path.join(self.CFG.parent_path, self.CFG.train_path, file_name)
        image = self.load_image(file_path)
        image2 = self.load_image(file_path)

        return (image,image2), label
    

class SupConFeatureMixupBatchDataset(Dataset):
    def __init__(self, df, CFG, features, labels, cos_sim_matrix, transform=None, inference=False):
        self.df = df.reset_index(drop=True)
        self.CFG = CFG
        self.transform = transform
        self.inference = inference
        self.label_key = 'ivt'  # Default label key for SupCon

        self.file_names = df["image_path"].values
        self.targets = df["target"].values
        self.inst = df["instrument"].values
        self.verbs = df["verb"].values
        self.inst_target = df["inst_target"].values
        self.triplet = df["triplet"].values
        self.inst_verb = df["inst_verb"].values
        self.target_verb = df["target_verb"].values

        self.features = features
        self.labels = labels
        self.cos_sim_matrix = cos_sim_matrix

    def set_label_key(self, label_key):
        self.label_key = label_key
        print(f"SupCon with label key: {self.label_key}")

    def load_image(self, file_path):
        image = cv2.imread(file_path)
        if image is None:
            print(f"Warning: Image not found or cannot be loaded at path: {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        file_path = os.path.join(self.CFG.parent_path, self.CFG.train_path, file_name)
        image = self.load_image(file_path)

        target = self.targets[index]
        inst_target = self.inst_target[index]
        triplet_label = self.triplet[index]
        verb_label = self.verbs[index]
        inst_verbs = self.inst_verb[index]
        target_verbs = self.target_verb[index]
        inst = self.inst[index]

        gt_labels = {
            'i': inst,
            't': target,
            'v': verb_label,
            'it': inst_target,
            'iv': inst_verbs,
            'tv': target_verbs,
            'ivt': triplet_label
        }

        if self.label_key not in self.labels:
            raise ValueError(f"Label key {self.label_key} not found in labels: {list(self.labels.keys())}")
        label = gt_labels[self.label_key]

        sim = self.cos_sim_matrix[index]
        all_labels = np.array(self.labels[self.label_key])
        selected_label = all_labels[index]

        indices_same_class = np.where(all_labels == selected_label)[0]
        indices_same_class = indices_same_class[indices_same_class != index]

        indices_different_class = np.where(all_labels != selected_label)[0]

        # Select top-N most similar samples from different class
        N = self.CFG.top_n
        sim_different_class = sim[indices_different_class]
        sorted_indices_diff = np.argsort(-sim_different_class)
        top_N = min(N, len(sorted_indices_diff))
        top_indices_different_class = indices_different_class[sorted_indices_diff[:top_N]]

        # Sample negatives and apply feature-level mixup
        num_neg_samples_needed = self.CFG.num_negative_samples * 2
        if len(top_indices_different_class) >= num_neg_samples_needed:
            selected_indices_different_class = np.random.choice(top_indices_different_class, size=num_neg_samples_needed, replace=False)
        else:
            print('Insufficient negative samples; padding with additional samples.')
            selected_indices_different_class = top_indices_different_class
            num_to_pad = num_neg_samples_needed - len(selected_indices_different_class)
            remaining_indices = np.setdiff1d(indices_different_class, selected_indices_different_class)
            pad_indices = np.random.choice(remaining_indices, size=num_to_pad, replace=False)
            selected_indices_different_class = np.concatenate((selected_indices_different_class, pad_indices))

        negative_features = []
        negative_labels = []
        for i in range(0, len(selected_indices_different_class), 2):
            idx1 = selected_indices_different_class[i]
            idx2 = selected_indices_different_class[i + 1]

            feature1 = self.features[idx1]
            feature2 = self.features[idx2]

            lam = np.random.beta(self.CFG.alpha, self.CFG.alpha)
            mixed_feature = lam * feature1 + (1 - lam) * feature2

            negative_features.append(mixed_feature)
            negative_labels.append(all_labels[idx1])  # Label doesn't affect contrastive loss

        negative_features = np.stack(negative_features)
        negative_labels = np.array(negative_labels)

        # Select one positive sample (least similar from the same class)
        if len(indices_same_class) > 0:
            sim_same_class = sim[indices_same_class]
            sorted_indices_same = np.argsort(sim_same_class)
            bottom_N = min(N, len(sorted_indices_same))
            bottom_indices_same_class = indices_same_class[sorted_indices_same[:bottom_N]]
            selected_indices_same_class = np.random.choice(bottom_indices_same_class, size=1, replace=False)
        else:
            selected_indices_same_class = np.array([index])

        positive_feature = self.features[selected_indices_same_class[0]]
        positive_label = all_labels[selected_indices_same_class[0]]

        contrast_features = np.concatenate(([positive_feature], negative_features), axis=0)
        contrast_labels = np.concatenate(([positive_label], negative_labels), axis=0)

        indices_selected = np.arange(len(contrast_features))
        np.random.shuffle(indices_selected)
        contrast_features = contrast_features[indices_selected]
        contrast_labels = contrast_labels[indices_selected]

        return image, label, contrast_features, contrast_labels
