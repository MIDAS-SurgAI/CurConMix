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
        """
        Retrieves an item from the dataset.

        Parameters:
        - index (int): Index of the item to retrieve.

        Returns:
        - torch.Tensor or tuple: Image and target label if not in inference mode,
          otherwise only the image.

        """
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
    
class Supcon_TrainDataset_mixup(Dataset):
   

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
        
        self.alpha_value = CFG.alpha
        # 트리플렛 값을 기준으로 같은 레이블을 가진 인덱스를 저장하는 딕셔너리 생성
        self.triplet_dict = {}
        for idx, triplet_value in enumerate(self.triplet):
            if triplet_value not in self.triplet_dict:
                self.triplet_dict[triplet_value] = []
            self.triplet_dict[triplet_value].append(idx)

    def load_image(self, file_path):

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image

    def mixup_data_sample(self, image1, image2, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1 
        
        mixed_x = lam * image1 + (1-lam) * image2
        return mixed_x
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        Parameters:
        - index (int): Index of the item to retrieve.

        Returns:
        - torch.Tensor or tuple: Image and target label if not in inference mode,
          otherwise only the image.

        """
        # Localize the image and targets

        file_name = self.file_names[index]
        target = self.target[index]
        inst_target = self.inst_target[index]
        triplet_label = self.triplet[index]
        verb_label = self.verbs[index]
        inst_verbs = self.inst_verb[index]
        target_verbs = self.target_verb[index]
        inst = self.inst[index]

        triplet_value = self.triplet[index]
        indices_with_same_triplet = self.triplet_dict[triplet_value]
        
        if len(indices_with_same_triplet) == 1:
            idx1 = idx2 = indices_with_same_triplet[0]  # same image
        else:
            # 동일한 triplet 레이블을 가진 인덱스 중에서 두 개를 랜덤으로 선택
            idx1, idx2 = np.random.choice(indices_with_same_triplet, 2, replace=False)
        
        mixup_file_name1 = self.file_names[idx1]
        mixup_file_name2 = self.file_names[idx2]

        mixup_file_path1 = os.path.join(self.CFG.parent_path, self.CFG.train_path, mixup_file_name1)
        mixup_file_path2 = os.path.join(self.CFG.parent_path, self.CFG.train_path, mixup_file_name2)

        mixup_image1 = self.load_image(mixup_file_path1)
        mixup_image2 = self.load_image(mixup_file_path2)

        # Mixup the two selected images
        mixed_image = self.mixup_data_sample(mixup_image1, mixup_image2, alpha=self.alpha_value)
        
        gt_labels = {'i': inst, 't': target, 'v': verb_label, 'it': inst_target, 'iv': inst_verbs, 'tv': target_verbs, 'ivt': triplet_label}
        label = ()
        for x in self.CFG.label_order:
            label += (gt_labels[x],)

        if self.CFG.auxillary !=False:
            aux = self.aux[index]
            label += (aux,)
        


        # Read the image
        file_path = os.path.join(self.CFG.parent_path, self.CFG.train_path, file_name)

        image = self.load_image(file_path)

        return (image,mixed_image), label
    

class SupConFeatureBatchDataset(Dataset):
    def __init__(self, df, CFG, features, labels, cos_sim_matrix, transform=None, inference=False):
        self.df = df.reset_index(drop=True)
        self.CFG = CFG
        self.transform = transform
        self.inference = inference
        self.label_key = 'ivt'
        
        # 데이터프레임에서 필요한 열들을 불러옵니다.
        self.file_names = df["image_path"].values
        self.targets = df["target"].values
        self.inst = df["instrument"].values
        self.verbs = df["verb"].values
        self.inst_target = df["inst_target"].values
        self.triplet = df["triplet"].values
        self.inst_verb = df["inst_verb"].values
        self.target_verb = df["target_verb"].values

        # 입력으로 받은 feature list와 label list를 저장합니다.
        self.features = features  # (전체 데이터셋 크기, 특징 차원)
        self.labels = labels      # (전체 데이터셋 크기,)

        # 입력으로 받은 cosine similarity matrix를 저장합니다.
        self.cos_sim_matrix = cos_sim_matrix  # Shape: (N, N)

    def set_label_key(self, label_key):
        self.label_key = label_key
        print(f'{self.label_key}로 supcon을 진행합니다.')

    def load_image(self, file_path):
        """
        이미지를 로드하고, 필요한 경우 변환을 적용합니다.
        """
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        데이터셋에서 하나의 샘플을 가져옵니다.
        """
        # 이미지 로드
        file_name = self.file_names[index]
        file_path = os.path.join(self.CFG.parent_path, self.CFG.train_path, file_name)
        image = self.load_image(file_path)

        # 레이블 처리
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

        # label = gt_labels['ivt']
        label = gt_labels[self.label_key] ## set_label_key = ['i', 'it', 'ivt']

        # 현재 샘플과 다른 샘플들에 대한 cosine similarity를 가져옵니다.
        sim = self.cos_sim_matrix[index]  # Shape: (N,)
        # all_labels = self.labels          # Shape: (N,)
        all_labels = self.labels[self.label_key]          # Shape: (N,)
        all_labels = np.array(all_labels)
        selected_label = all_labels[index]

        # 같은 클래스의 샘플 인덱스 (자기 자신 제외)
        indices_same_class = np.where(all_labels == selected_label)[0]
        indices_same_class = indices_same_class[indices_same_class != index]
        
        # 다른 클래스의 샘플 인덱스
        indices_different_class = np.where(all_labels != selected_label)[0]

        # 다른 클래스 중에서 상위 N개의 유사한 샘플 중에서 63개를 랜덤으로 선택
        N = self.CFG.top_n
        sim_different_class = sim[indices_different_class]
        sorted_indices_diff = np.argsort(-sim_different_class)  # 내림차순 정렬
        top_N = min(N, len(sorted_indices_diff))
        top_indices_different_class = indices_different_class[sorted_indices_diff[:top_N]]

        if len(top_indices_different_class) >= self.CFG.num_negative_samples:
            selected_indices_different_class = np.random.choice(top_indices_different_class, size=self.CFG.num_negative_samples, replace=False)
        else:
            print(f'negative sample 수가 {self.CFG.num_negative_samples}보다 작음.')
            # 63개보다 적으면 남은 부분을 랜덤 샘플로 채움
            selected_indices_different_class = top_indices_different_class
            num_to_pad = self.CFG.num_negative_samples - len(selected_indices_different_class)
            remaining_indices = np.setdiff1d(
                indices_different_class,
                selected_indices_different_class
            )
            pad_indices = np.random.choice(remaining_indices, size=num_to_pad, replace=False)
            selected_indices_different_class = np.concatenate((selected_indices_different_class, pad_indices))

        # 같은 클래스 중에서 하위 N개의 유사하지 않은 샘플 중에서 1개를 랜덤으로 선택
        if len(indices_same_class) > 0:
            sim_same_class = sim[indices_same_class]
            sorted_indices_same = np.argsort(sim_same_class)  # 오름차순 정렬 (유사하지 않은 순)
            bottom_N = min(N, len(sorted_indices_same))
            bottom_indices_same_class = indices_same_class[sorted_indices_same[:bottom_N]]
            selected_indices_same_class = np.random.choice(bottom_indices_same_class, size=1, replace=False)
        else:
            # 같은 클래스의 샘플이 없을 경우 랜덤으로 샘플 선택
            selected_indices_same_class = np.array([index])

        # 선택된 샘플들의 인덱스 합치기
        indices_selected = np.concatenate((selected_indices_different_class, selected_indices_same_class))
        # indices_selected의 순서를 무작위로 섞기
        shuffled_indices = np.random.permutation(len(indices_selected))
        indices_selected = indices_selected[shuffled_indices]  # 섞인 순서로 재배열

        # 선택된 샘플들의 특징과 레이블 가져오기 (섞인 순서로 반영)
        contrast_feature = self.features[indices_selected]
        contrast_label = all_labels[indices_selected]

        return image, label, contrast_feature, contrast_label

class SupConFeatureMixupBatchDataset(Dataset):
    def __init__(self, df, CFG, features, labels, cos_sim_matrix, transform=None, inference=False):
        self.df = df.reset_index(drop=True)
        self.CFG = CFG
        self.transform = transform
        self.inference = inference
        self.label_key = 'ivt'  # 기본 레이블 키 설정

        # 데이터프레임에서 필요한 열들을 불러옵니다.
        self.file_names = df["image_path"].values
        self.targets = df["target"].values
        self.inst = df["instrument"].values
        self.verbs = df["verb"].values
        self.inst_target = df["inst_target"].values
        self.triplet = df["triplet"].values
        self.inst_verb = df["inst_verb"].values
        self.target_verb = df["target_verb"].values

        # 입력으로 받은 feature list와 label list를 저장합니다.
        self.features = features  # (전체 데이터셋 크기, 특징 차원)
        self.labels = labels      # labels는 딕셔너리 형태

        # 입력으로 받은 cosine similarity matrix를 저장합니다.
        self.cos_sim_matrix = cos_sim_matrix  # Shape: (N, N)

    def set_label_key(self, label_key):
        self.label_key = label_key
        print(f'{self.label_key}로 supcon을 진행합니다.')

    def load_image(self, file_path):
        """
        이미지를 로드하고, 필요한 경우 변환을 적용합니다.
        """
        image = cv2.imread(file_path)
        # 이미지가 제대로 로드되지 않으면 경고 메시지 출력
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
        """
        데이터셋에서 하나의 샘플을 가져옵니다.
        """
        # 이미지 로드
        file_name = self.file_names[index]
        file_path = os.path.join(self.CFG.parent_path, self.CFG.train_path, file_name)
        image = self.load_image(file_path)

        # 레이블 처리
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
        label = gt_labels[self.label_key]  # set_label_key = ['i', 'it', 'ivt']

        # 현재 샘플과 다른 샘플들에 대한 cosine similarity를 가져옵니다.
        sim = self.cos_sim_matrix[index]  # Shape: (N,)
        all_labels = self.labels[self.label_key]  # Shape: (N,)
        all_labels = np.array(all_labels)
        selected_label = all_labels[index]

        # 같은 클래스의 샘플 인덱스 (자기 자신 제외)
        indices_same_class = np.where(all_labels == selected_label)[0]
        indices_same_class = indices_same_class[indices_same_class != index]

        # 다른 클래스의 샘플 인덱스
        indices_different_class = np.where(all_labels != selected_label)[0]

        # 다른 클래스 중에서 상위 N개의 유사한 샘플 중에서 126개를 랜덤으로 선택 (두 개씩 짝지어 63개 생성)
        N = self.CFG.top_n
        sim_different_class = sim[indices_different_class]
        sorted_indices_diff = np.argsort(-sim_different_class)  # 내림차순 정렬
        top_N = min(N, len(sorted_indices_diff))
        top_indices_different_class = indices_different_class[sorted_indices_diff[:top_N]]

        num_neg_samples_needed = self.CFG.num_negative_samples * 2  # 총 필요한 negative 샘플 수 (두 개씩 짝지을 것이므로 63 * 2)
        if len(top_indices_different_class) >= num_neg_samples_needed:
            selected_indices_different_class = np.random.choice(top_indices_different_class, size=num_neg_samples_needed, replace=False)
        else:
            print('Negative sample 수 부족')
            # 필요한 수보다 적으면 남은 부분을 랜덤 샘플로 채움
            selected_indices_different_class = top_indices_different_class
            num_to_pad = num_neg_samples_needed - len(selected_indices_different_class)
            remaining_indices = np.setdiff1d(
                indices_different_class,
                selected_indices_different_class
            )
            pad_indices = np.random.choice(remaining_indices, size=num_to_pad, replace=False)
            selected_indices_different_class = np.concatenate((selected_indices_different_class, pad_indices))

        # Negative 샘플들을 두 개씩 짝지어 mixup 적용
        negative_features = []
        negative_labels = []

        for i in range(0, len(selected_indices_different_class), 2):
            idx1 = selected_indices_different_class[i]
            idx2 = selected_indices_different_class[i + 1]

            feature1 = self.features[idx1]
            feature2 = self.features[idx2]

            # 각 negative sample마다 개별적으로 lambda 값을 추출
            lam = np.random.beta(self.CFG.alpha, self.CFG.alpha)

            # Feature mixup
            mixed_feature = lam * feature1 + (1 - lam) * feature2

            negative_features.append(mixed_feature)
            negative_labels.append(all_labels[idx1])  # 또는 두 레이블 중 하나를 선택 (negative에서는 레이블이 중요하지 않음)

        negative_features = np.stack(negative_features)
        negative_labels = np.array(negative_labels)

        # Positive sample 선택 (기존 방식 유지)
        if len(indices_same_class) > 0:
            sim_same_class = sim[indices_same_class]
            sorted_indices_same = np.argsort(sim_same_class)  
            bottom_N = min(N, len(sorted_indices_same))
            bottom_indices_same_class = indices_same_class[sorted_indices_same[:bottom_N]]
            selected_indices_same_class = np.random.choice(bottom_indices_same_class, size=1, replace=False)
        else:
            selected_indices_same_class = np.array([index])

        # Positive 샘플의 특징과 레이블 가져오기
        positive_feature = self.features[selected_indices_same_class[0]]
        positive_label = all_labels[selected_indices_same_class[0]]

        # Anchor, Positive, Negative 샘플들을 합치기
        contrast_features = np.concatenate(([positive_feature], negative_features), axis=0)
        contrast_labels = np.concatenate(([positive_label], negative_labels), axis=0)

        indices_selected = np.arange(len(contrast_features))
        np.random.shuffle(indices_selected)
        contrast_features = contrast_features[indices_selected]
        contrast_labels = contrast_labels[indices_selected]

        return image, label, contrast_features, contrast_labels

class WeightedSupConFeatureMixupBatchDataset(Dataset):
    def __init__(self, df, CFG, features, labels, cos_sim_matrix, label_sim_matrix, transform=None, inference=False):
        self.df = df.reset_index(drop=True)
        self.CFG = CFG
        self.transform = transform
        self.inference = inference
        self.label_key = 'ivt'
        self.sampling_alpha = 0.5

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
        self.label_sim_matrix = label_sim_matrix
        self.final_sim_matrix = self._compute_final_sim_matrix()

    def _compute_final_sim_matrix(self):
        # 매 epoch마다 호출하여 최신의 final_sim_matrix를 계산합니다.
        return self.sampling_alpha * self.cos_sim_matrix + (1 - self.sampling_alpha) * self.label_sim_matrix

    def update_label_sim_matrix(self, sim_matrix_path):
        with open(sim_matrix_path, "rb") as f:
            self.label_sim_matrix = pickle.load(f)
        # 업데이트 후 final_sim_matrix 재계산
        self.final_sim_matrix = self._compute_final_sim_matrix()
        print(f'Successfully load {sim_matrix_path}')

    def set_label_key(self, label_key):
        self.label_key = label_key
        print(f'{self.label_key}로 SupCon을 진행합니다.')

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
        label = gt_labels[self.label_key]

        # 최신 final_sim_matrix에서 해당 index의 행을 가져옵니다.
        final_sim = self.final_sim_matrix[index].copy()

        all_labels = np.array(self.labels[self.label_key])
        selected_label = all_labels[index]

        indices_same_class = np.where(all_labels == selected_label)[0]
        indices_same_class = indices_same_class[indices_same_class != index]
        indices_different_class = np.where(all_labels != selected_label)[0]

        final_sim[indices_same_class] = 0

        if self.CFG.use_softmax_sampling:
            final_sim_different_class = final_sim[indices_different_class]
            sampling_prob = np.exp(final_sim_different_class) / np.sum(np.exp(final_sim_different_class))
            num_neg_samples_needed = self.CFG.num_negative_samples * 2  
            selected_indices_different_class = np.random.choice(
                indices_different_class, 
                size=num_neg_samples_needed, 
                replace=False, 
                p=sampling_prob
            )
        else:
            sorted_indices_diff = np.argsort(-final_sim[indices_different_class])
            top_N = min(self.CFG.top_n, len(sorted_indices_diff))
            selected_indices_different_class = indices_different_class[sorted_indices_diff[:top_N]]

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
            negative_labels.append(all_labels[idx1])
        negative_features = np.stack(negative_features)
        negative_labels = np.array(negative_labels)

        if len(indices_same_class) > 0:
            sim_same_class = self.cos_sim_matrix[index][indices_same_class]
            sorted_indices_same = np.argsort(sim_same_class)
            bottom_N = min(self.CFG.top_n, len(sorted_indices_same))
            selected_indices_same_class = np.random.choice(
                indices_same_class[sorted_indices_same[:bottom_N]], size=1, replace=False
            )
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
    
# class WeightedSupConFeatureMixupBatchDataset(Dataset):
#     def __init__(self, df, CFG, features, labels, cos_sim_matrix, label_sim_matrix=None, transform=None, inference=False):
#         self.df = df.reset_index(drop=True)
#         self.CFG = CFG
#         self.transform = transform
#         self.inference = inference
#         self.label_key = 'ivt'  # 기본 레이블 키 설정
#         self.sampling_alpha = 0.5  # Cosine Similarity vs Label Similarity 가중치 설정

#         # 데이터프레임에서 필요한 열들을 불러옵니다.
#         self.file_names = df["image_path"].values
#         self.targets = df["target"].values
#         self.inst = df["instrument"].values
#         self.verbs = df["verb"].values
#         self.inst_target = df["inst_target"].values
#         self.triplet = df["triplet"].values
#         self.inst_verb = df["inst_verb"].values
#         self.target_verb = df["target_verb"].values

#         # 입력 데이터 저장
#         self.features = features  # (전체 데이터셋 크기, 특징 차원)
#         self.labels = labels      # labels는 딕셔너리 형태

#         # Cosine Similarity Matrix (이미지 임베딩 기반)
#         self.cos_sim_matrix = cos_sim_matrix  # Shape: (N, N)

#         # Label Similarity Matrix (라벨 기반 유사도)
#         self.label_sim_matrix = label_sim_matrix
        
#     def set_label_key(self, label_key):
#         self.label_key = label_key
#         print(f'{self.label_key}로 SupCon을 진행합니다.')

#     def set_label_sim_matrix(self, sim_matrix_path):
#         with open(sim_matrix_path, "rb") as f:
#             self.label_sim_matrix = pickle.load(f)
#         print(f'Successfully load {sim_matrix_path}')

#     def load_image(self, file_path):
#         """
#         이미지를 로드하고, 필요한 경우 변환을 적용합니다.
#         """
#         image = cv2.imread(file_path)
#         if image is None:
#             print(f"Warning: Image not found or cannot be loaded at path: {file_path}")
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         if self.transform:
#             augmented = self.transform(image=image)
#             image = augmented["image"]
#         return image

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, index):
#         """
#         데이터셋에서 하나의 샘플을 가져옵니다.
#         """
#         # 이미지 로드
#         file_name = self.file_names[index]
#         file_path = os.path.join(self.CFG.parent_path, self.CFG.train_path, file_name)
#         image = self.load_image(file_path)

#         # 레이블 처리
#         target = self.targets[index]
#         inst_target = self.inst_target[index]
#         triplet_label = self.triplet[index]
#         verb_label = self.verbs[index]
#         inst_verbs = self.inst_verb[index]
#         target_verbs = self.target_verb[index]
#         inst = self.inst[index]

#         gt_labels = {
#             'i': inst,
#             't': target,
#             'v': verb_label,
#             'it': inst_target,
#             'iv': inst_verbs,
#             'tv': target_verbs,
#             'ivt': triplet_label
#         }

#         label = gt_labels[self.label_key]

#         # Similarity 값 가져오기
#         cos_sim = self.cos_sim_matrix[index]  # Shape: (N,)
#         label_sim = self.label_sim_matrix[index]  # Shape: (N,)

#         # 전체 라벨 가져오기
#         all_labels = np.array(self.labels[self.label_key])
#         selected_label = all_labels[index]

#         # 같은 클래스 샘플 선택 (Positive Pair)
#         indices_same_class = np.where(all_labels == selected_label)[0]
#         indices_same_class = indices_same_class[indices_same_class != index]

#         # 다른 클래스 샘플 선택 (Negative Pair)
#         indices_different_class = np.where(all_labels != selected_label)[0]

#         # ✅ Positive 샘플의 similarity 값을 0으로 설정하여 Negative Sampling에서 제외
#         final_sim = self.sampling_alpha * cos_sim + (1 - self.sampling_alpha) * label_sim
#         final_sim[indices_same_class] = 0  # Positive 샘플 제거

#         # ✅ Negative 샘플 선택 방식
#         if self.CFG.use_softmax_sampling:
#             # Softmax 기반 확률적 샘플링
#             final_sim_different_class = final_sim[indices_different_class]
#             sampling_prob = np.exp(final_sim_different_class) / np.sum(np.exp(final_sim_different_class))

#             # Negative 샘플링 (Weighted Random Choice)
#             num_neg_samples_needed = self.CFG.num_negative_samples * 2  
#             selected_indices_different_class = np.random.choice(
#                 indices_different_class, 
#                 size=num_neg_samples_needed, 
#                 replace=False, 
#                 p=sampling_prob  # 확률 기반 샘플링
#             )
#         else:
#             # ✅ 기존 방식 (Top-N Sampling)
#             sorted_indices_diff = np.argsort(-final_sim[indices_different_class])  # 내림차순 정렬
#             top_N = min(self.CFG.top_n, len(sorted_indices_diff))
#             selected_indices_different_class = indices_different_class[sorted_indices_diff[:top_N]]

#         # Negative 샘플 Mixup
#         negative_features = []
#         negative_labels = []

#         for i in range(0, len(selected_indices_different_class), 2):
#             idx1 = selected_indices_different_class[i]
#             idx2 = selected_indices_different_class[i + 1]

#             feature1 = self.features[idx1]
#             feature2 = self.features[idx2]

#             lam = np.random.beta(self.CFG.alpha, self.CFG.alpha)

#             mixed_feature = lam * feature1 + (1 - lam) * feature2

#             negative_features.append(mixed_feature)
#             negative_labels.append(all_labels[idx1])

#         negative_features = np.stack(negative_features)
#         negative_labels = np.array(negative_labels)

#         # Positive 샘플 선택 (기존 방식 유지)
#         if len(indices_same_class) > 0:
#             sim_same_class = cos_sim[indices_same_class]
#             sorted_indices_same = np.argsort(sim_same_class)  
#             bottom_N = min(self.CFG.top_n, len(sorted_indices_same))
#             selected_indices_same_class = np.random.choice(indices_same_class[sorted_indices_same[:bottom_N]], size=1, replace=False)
#         else:
#             selected_indices_same_class = np.array([index])

#         # Positive 샘플의 특징과 레이블 가져오기
#         positive_feature = self.features[selected_indices_same_class[0]]
#         positive_label = all_labels[selected_indices_same_class[0]]

#         # Anchor, Positive, Negative 샘플 합치기
#         contrast_features = np.concatenate(([positive_feature], negative_features), axis=0)
#         contrast_labels = np.concatenate(([positive_label], negative_labels), axis=0)

#         indices_selected = np.arange(len(contrast_features))
#         np.random.shuffle(indices_selected)
#         contrast_features = contrast_features[indices_selected]
#         contrast_labels = contrast_labels[indices_selected]

#         return image, label, contrast_features, contrast_labels