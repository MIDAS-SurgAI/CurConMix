import os
import pandas as pd
from sklearn.model_selection import GroupKFold
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer 

def split_selector( case='cholect50'):
    switcher = {
        'cholect50': {
            'train': [1, 15, 26, 40, 52, 65, 79, 2, 18, 27, 43, 56, 66, 92, 4, 22, 31, 47, 57, 68, 96, 5, 23, 35, 48, 60, 70, 103, 13, 25, 36, 49, 62, 75, 110],
            'val'  : [8, 12, 29, 50, 78],
            'test' : [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]
        },
        'cholect50-challenge': {
            'train': [1, 15, 26, 40, 52, 79, 2, 27, 43, 56, 66, 4, 22, 31, 47, 57, 68, 23, 35, 48, 60, 70, 13, 25, 49, 62, 75, 8, 12, 29, 50, 78, 6, 51, 10, 73, 14, 32, 80, 42],
            'val':   [5, 18, 36, 65, 74],
            'test':  [92, 96, 103, 110, 111]
        },
        'cholect45-crossval': {
            1: [79,  2, 51,  6, 25, 14, 66, 23, 50,],
            2: [80, 32,  5, 15, 40, 47, 26, 48, 70,],
            3: [31, 57, 36, 18, 52, 68, 10,  8, 73,],
            4: [42, 29, 60, 27, 65, 75, 22, 49, 12,],
            5: [78, 43, 62, 35, 74,  1, 56,  4, 13,],
        },
        'cholect50-crossval': {
            1: [79,  2, 51,  6, 25, 14, 66, 23, 50, 111],
            2: [80, 32,  5, 15, 40, 47, 26, 48, 70,  96],
            3: [31, 57, 36, 18, 52, 68, 10,  8, 73, 103],
            4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
            5: [78, 43, 62, 35, 74,  1, 56,  4, 13,  92],
        },
    }
    return switcher.get(case)

def get_folds(CFG):
    print("\033[94mPreparing the data\033[0m")
    # Read the dataframe
    train = pd.read_csv(os.path.join(CFG.parent_path, CFG.path_csv))

    print("Preprocessing the data...")

    # Start a folds df to map the folds
    folds = train.copy()
    fold_map = split_selector(CFG.split_selector)

    # Initialize the fold column
    folds["fold"] = -1
    # Function to find indices of ones using NumPy
    def find_indices_of_ones(binary_list):
        return tuple(np.nonzero(binary_list)[0])
    def index_to_label_mapping(binary_array):
        index_to_label = {}
        current_label = 0
        # Process each binary array row
        for binary_list in binary_array:
            indices_tuple = find_indices_of_ones(binary_list)
            if indices_tuple not in index_to_label:
                index_to_label[indices_tuple] = current_label
                current_label += 1

        # Assign labels to each binary list
        labels = [index_to_label[find_indices_of_ones(row)] for row in binary_array]

        # Convert labels to a NumPy array if needed
        label_array = np.array(labels)
        return label_array
    def index_to_label_mapping_txt(binary_array, dict_mapping):
        text_labels = []

        # Process each binary array row
        for binary_list in binary_array:
            indices_tuple = find_indices_of_ones(binary_list)
            
            if not indices_tuple:
                current_label = dict_mapping[-1]
            else:
                # Get the labels for all indices
                labels = [dict_mapping[x] for x in indices_tuple]
                # Join the labels with 'and' between all items
                current_label = ' and '.join(labels)
            
            text_labels.append(current_label)

        return text_labels
  
    # Get column indices
    index_no = folds.columns.get_loc(CFG.col0)
    t_index_no = folds.columns.get_loc('t0')
    v_index_no = folds.columns.get_loc('v0')
    inst_index_no = folds.columns.get_loc('inst0')

    binary_array_triplet = folds.iloc[:, index_no:index_no + 100].values.astype(int)
    binary_array_target = folds.iloc[:, t_index_no:t_index_no + 15].values.astype(int)
    binary_array_verb = folds.iloc[:, v_index_no:v_index_no + 10].values.astype(int)
    binary_array_inst = folds.iloc[:, inst_index_no:inst_index_no + 6].values.astype(int)
    binary_array_inst_target = np.concatenate((binary_array_inst, binary_array_target), axis=1)
    binary_arrary_inst_verb = np.concatenate((binary_array_inst, binary_array_verb), axis=1)
    binary_array_target_verb = np.concatenate((binary_array_target, binary_array_verb), axis=1)


    trip_num = binary_array_triplet.sum(axis=1)
    trip_num = list(trip_num)

    # Convert binary sequences to string labels
    triplet_labels = index_to_label_mapping(binary_array_triplet)
    target_labels = index_to_label_mapping(binary_array_target)
    verb_labels = index_to_label_mapping(binary_array_verb)
    inst_labels = index_to_label_mapping(binary_array_inst)
    inst_target_labels = index_to_label_mapping(binary_array_inst_target)
    inst_verb_labels = index_to_label_mapping(binary_arrary_inst_verb)
    target_verb_labels = index_to_label_mapping(binary_array_target_verb)

    # Adding the labels to the DataFrame
    folds["triplet"] = triplet_labels
    folds["verb"] = verb_labels
    folds["instrument"] = inst_labels
    folds["target"] = target_labels
    folds["inst_target"] = inst_target_labels
    folds["inst_verb"] = inst_verb_labels
    folds["target_verb"] = target_verb_labels
    folds["num"] = trip_num
    
    # Assign each video to a fold based on the predefined lists in fold_map
    for fold, video_list in fold_map.items():
        video_list = [f"VID{vid:02d}" for vid in video_list]
        if isinstance(fold, int):
            folds.loc[folds["video"].isin(video_list), "fold"] = fold - 1
        else:
            folds.loc[folds["video"].isin(video_list), "fold"] = fold

    print("Dataset ready!\n")
    wrong_idx = folds[folds["fold"] != -1].index
    folds = folds.loc[wrong_idx].reset_index(drop=True)

    return folds