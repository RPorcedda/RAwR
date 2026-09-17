import os
import numpy as np
from utils import *# get_shift_operator, compute_etas, read_labels
from synth_model.generate_teacher import generate_data_from_S


# -------------------------------
# Config
# -------------------------------
dataset_dict = {
    "Actor":      [0, 2, 4, 8, 1303],
    "Chameleon":  [0, 6, 12, 29, 732],
    "Cora":       [0, 2, 3, 5, 168],
    "Cornell":    [0, 1, 2, 4, 94],
    "PubMed":     [0, 1, 2, 4, 171],
    "Squirrel":   [0, 7, 17, 166, 1905],
    "Texas":      [0, 1, 2, 3, 104],
    "Wisconsin":  [0, 1, 2, 4, 122]
}

AUGMENTATIONS = [None, "RepNodes", "RepEdges"]  # None -> just A (GCN shift), others as you defined
EPS_FOR_NONE = [0]                               # sensible default if augmentation=None ignores eps


for dataset, eps_list in dataset_dict.items():
    print(f"\nDATASET: {dataset}")
    S_obs = get_shift_operator(dataset, augmentation=None, eps=None)
    
    for aug in AUGMENTATIONS:
        print(f"AUGMENTATION: {aug}")
        current_eps_list = eps_list if aug is not None else EPS_FOR_NONE

        for eps in current_eps_list:
            directory = dataset + "/" + f"{aug}" + "/" + f"{eps}" + "/"
            os.makedirs(directory, exist_ok=True)

            S = get_shift_operator(dataset, augmentation=aug, eps=eps)
            X, Y_teacher = generate_data_from_S(S)
            np.savetxt(directory + "features.csv", X, delimiter=",")
            np.savetxt(directory + "labels.csv", Y_teacher, delimiter=",")