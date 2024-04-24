import sys
import pandas as pd
import gc
import numpy as np
import json

np.random.seed(0)
import os
import torch

sys.path.insert(0, "C:/Users/Amy/Desktop/Green_Git/eegClassification/utils")
from CustomDataLoaderNPY import CustomDatasetNPY
from torch.utils.data import DataLoader

from utils import create_data_loaders, run, save_data

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not train_on_gpu:
    print("CUDA is not available.  Training on CPU ...")
else:
    print("CUDA is available!  Training on GPU ...")
    print("GPU count", torch.cuda.device_count())

# Path to data subfolder
path = "C:/Users/Amy/Desktop/Green_Git/eegClassification/data/data/"
# Path to output to
path_out = "C:/Users/Amy/Desktop/Green_Git/eegClassification/models/"
# Path to save temporary data
save_path = "C:/Users/Amy/Desktop/Green_Git/eegClassification/data/tmp/"

# if path does not exist create it
if not os.path.exists(path_out):
    os.makedirs(path_out)

# number of subprocesses to use for data loading
# import multiprocessing as cpu
num_workers = 0  # cpu.cpu_count()  # - 1 #
# how many samples per batch to load
batch_size = 64  # 8  #
# Is a test?
test = False

# specify the image classes
classes = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_classes = len(classes)

get_batch_transform = lambda x, y: (
    x[0, :],
    y[0, :],
)

########

path_df = "C:\\Users\\Amy\\Desktop\\Green_Git\\eegClassification\\sample_data\\"
df = pd.read_csv(path_df + f"train.csv")

df["total_votes"] = df[
    ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
].sum(axis=1)

# if path_out + "split.json" exist load it
if os.path.exists(path_out + "split.json"):
    with open(path_out + "split.json", "r") as f:
        split = json.load(f)
    train_p = np.array(split["train"])
    test_p = np.array(split["test"])
    print("Loaded train/test split")
else:
    p_id = df["patient_id"].unique()
    np.random.shuffle(p_id)
    # test train split
    train_p = p_id[: int(0.8 * len(p_id))]
    test_p = p_id[int(0.8 * len(p_id)) :]
    print("Created train/test split")

    # record the split in dict and save to json
    split = {"train": train_p.tolist(), "test": test_p.tolist()}
    with open(path_out + "split.json", "w") as f:
        json.dump(split, f)

######

# # filter data, eg number of votes >5
# min_votes = [0, 5]
# # augment data (window shifting)
# augmentation = [False, True]
# # label smoothing
# label_smoothing = [0, 0.01]
# # number of epochs to train the model
# n_epochs = 40 if not test else 1
# # data type
# data_type = "eeg_spec"  # "eeg_raw" #"spec" #
# # model name
# model_name = "CustomCNN_eeg"
# input_shape = (3, 140, 129)

# # filter data for #votes >5
# min_votes = [0, 5]
# # augment data
# augmentation = [False]
# # label smoothing
# label_smoothing = [0, 0.01]
# # number of epochs to train the model
# n_epochs = 20 if not test else 1
# # data type
# data_type = "spec"  # "eeg_spec"  # "eeg_raw" #
# model_name = "CustomCNN"
# input_shape = (3, 400, 299)

# filter data, eg number of votes >5
min_votes = [0, 5]
# augment data (window shifting)
augmentation = [False, True]
# label smoothing
label_smoothing = [0, 0.01]
# number of epochs to train the model
n_epochs = 40 if not test else 1
# data type
data_type = "eeg_spec"  # "eeg_raw" #"spec" #
# model name
model_name = "CustomCNN_eeg_small"

input_shape = (1, 140, 129)
# select one of the 3 channels
transform = (lambda batch_size, x: x[:, 0, :, :].reshape(batch_size, *input_shape),)


first = True
for mv in min_votes:
    for aug in augmentation:

        train_loader, valid_loader, test_loader = create_data_loaders(
            path, df, train_p, test_p, data_type, mv, aug, batch_size, num_workers
        )

        if first:
            # save_data(
            #     [train_loader, valid_loader, test_loader],
            #     ["train", "valid", "test"],
            #     save_path,
            #     test,
            # )
            first = False
        else:
            save_data([train_loader, valid_loader], ["train", "valid"], save_path, test)

        train_data = CustomDatasetNPY(
            save_path + "train/",
            [str(i) for i in range(len(train_loader))],
            transform=transform,
        )
        train_loader = DataLoader(
            train_data, batch_size=1, shuffle=False, num_workers=num_workers
        )

        valid_data = CustomDatasetNPY(
            save_path + "valid/",
            [str(i) for i in range(len(valid_loader))],
            transform=transform,
        )
        valid_loader = DataLoader(
            valid_data, batch_size=1, shuffle=False, num_workers=num_workers
        )

        test_data = CustomDatasetNPY(
            save_path + "test/",
            [str(i) for i in range(len(test_loader))],
            transform=transform,
        )
        test_loader = DataLoader(
            test_data, batch_size=1, shuffle=False, num_workers=num_workers
        )

        for ls in label_smoothing:
            model_info = {}
            model_info["track_loss"] = []
            model_info["track_loss_val"] = []
            model_info["min_votes"] = mv
            model_info["augmentation"] = aug
            model_info["data_type"] = data_type

            run(
                path_out,
                model_name,
                ls,
                input_shape,
                n_epochs,
                model_info,
                train_loader,
                valid_loader,
                test_loader,
                is_test=test,
            )

            gc.collect()
            torch.cuda.empty_cache()

            # save model_info as json
            model_info_path = model_info["configs"]["path_model_out"].replace(
                ".pt", ".json"
            )
            with open(model_info_path, "w") as f:
                json.dump(model_info, f)

            print("Done")

        # permanently delete all files and folders in the train and valid tmp data folder
        for folder in ["train/", "valid/"]:
            for file in os.listdir(save_path + folder):
                os.remove(save_path + folder + file)
