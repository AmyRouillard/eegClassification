import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)
import os
import torch

sys.path.insert(0, "C:/Users/Amy/Desktop/Green_Git/eegClassification/utils")
from CustomDataLoaderNPY import CustomDatasetNPY
from torch.utils.data import DataLoader

from utils import run_inference

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

# if path_out + "split.json" exist load it
if os.path.exists(path_out + "split.json"):
    with open(path_out + "split.json", "r") as f:
        split = json.load(f)
    train_p = np.array(split["train"])
    test_p = np.array(split["test"])
    print("Loaded train/test split")
else:
    raise ValueError("No train/test split found")

path_df = "C:\\Users\\Amy\\Desktop\\Green_Git\\eegClassification\\sample_data\\"
df = pd.read_csv(path_df + f"train.csv")
N_samples = len(df[df["patient_id"].isin(test_p)])
N_samples = np.ceil(N_samples / batch_size).astype(int)

######

# # data type
# data_type = "spec"
# # model name
# model_name = "CustomCNN"
# index = 0

# data type
data_type = "eeg_raw"
# model name
model_name = "CustomCNN_eeg"
# index = 0

######
# for index in range(4):
for index in range(1):
    test_data = CustomDatasetNPY(
        save_path + f"test_{data_type}/",
        [str(i) for i in range(N_samples)],
    )
    test_loader = DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=num_workers
    )
    if data_type == "spec":
        input_shape = (3, 400, 299)
    elif data_type == "eeg_spec":
        input_shape = (3, 140, 129)  # (20,129,43)
    elif data_type == "eeg_raw":
        input_shape = (20, 9800)

    path_model = path_out + f"model_{model_name}_{index}.pt"

    accuracy, test_loss, test_loss_baseline, cm, cm_p, predictions, labels = (
        run_inference(model_name, path_model, test_loader, input_shape, is_test=test)
    )

    #####

    # save results

    loss_dict = {
        "test_loss_kl": test_loss,
        "test_loss_baseline_kl": test_loss_baseline,
        "accuracy": accuracy,
    }
    # save loss_dict to json
    with open(path_out + f"eval_{model_name}_{index}.json", "w") as f:
        json.dump(loss_dict, f)

    # save predictions and labels as npy
    np.save(path_out + f"predictions_{model_name}_{index}.npy", predictions)
    np.save(path_out + f"labels_{model_name}_{index}.npy", labels)

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xticks(ticks=np.arange(6) + 0.5, labels=classes)
    plt.yticks(ticks=np.arange(6) + 0.5, labels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig(path_out + f"cm_{model_name}_{index}.png")

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_p * 100, annot=True, fmt=".1f", cmap="Blues")
    plt.xticks(ticks=np.arange(6) + 0.5, labels=classes)
    plt.yticks(ticks=np.arange(6) + 0.5, labels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig(path_out + f"cm_probs_{model_name}_{index}.png")
