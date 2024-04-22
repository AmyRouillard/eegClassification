import multiprocessing as cpu
import sys
import pandas as pd
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import json


np.random.seed(0)
import os
import torch

# PyTorch model
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.insert(0, "C:/Users/Amy/Desktop/Green_Git/eegClassification/utils")

from CustomDataLoader import CustomDataset
from CustomDataLoaderNPY import CustomDatasetNPY
from torch.utils.data import DataLoader
from model_architectures import (
    CustomCNN,
    CustomCNN_eeg,
    TransNet_Resnet18_unfrozen,
    TransNet_Efficientnetb0_unfrozen,
)

from utils import get_data_info, shuffle, lrfn, train_func, egg_spec_augmentation

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not train_on_gpu:
    print("CUDA is not available.  Training on CPU ...")
else:
    print("CUDA is available!  Training on GPU ...")
    print("GPU count", torch.cuda.device_count())

path = "C:/Users/Amy/Desktop/Green_Git/eegClassification/data/data/"  # sample_data
path_df = "C:\\Users\\Amy\\Desktop\\Green_Git\\eegClassification\\sample_data\\"
path_out = "C:/Users/Amy/Desktop/Green_Git/eegClassification/models/"
# if path does not exist create it
if not os.path.exists(path_out):
    os.makedirs(path_out)

# number of subprocesses to use for data loading
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


def create_data_loaders(data_type, min_votes, augmentation):
    data_dir = f"train_eegs/" if "eeg" in data_type else f"train_spectrograms/"
    data_dir = path + data_dir

    df_test = df[df["patient_id"].isin(test_p)]
    print("Fetching info...")
    info_test = get_data_info(df_test, data_type)

    dataset_test = CustomDataset(data_dir, data_type, info_test)
    test_loader = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    df_train = df[df["patient_id"].isin(train_p)]
    df_train = df_train[df_train["total_votes"] >= min_votes]
    info_train = get_data_info(df_train, data_type)
    info_train = shuffle(info_train)

    if augmentation:
        info_train_aug = egg_spec_augmentation(info_train)
    else:
        info_train_aug = info_train.copy()

    print("Creating data loaders...")
    dataset = CustomDataset(data_dir, data_type, info_train_aug)
    # split the dataset into training and validation
    valid_size = 0.2
    num_train = len(dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    valid_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader


def run(
    label_smoothing,
    data_type,
    n_epochs,
    model_info,
    train_loader,
    valid_loader,
    test_loader,
    is_test=False,
):

    model_name = (
        "CustomCNN_eeg"
        if "eeg" in data_type
        else "CustomCNN"  # "TransNet_Efficientnetb0" #"TransNet_Resnet18" #
    )

    if data_type == "spec":
        input_shape = (3, 400, 299)
    elif data_type == "eeg_spec":
        input_shape = (3, 140, 129)  # (20,129,43)
    elif data_type == "eeg_raw":
        input_shape = (20, 9800)

    if model_name == "CustomCNN":
        model = CustomCNN(input_shape=input_shape, N_classes=N_classes)
    elif model_name == "CustomCNN_eeg":
        model = CustomCNN_eeg(input_shape=input_shape, N_classes=N_classes)
    elif model_name == "TransNet_Resnet18":
        model = TransNet_Resnet18_unfrozen(input_shape=input_shape, N_classes=N_classes)
    elif model_name == "TransNet_Efficientnetb0":
        model = TransNet_Efficientnetb0_unfrozen(
            input_shape=input_shape, N_classes=N_classes
        )
    else:
        raise ValueError("Model not found")

    if train_on_gpu:
        model.cuda()

    num_parameters = sum(p.numel() for p in model.parameters())
    print("Number of parameters in the model", num_parameters)
    model_info["num_parameters"] = num_parameters

    # number of trainable parameters
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters in the model", num_parameters)
    model_info["num_trainable_parameters"] = num_parameters

    # check if "f"./model_{model_name}_*" exists and add 1 to the index
    index = 0
    model_path = path_out + "model_{model_name}_{index}.pt"
    while os.path.exists(model_path.format(model_name=model_name, index=index)):
        index += 1

    configs = {
        "n_epochs": n_epochs,
        "path_model_out": path_out + f"model_{model_name}_{index}.pt",
        "learning_rate": [lrfn(epoch, n_epochs) for epoch in np.arange(n_epochs)],
        "label_smoothing": label_smoothing,
    }

    model_info["configs"] = configs

    valid_loss_min = (
        np.min(model_info["track_loss_val"])
        if len(model_info["track_loss_val"]) > 0
        else np.Inf
    )

    track_loss, track_loss_val = train_func(
        model,
        train_loader,
        valid_loader,
        valid_loss_min=valid_loss_min,
        test=is_test,
        **configs,
    )

    model_info["track_loss"] += track_loss
    model_info["track_loss_val"] += track_loss_val

    model.load_state_dict(torch.load(configs["path_model_out"]))

    criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)

    # track test loss
    test_loss = 0.0
    test_loss_baseline = 0.0
    class_correct = list(0.0 for i in range(N_classes))
    class_total = list(0.0 for i in range(N_classes))
    model.eval()

    max_samples = 1
    cm_y_pred = []
    cm_y_true = []
    # iterate over test data
    count = 0
    for data, votes in tqdm(test_loader):
        if is_test and count >= max_samples:
            break

        data, votes = get_batch_transform(data, votes)
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, votes = data.cuda(), votes.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        if torch.isnan(output).sum() > 0:
            print("Nan in output")
        else:
            count += 1
            # calculate the batch loss
            loss = criterion(output.float(), F.log_softmax(votes.float(), dim=1))
            # dummy is a tensor filled with 1/6 of shape [64,6]
            dummy = torch.ones(data.size(0), N_classes).to(device)
            dummy = dummy / N_classes
            loss_baseline = criterion(output.float(), F.log_softmax(dummy, dim=1))
            # update test loss
            test_loss += loss.item()  # *data.size(0)
            test_loss_baseline += loss_baseline.item()  # *data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            target = torch.argmax(votes, axis=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = (
                np.squeeze(correct_tensor.numpy())
                if not train_on_gpu
                else np.squeeze(correct_tensor.cpu().numpy())
            )
            # calculate test accuracy for each object class
            for i in range(target.shape[0]):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

                cm_y_pred.append(pred[i].item())
                cm_y_true.append(target.data[i].item())

    # average test loss
    print()
    test_loss = test_loss / count  # len(test_loader.dataset)  # /batch_size
    print("Test Loss: {:.6f}\n".format(test_loss))

    test_loss_baseline = (
        test_loss_baseline / count
    )  # len(test_loader.dataset)  # /batch_size
    print("Test Loss Baseline: {:.6f}\n".format(test_loss_baseline))

    cm = confusion_matrix(cm_y_true, cm_y_pred)
    cm_p = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    model_info["test_loss"] = test_loss
    model_info["test_loss_baseline"] = test_loss_baseline
    model_info["confusion_matrix"] = cm.tolist()
    model_info["confusion_matrix_p"] = cm_p.tolist()


save_path = "C:/Users/Amy/Desktop/Green_Git/eegClassification/data/tmp/"


# filter data for #votes >5
min_votes = [0, 5]
# augment data
augmentation = [False, True]
# label smoothing
label_smoothing = [0, 0.01]
# number of epochs to train the model
n_epochs = 40 if not test else 1
# data type
data_type = "eeg_spec"  # "eeg_raw" #"spec" #


for mv in min_votes:
    for aug in augmentation:

        train_loader, valid_loader, test_loader = create_data_loaders(
            data_type, mv, aug
        )

        for data_loader, text in zip(
            [train_loader, valid_loader, test_loader],
            ["train", "valid", "test"],
        ):
            tmp_path = save_path + f"{text}/"
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)

            count = 0
            for X, votes in tqdm(data_loader, desc=f"Saving {text} data"):

                if test and count >= 3:
                    break

                # save the images
                np.save(tmp_path + f"images_{count}.npy", X.numpy())
                # save the votes
                np.save(tmp_path + f"votes_{count}.npy", votes.numpy())
                count += 1

        train_data = CustomDatasetNPY(
            save_path + "train/",
            [str(i) for i in range(len(train_loader))],
        )
        train_loader = DataLoader(
            train_data, batch_size=1, shuffle=False, num_workers=num_workers
        )

        valid_data = CustomDatasetNPY(
            save_path + "valid/",
            [str(i) for i in range(len(valid_loader))],
        )
        valid_loader = DataLoader(
            valid_data, batch_size=1, shuffle=False, num_workers=num_workers
        )

        test_data = CustomDatasetNPY(
            save_path + "test/",
            [str(i) for i in range(len(test_loader))],
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
                ls,
                data_type,
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

        # permanently delete all files and folders in the data folder
        for folder in ["train/", "valid/", "test/"]:
            for file in os.listdir(save_path + folder):
                os.remove(save_path + folder + file)


# filter data for #votes >5
min_votes = [0, 5]
# augment data
augmentation = [False]
# label smoothing
label_smoothing = [0, 0.01]
# number of epochs to train the model
n_epochs = 20 if not test else 1
# data type
data_type = "spec"  # "eeg_spec"  # "eeg_raw" #

for mv in min_votes:
    for aug in augmentation:

        train_loader, valid_loader, test_loader = create_data_loaders(
            data_type, mv, aug
        )

        for data_loader, text in zip(
            [train_loader, valid_loader, test_loader],
            ["train", "valid", "test"],
        ):
            tmp_path = save_path + f"{text}/"
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)

            count = 0
            for X, votes in tqdm(data_loader, desc=f"Saving {text} data"):
                print(X.shape, votes.shape)
                if test and count >= 3:
                    break

                # save the images
                np.save(tmp_path + f"images_{count}.npy", X.numpy())
                # save the votes
                np.save(tmp_path + f"votes_{count}.npy", votes.numpy())
                count += 1

        train_data = CustomDatasetNPY(
            save_path + "train/",
            [str(i) for i in range(len(train_loader))],
        )
        train_loader = DataLoader(
            train_data, batch_size=1, shuffle=False, num_workers=num_workers
        )

        valid_data = CustomDatasetNPY(
            save_path + "valid/",
            [str(i) for i in range(len(valid_loader))],
        )
        valid_loader = DataLoader(
            valid_data, batch_size=1, shuffle=False, num_workers=num_workers
        )

        test_data = CustomDatasetNPY(
            save_path + "test/",
            [str(i) for i in range(len(test_loader))],
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
                ls,
                data_type,
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

        # permanently delete all files and folders in the data folder
        for folder in ["train/", "valid/", "test/"]:
            for file in os.listdir(save_path + folder):
                os.remove(save_path + folder + file)
