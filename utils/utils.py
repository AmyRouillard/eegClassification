import gc
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import os

np.random.seed(0)
import torch

# PyTorch model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from CustomDataLoader import CustomDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

module = __import__("model_architectures")
# from model_architectures import (
#     CustomCNN,
#     CustomCNN_eeg,
#     TransNet_Resnet18_unfrozen,
#     TransNet_Efficientnetb0_unfrozen,
# )

votes_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_classes = len(votes_cols)

get_batch_transform = lambda x, y: (
    x[0, :],
    y[0, :],
)


def get_data_info(df, data_type):

    # train is true id votes_cols are available
    train = all([col in df.columns for col in votes_cols])

    label_cols = (
        ["eeg_id", "label_id", "eeg_label_offset_seconds"]
        if "eeg" in data_type
        else ["spectrogram_id", "label_id", "spectrogram_label_offset_seconds"]
    )
    offset = (
        ["eeg_label_offset_seconds"]
        if "eeg" in data_type
        else ["spectrogram_label_offset_seconds"]
    )

    # if info_cols not in df add it and set to zero
    for col in offset:
        if col not in df.columns:
            df[col] = 0
    # if df does not contain "label_id" add a unique label_id
    if "label_id" not in df.columns:
        df["label_id"] = range(len(df))

    info = {}
    df_gr = df.groupby(label_cols)
    for name, group in df_gr:
        # first row of group
        info[name] = {"votes": group[votes_cols].values[0] if train else None}

    return info


def shuffle(info):
    info_shuffled = {}
    keys = [k for k in info.keys()]
    np.random.shuffle(keys)
    for key in keys:
        info_shuffled[key] = info[key]

    return info_shuffled


def egg_spec_augmentation(info, rates=[0.75, 0.2, 0.2, 0.2, 0.2, 0.2]):

    info_aug = info.copy()

    offset = [-1, 1, -2, 2, -3, 3, -4, 4]
    keys = [k for k in info_aug.keys()]
    N_item = len(keys)

    counts = [0] * 6
    for k in info_aug.keys():
        idx = np.argmax(info_aug[k]["votes"])
        counts[idx] += 1

    for key in keys:
        # generate random between 0 and 1
        r = np.random.rand()
        idx = np.argmax(info_aug[key]["votes"])

        if r < rates[idx]:
            # select a random offset
            off = np.random.choice(offset)
            new_key = (key[0], key[1], key[2] + off)
            # if new_key not in info add it
            if new_key not in info_aug.keys():
                info_aug[new_key] = info_aug[key]

            counts[idx] += 1

        # if len(info) >= N_item*(1+0.4):
        #     break

    return info_aug


def lrfn(
    epoch,
    epochs,
    mode="cos",
    lr_start=2e-4,
    lr_max=3e-5 * 64,
    lr_min=1e-5,
    lr_ramp_ep=4,
    lr_sus_ep=0,
    lr_decay=0.75,
):
    if epoch < lr_ramp_ep:
        lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
    elif epoch < lr_ramp_ep + lr_sus_ep:
        lr = lr_max
    elif mode == "exp":
        lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
    elif mode == "step":
        lr = lr_max * lr_decay ** ((epoch - lr_ramp_ep - lr_sus_ep) // 2)
    elif mode == "cos":
        decay_total_epochs, decay_epoch_index = (
            epochs - lr_ramp_ep - lr_sus_ep + 3,
            epoch - lr_ramp_ep - lr_sus_ep,
        )
        phase = np.pi * decay_epoch_index / decay_total_epochs
        lr = (lr_max - lr_min) * 0.5 * (1 + np.cos(phase)) + lr_min
    return lr


def train_func(
    model,
    train_loader,
    valid_loader,
    path_model_out,
    n_epochs,
    learning_rate,
    label_smoothing,
    test=False,
    valid_loss_min=np.Inf,
):
    criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)
    optimizer = optim.Adam(model.parameters())

    track_loss = []
    track_loss_val = []

    max_samples = 1

    train_on_gpu = torch.cuda.is_available()

    for epoch in range(1, n_epochs + 1):

        for g in optimizer.param_groups:
            g["lr"] = learning_rate[epoch - 1]  # lrfn(epoch, n_epochs)
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        count = 0

        for data, votes in tqdm(train_loader):
            if test and count >= max_samples:
                break

            data, votes = get_batch_transform(data, votes)
            # offset vote by adding label smoothing as offset
            votes = votes * (1 - label_smoothing) + label_smoothing / N_classes

            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, votes = data.cuda(), votes.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)

            if torch.isnan(output).sum() > 0:
                print("Nan in output")
            else:

                # loss
                loss = criterion(output.float(), F.log_softmax(votes.float(), dim=1))

                # update training loss
                train_loss += loss.item()  # *data.size(0)

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                optimizer.step()

                count += 1

        train_loss = train_loss / count  # len(train_loader)#/batch_size

        torch.cuda.empty_cache()
        ######################
        # validate the model #
        ######################
        model.eval()
        count = 0
        for data, votes in tqdm(valid_loader):
            if test and count >= max_samples:
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
                # calculate the batch loss
                loss = criterion(output.float(), F.log_softmax(votes.float(), dim=1))
                # update average validation loss
                valid_loss += loss.item()  # *data.size(0)

                count += 1

        torch.cuda.empty_cache()
        # calculate average losses

        valid_loss = valid_loss / count  # len(valid_loader)#/batch_size

        track_loss.append(train_loss)
        track_loss_val.append(valid_loss)

        # print training/validation statistics
        print("{}; \t{:.6f}; \t{:.6f}".format(epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            torch.save(model.state_dict(), path_model_out)
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    valid_loss_min, valid_loss
                )
            )
            valid_loss_min = valid_loss

        gc.collect()

    return track_loss, track_loss_val


def create_data_loaders(
    path,
    df,
    train_p,
    test_p,
    data_type,
    min_votes,
    augmentation,
    batch_size=64,
    num_workers=0,
):
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
    path_out,
    model_name,
    label_smoothing,
    data_type,
    n_epochs,
    model_info,
    train_loader,
    valid_loader,
    test_loader,
    is_test=False,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_on_gpu = torch.cuda.is_available()

    if data_type == "spec":
        input_shape = (3, 400, 299)
    elif data_type == "eeg_spec":
        input_shape = (3, 140, 129)  # (20,129,43)
    elif data_type == "eeg_raw":
        input_shape = (20, 9800)

    # get model name as executable function
    model = getattr(module, model_name)(input_shape=input_shape, N_classes=N_classes)
    # if model_name == "CustomCNN":
    #     model = CustomCNN(input_shape=input_shape, N_classes=N_classes)
    # elif model_name == "CustomCNN_eeg":
    #     model = CustomCNN_eeg(input_shape=input_shape, N_classes=N_classes)
    # elif model_name == "TransNet_Resnet18":
    #     model = TransNet_Resnet18_unfrozen(input_shape=input_shape, N_classes=N_classes)
    # elif model_name == "TransNet_Efficientnetb0":
    #     model = TransNet_Efficientnetb0_unfrozen(
    #         input_shape=input_shape, N_classes=N_classes
    #     )
    # else:
    #     raise ValueError("Model not found")

    if train_on_gpu:
        model.cuda()

    num_parameters = sum(p.numel() for p in model.parameters())
    # print("Number of parameters in the model", num_parameters)
    model_info["num_parameters"] = num_parameters

    # number of trainable parameters
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Number of trainable parameters in the model", num_parameters)
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
            test_loss += loss.item()  # *data.size(0)

            # dummy is a tensor filled with 1/6 of shape [64,6]
            dummy = torch.ones(data.size(0), N_classes).to(device)
            dummy = dummy / N_classes
            loss_baseline = criterion(
                F.log_softmax(dummy, dim=1), F.log_softmax(votes.float(), dim=1)
            )
            test_loss_baseline += loss_baseline.item()

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


def save_data(loaders, save_path, test=False):
    for data_loader, text in zip(
        loaders,
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


def run_inference(model_name, path_model_out, test_loader, input_shape, is_test=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_on_gpu = torch.cuda.is_available()

    # get model name as executable function
    model = getattr(module, model_name)(input_shape=input_shape, N_classes=N_classes)
    model.load_state_dict(torch.load(path_model_out))

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
    predictions = []
    labels = []
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
            test_loss += loss.item()  # *data.size(0)

            # dummy is a tensor filled with 1/6 of shape [64,6]
            dummy = torch.ones(data.size(0), N_classes).to(device)
            dummy = dummy / N_classes
            loss_baseline = criterion(
                F.log_softmax(dummy, dim=1), F.log_softmax(votes.float(), dim=1)
            )
            test_loss_baseline += loss_baseline.item()

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

        predictions.append(output.cpu().detach().numpy())
        labels.append(votes.cpu().detach().numpy())

    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)

    # average test loss
    test_loss = test_loss / count  # len(test_loader.dataset)  # /batch_size
    print("Test Loss: {:.6f}\n".format(test_loss))

    test_loss_baseline = (
        test_loss_baseline / count
    )  # len(test_loader.dataset)  # /batch_size
    print("Test Loss Baseline: {:.6f}\n".format(test_loss_baseline))

    cm = confusion_matrix(cm_y_true, cm_y_pred)
    cm_p = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    return test_loss, test_loss_baseline, cm, cm_p, predictions, labels
