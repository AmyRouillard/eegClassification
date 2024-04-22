import gc
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix

np.random.seed(0)
import torch

# PyTorch model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


def egg_spec_augmentation(info):

    info_aug = info.copy()

    offset = [-1, 1, -2, 2, -3, 3, -4, 4]
    keys = [k for k in info_aug.keys()]
    N_item = len(keys)

    counts = [0] * 6
    for k in info_aug.keys():
        idx = np.argmax(info_aug[k]["votes"])
        counts[idx] += 1

    for key in keys:

        # p = [x/np.sum(counts) for x in counts]
        # p_inv = [1/x for x in p]

        rates = [0.75, 0.2, 0.2, 0.2, 0.2, 0.2]
        # [(1-x)/x/np.sum(p_inv) for x in p]#[x/np.sum(p_inv) for x in p_inv]# [(1-x/np.sum(counts)) for x in counts]

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

    print("Epoch: \tTraining Loss:  \tValidation Loss:")
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
