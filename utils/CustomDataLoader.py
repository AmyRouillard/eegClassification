from torch.utils.data import Dataset
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import spectrogram
import torch


def transpose_stack(x):
    return x.transpose(0, 1).reshape(299, 400).transpose(0, 1)


def transpose_stack_eeg_spec(x):
    return x.transpose(0, 1).reshape(129, 7 * 20).transpose(0, 1)


def normalize(x):
    # Calculate mean and standard deviation once
    mean = torch.mean(x)
    std = torch.std(x)
    # Avoid reshaping multiple times
    x_flat = x.reshape(1, -1)
    # Normalize using calculated mean and std
    return (x_flat - mean) / std


def tile(x):
    return torch.tile(x, (3, 1, 1))


def normalize_special(x):
    # Use broadcasting to avoid unnecessary reshape operations
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
    return (x * std - mean).reshape(3, 400, 299)


def normalize_special_eeg_spec(x):
    # Use broadcasting to avoid unnecessary reshape operations
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
    return (x * std - mean).reshape(3, 7 * 20, 129)


def min_max_scaling(x):
    # Calculate min and max once
    min_val = torch.min(x, dim=1).values.unsqueeze(1)
    max_val = torch.max(x, dim=1).values.unsqueeze(1)

    # Normalize using calculated min and max
    return torch.div(x - min_val, max_val - min_val)


class CustomDataset(Dataset):
    def __init__(
        self,
        data_dir,
        data_type,
        data_info_dict,
        transform=None,
    ):
        """
        A custom data loader for the harmful brain activity dataset.
        CustomDataset inherits from torch.utils.data.Dataset
        and overrides the __len__ and __getitem__ methods.

        Parameters
        ----------
        data_dir : str
            The directory where the data is stored.
        data_type : str
            The type of data to load. Must be one of ['spec', 'eeg_raw', 'eeg_spec'].
        data_info_dict : dict
            A dictionary containing the data information. The keys are tuples of the form
            (data_id, item_id, offset) and the values are dictionaries containing the votes
            for each class.
        """

        self.data_dir = data_dir

        self.data_type = data_type

        if self.data_type not in ["eeg_spec", "spec", "eeg_raw"]:
            raise ValueError(
                "Invalid data type provided. Must be one of ['spec', 'eeg_raw', 'eeg_spec']"
            )

        if transform is None and self.data_type == "spec":
            self.transform = (
                torch.tensor,
                transpose_stack,
                normalize,
                tile,
                normalize_special,
            )
        elif transform is None and self.data_type == "eeg_spec":
            self.transform = (
                torch.tensor,
                transpose_stack_eeg_spec,
                normalize,
                tile,
                normalize_special_eeg_spec,
            )
        else:
            self.transform = transform

        self.item_list = [k for k in data_info_dict.keys()]
        self.data_info_dict = data_info_dict

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):

        item = self.item_list[idx]
        data_id, _, offset = item
        path = self.data_dir + str(data_id) + ".parquet"

        data = self.preprocessing(path, offset)

        class_votes = self.data_info_dict[item]["votes"]
        if class_votes is None:
            class_votes = np.array([])
            label = np.array([])
        else:
            label = np.argmax(class_votes)

        if self.transform:
            for trans in self.transform:
                data = trans(data)

        return data, class_votes

    def preprocessing(self, path, offset):
        freq = 200 if "eeg" in self.data_type else 0.5  # Hz

        data = pq.read_table(path).to_pandas()

        # fill nan with zeros
        data = data.fillna(0)
        # enforce data type to float32
        data = data.astype(np.float32)

        collected_data = []
        if self.data_type == "spec":
            for spec_type in ["LL", "RL", "LP", "RP"]:
                collected_data.append(data.filter(regex=spec_type).values)
        elif self.data_type in ["eeg_raw", "eeg_spec"]:
            for col in data.columns:
                collected_data.append(data[col].values)

        # concatenate spec_data into a single array of size (n,m,4)
        data = np.moveaxis(np.array(collected_data), 0, -1)

        if self.data_type == "spec":
            start = int((0 + offset) * freq)
            end = int((600 + offset) * freq) - 1

            # clip values between exp(-4) and exp(8)
            data = np.clip(data[start:end, :], np.exp(-4), np.exp(8))
            data = np.log(data)
            # move last axis to first
            data = np.moveaxis(data, -1, 0)
        elif self.data_type == "eeg_raw":

            # select the central 50 seconds of the data
            # start = int((0 + offset) * freq)
            # end = int((49 + offset) * freq)
            # select the central 10 seconds of the data
            start = int((20 + offset) * freq)
            end = int((29 + offset) * freq)

            data = data[start:end]
            # move last axis to first
            data = np.moveaxis(data, -1, 0)
        elif self.data_type == "eeg_spec":

            # select the central 50 seconds of the data
            # start = int((0 + offset) * freq)
            # end = int((49 + offset) * freq)
            # select the central 10 seconds of the data
            start = int((20 + offset) * freq)
            end = int((29 + offset) * freq)

            data = data[start:end]
            # apply spectrogram to across all channels, axis 0
            spec_data = []
            for d in range(data.shape[1]):
                _, _, Sxx = spectrogram(data[:, d], fs=freq)
                # clip
                Sxx = np.clip(Sxx, np.exp(-4), np.exp(8))
                spec_data.append(np.log(Sxx))

            data = np.moveaxis(np.array(spec_data), 0, -1)
            data = np.array(spec_data)
        else:
            raise ValueError(
                "Invalid data type provided. Must be one of ['spec', 'eeg_raw', 'eeg_spec']"
            )

        return data


# %%

# from torch.utils.data import DataLoader
# import pandas as pd
# import os

# # Load (train or test) data from csv file
# path = "C:/Users/Amy/Desktop/Green_Git/eegClassification/"
# data_type = "eeg_raw"  # "spec"  #"eeg_spec"
# train = True
# text = "train" if train else "test"

# # Test
# data_dir = (
#     f"sample_data/{text}_eegs/"
#     if "eeg" in data_type
#     else f"sample_data/{text}_spectrograms/"
# )
# data_dir = path + data_dir

# df = pd.read_csv(path + f"sample_data/{text}.csv")

# votes_cols = [
#     "seizure_vote",
#     "lpd_vote",
#     "gpd_vote",
#     "lrda_vote",
#     "grda_vote",
#     "other_vote",
# ]
# label_cols = (
#     ["eeg_id", "label_id", "eeg_label_offset_seconds"]
#     if "eeg" in data_type
#     else ["spectrogram_id", "label_id", "spectrogram_label_offset_seconds"]
# )
# offset = (
#     ["eeg_label_offset_seconds"]
#     if "eeg" in data_type
#     else ["spectrogram_label_offset_seconds"]
# )

# files = os.listdir(data_dir)
# df = df[
#     df["eeg_id" if "eeg" in data_type else "spectrogram_id"].isin(
#         [int(f.split(".")[0]) for f in files]
#     )
# ]

# # if info_cols not in df add it and set to zero
# for col in offset:
#     if col not in df.columns:
#         df[col] = 0
# # if df does not contain "label_id" add a unique label_id
# if "label_id" not in df.columns:
#     df["label_id"] = range(len(df))

# info = {}
# df_gr = df.groupby(label_cols)
# for name, group in df_gr:
#     # first row of group
#     info[name] = {"votes": group[votes_cols].values[0] if train else None}

# print()
# # %%

# dataset = CustomDataset(data_dir, data_type, info)
# train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# count = 0
# for data, label, class_probs in train_dataloader:
#     print(data.shape, label.shape, class_probs.shape)
#     count += 1
#     if count > 2:
#         break


# print()
