from torch.utils.data import Dataset
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import spectrogram


class CustomDataset(Dataset):
    def __init__(self, data_dir, data_type, data_info_dict, transform=None):
        self.data_dir = data_dir
        self.data_type = data_type
        self.transform = transform

        self.item_list = [k for k in data_info_dict.keys()]
        self.data_info_dict = data_info_dict

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):

        data_id, item_id, offset = self.item_list[idx]
        path = self.data_dir + str(data_id) + ".parquet"

        data = self.preprocessing(path, offset)
        class_votes = self.data_info_dict[(data_id, item_id, offset)]["votes"]
        if class_votes is None:
            class_votes = np.array([])
            label = np.array([])
        else:
            label = np.argmax(class_votes)
        # if class_votes is not None:
        #     class_probs = class_votes / np.sum(class_votes)
        #     label = np.argmax(class_votes)
        # else:
        #     class_probs = np.array([])
        #     label = np.array([])

        if self.transform:
            for trans in self.transform:
                # print(data.shape, data.reshape(-1, 1).shape)
                data = trans(data)

        return data, label, class_votes

    def preprocessing(self, path, offset):
        freq = 200 if "eeg" in self.data_type else 0.5  # Hz

        data = pq.read_table(path).to_pandas()

        # fill nan with zeros
        data = data.fillna(0)

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
            data = np.log(data[start:end, :] + 1)
        elif self.data_type == "eeg_raw":
            start = int((0 + offset) * freq)
            end = int((49 + offset) * freq)
            data = data[start:end]
        elif self.data_type == "eeg_spec":
            start = int((0 + offset) * freq)
            end = int((49 + offset) * freq)
            data = data[start:end]
            # apply spectrogram to across all channels, axis 0
            spec_data = []
            for d in range(data.shape[1]):
                _, _, Sxx = spectrogram(data[:, d], fs=freq)
                spec_data.append(np.log(Sxx + 1))

            data = np.moveaxis(np.array(spec_data), 0, -1)
        else:
            raise ValueError(
                "Invalid data type provided. Must be one of ['spec', 'eeg_raw', 'eeg_spec']"
            )

        # move last axis to first
        data = np.moveaxis(data, -1, 0)

        return data


# # %%

# from torch.utils.data import DataLoader
# import pandas as pd
# import os

# # Load (train or test) data from csv file
# path = "C:/Users/Amy/Desktop/Green_Git/eegClassification/"
# data_type = "eeg_spec"  # "spec" # "eeg_raw"
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
