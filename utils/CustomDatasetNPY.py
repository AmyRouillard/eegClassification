from torch.utils.data import Dataset
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import spectrogram


class CustomDataset(Dataset):
    def __init__(self, data_path, N_items, transform=None):
        self.data_path = data_path
        self.N_items = N_items
        self.transform = transform

    def __len__(self):
        return self.N_items

    def __getitem__(self, idx):

        # read data from path, npy
        data = np.load(self.data_path + "images_" + str(idx) + ".npy")
        label = np.load(self.data_path + "labels_" + str(idx) + ".npy")
        class_votes = np.load(self.data_path + "votes_" + str(idx) + ".npy")

        if self.transform:
            for trans in self.transform:
                # print(data.shape, data.reshape(-1, 1).shape)
                data = trans(data)

        return data, label, class_votes
