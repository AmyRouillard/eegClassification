from torch.utils.data import Dataset
import numpy as np


class CustomDatasetNPY(Dataset):
    def __init__(
        self,
        data_path,
        data_files,
        transform=None,
    ):
        self.data_path = data_path
        self.data_files = data_files
        self.N_items = len(self.data_files)
        self.transform = transform

    def __len__(self):
        return self.N_items

    def __getitem__(self, idx):

        idx = self.data_files[idx]
        # read data from path, npy
        data = np.load(self.data_path + "images_" + str(idx) + ".npy")
        # label = np.load(self.data_path + "labels_" + str(idx) + ".npy")
        class_votes = np.load(self.data_path + "votes_" + str(idx) + ".npy")

        if self.transform:
            batch_size = data.shape[0]
            for trans in self.transform:
                # print(data.shape, data.reshape(-1, 1).shape)
                data = trans(batch_size, data)

        return data, class_votes


# # %%

# import torch
# from torchvision.transforms import Normalize
# from torch.utils.data import DataLoader
# from torchvision.transforms import ToPILImage
# import os

# # Load (train or test) data from csv file
# data_path = "C:\\Users\\Amy\\Desktop\\Green_Git\\eegClassification\\sample_data\\data_prep_all_spec\\train\\"
# # count the number of items in the directory
# N_items = len(os.listdir(data_path)) // 2
# print("Number of items", N_items)

# classes = [
#     "seizure_vote",
#     "lpd_vote",
#     "gpd_vote",
#     "lrda_vote",
#     "grda_vote",
#     "other_vote",
# ]
# N_classes = len(classes)

# # # %%

# dataset = CustomDatasetNPY(data_path, N_items)
# # already batched into batches of 64
# train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# get_batch_transform = lambda x, y: (
#     x[0, :],
#     y[0, :],
# )

# for data, class_votes in train_dataloader:
#     data, class_votes = get_batch_transform(data, class_votes)
#     print(data.shape, class_votes.shape)
#     break

# print()

# # # %%

# # sort the labels in accending order
# label = np.argmax(class_votes, axis=1)
# label, indices = torch.sort(label)
# data = data[indices, :, :]
# class_votes = class_votes[indices, :]

# # image show data
# import matplotlib.pyplot as plt

# batch_size = data.shape[0]
# fig = plt.figure(figsize=(10, 5))
# # display 20 images
# for idx in np.arange(batch_size):
#     # if classes[label[idx]] != "seizure_vote":
#     #     continue
#     ax = fig.add_subplot(batch_size // 4, 4, idx + 1, xticks=[], yticks=[])
#     # show black and white images
#     # ax.imshow(data[idx, :, :])  # , cmap="gray"
#     img = ToPILImage()(data[idx, :, :])
#     ax.imshow(img)
#     ax.set_title(classes[label[idx]])

# plt.show()

# print()
