from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision.transforms import Normalize

to_tensor = lambda batch_size, x, y, z: (
    torch.tensor(x),
    torch.tensor(y),
    torch.tensor(z),
)

# crop my differ for different architectures
crop_reshape_transform = lambda batch_size, x, y, z: (
    x[:, :, 5 : 299 - 6, 0 : 288 // 4].reshape(batch_size, 288, 288).transpose(2, 1),
    y,
    z,
)

scale_transform = lambda batch_size, x, y, z: (  # scale to 0-1
    torch.div(
        x.reshape(batch_size, -1)
        - torch.min(x.reshape(batch_size, -1), dim=1).values.unsqueeze(1),
        torch.max(x.reshape(batch_size, -1), dim=1).values.unsqueeze(1)
        - torch.min(x.reshape(batch_size, -1), dim=1).values.unsqueeze(1),
    ).reshape(batch_size, 288, 288),
    y,
    z,
)

normalize = lambda batch_size, x, y, z: (
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
        torch.tile(x, (3, 1, 1, 1)).transpose(0, 1)
    ),
    y,
    z,
)


class CustomDataset(Dataset):
    def __init__(
        self,
        data_path,
        N_items,
        transform=(
            to_tensor,
            # x_transform,
            crop_reshape_transform,
            scale_transform,
            normalize,
        ),
    ):
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

        batch_size = data.shape[0]

        if self.transform:
            for trans in self.transform:
                # print(data.shape, data.reshape(-1, 1).shape)
                data, label, class_votes = trans(batch_size, data, label, class_votes)

        return data, label, class_votes


# # %%

# from torch.utils.data import DataLoader
# from torchvision.transforms import ToPILImage

# # Load (train or test) data from csv file
# data_path = "G:\\My Drive\\Sun\\ML Shock\\Final project\\data_prep\\train\\"
# N_items = 101

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

# dataset = CustomDataset(data_path, N_items)
# # already batched into batches of 64
# train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# x_transform = lambda x, y, z: (
#     x[0, :],
#     y[0, :],
#     z[0, :],
# )

# for data, label, class_votes in train_dataloader:
#     data, label, class_votes = x_transform(data, label, class_votes)
#     print(data.shape, label.shape, class_votes.shape)
#     break

# print()

# # # %%

# # sort the labels in accending order
# label, indices = torch.sort(label)
# data = data[indices, :, :]
# class_votes = class_votes[indices, :]

# # image show data
# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(10, 5))
# # display 20 images
# for idx in np.arange(data.shape[0]):
#     # if classes[label[idx]] != "seizure_vote":
#     #     continue
#     ax = fig.add_subplot(8, 8, idx + 1, xticks=[], yticks=[])
#     # show black and white images
#     # ax.imshow(data[idx, :, :])  # , cmap="gray"
#     img = ToPILImage()(data[idx, :, :])
#     ax.imshow(img)
#     ax.set_title(classes[label[idx]])

# plt.show()

# print()
