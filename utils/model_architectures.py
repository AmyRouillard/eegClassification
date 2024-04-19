import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# define the CNN architecture
class CustomCNN(nn.Module):
    def __init__(self, input_shape, N_classes, batch_size=64):
        super(CustomCNN, self).__init__()

        self.input_shape = input_shape
        self.N_classes = N_classes
        self.batch_size = batch_size

        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(self.input_shape[0], 8, 3, padding=1)
        # convolutional layer (sees 16x16x16 tensor)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.N_out = (
            8
            * (self.input_shape[1] // 2 // 2 // 2)
            * (self.input_shape[2] // 2 // 2 // 2)
        )
        self.fc1 = nn.Linear(self.N_out, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, self.N_classes)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(self.batch_size, self.N_out)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = F.log_softmax(self.fc2(x), dim=1)

        return x


# define the CNN architecture
class TransNet_Resnet18(nn.Module):
    def __init__(self, input_shape, N_classes):
        super(TransNet_Resnet18, self).__init__()

        self.input_shape = input_shape

        self.N_classes = N_classes

        self.model = models.resnet18(pretrained=True)

        # freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # remove the last layer
        self.model.fc = nn.Identity()

        # a layer to go some shape (4,299,100) to (3,299,100)
        # self.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # replace first layer with new layer
        # self.model.conv1 = self.conv1

        # add new layer
        self.fc = nn.Linear(self.model.fc.in_features, self.N_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(self.fc(x), dim=1)

        return x


# define the CNN architecture
class TransNet_Efficientnetb0(nn.Module):
    def __init__(self, input_shape, N_classes):
        super(TransNet_Efficientnetb0, self).__init__()

        self.N_classes = N_classes

        self.model = models.efficientnet_b0(pretrained=True)
        # freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # # remove the last layer
        self.model.classifier[-1] = nn.Identity()

        # a layer to go some shape (4,299,100) to (3,299,100)
        # self.conv1 = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # replace first layer with new layer
        # self.model.features[0] = self.conv1

        # add new layer
        self.fc = nn.Linear(self.model.classifier[-1].in_features, self.N_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(self.fc(x), dim=1)

        return x


class EEGNet(nn.Module):
    """
    Example usage:
    model = EEGNet(nb_classes=4, Chans=64, Samples=128)
    print(model)
    """

    def __init__(
        self,
        nb_classes,
        Chans=64,
        Samples=128,
        dropoutRate=0.5,
        kernLength=64,
        F1=8,
        D=2,
        F2=16,
        norm_rate=0.25,
        dropoutType="Dropout",
    ):
        super(EEGNet, self).__init__()

        if dropoutType == "SpatialDropout2D":
            self.dropoutType = nn.Dropout2d
        elif dropoutType == "Dropout":
            self.dropoutType = nn.Dropout
        else:
            raise ValueError(
                "dropoutType must be one of SpatialDropout2D "
                "or Dropout, passed as a string."
            )

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(
                1,
                F1,
                kernel_size=(1, kernLength),
                padding=(0, kernLength // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            self.dropoutType(dropoutRate),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            self.dropoutType(dropoutRate),
        )

        # Fully connected layer
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * (Samples // 32), nb_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return F.log_softmax(x, dim=1)
