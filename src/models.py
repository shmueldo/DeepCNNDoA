"""Subspace-Net 
Details
----------
Name: models.py
Authors: Dor Haim Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the tested NN-model, which used for simulation.
The implemented models:
    * DeepCNN: Deep learning algorithm as described in:
        [1] G. K. Papageorgiou, M. Sellathurai, and Y. C. Eldar, “Deep networks
        for direction-of-arrival estimation in low SNR,” IEEE Trans. Signal
        Process., vol. 69, pp. 3714-3729, 2021.

"""

# Imports
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import warnings
from torchsummary import summary


warnings.simplefilter("ignore")
# Constants
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelGenerator(object):
    """
    Generates an instance of the desired model, according to model configuration parameters.
    """

    def __init__(self):
        """
        Initialize ModelParams object.
        """
        self.model_type = "None"
        self.grid_size = None

    def set_grid_size(self, grid_size: int = None):
        """
        Set the value of grid size for multi-classifier model.

        Parameters:
            grid_size (int): The grid size.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            ValueError: If grid size parameter is not provided for model.
        """
        if not isinstance(grid_size, int):
            raise ValueError(
                "ModelParams.set_grid_size: grid size parameter must be provided"
            )
        self.grid_size = grid_size
        return self

    def set_model_type(self, model_type: str):
        """
        Set the model type.

        Parameters:
            model_type (str): The model type.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            ValueError: If model type is not provided.
        """
        if not isinstance(model_type, str):
            raise ValueError(
                "ModelParams.set_model_type: model type has not been provided"
            )
        self.model_type = model_type
        return self

    def set_model(self, N):
        """
        Set the model based on the grid size and system model parameters.

        Parameters:
            system_model_params (SystemModelParams): The system model parameters.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            Exception: If the Grid size is not defined.
        """
        if self.model_type.startswith("DeepCNN"):
            self.model = DeepCNN(N=N,
                                 grid_size=self.grid_size)
        else:
            raise Exception(
                f"ModelGenerator.set_model: Model type {self.model_type} is not defined"
            )
        return self


class DeepCNN(nn.Module):
    """DeepCNN is a convolutional neural network model for DoA  estimation.

    Args:
        N (int): Input dimension size.
        grid_size (int): Size of the output grid.

    Attributes:
        N (int): Input dimension size.
        grid_size (int): Size of the output grid.
        conv1 (nn.Conv2d): Convolutional layer 1.
        conv2 (nn.Conv2d): Convolutional layer 2.
        fc1 (nn.Linear): Fully connected layer 1.
        BatchNorm (nn.BatchNorm2d): Batch normalization layer.
        fc2 (nn.Linear): Fully connected layer 2.
        fc3 (nn.Linear): Fully connected layer 3.
        fc4 (nn.Linear): Fully connected layer 4.
        DropOut (nn.Dropout): Dropout layer.
        Sigmoid (nn.Sigmoid): Sigmoid activation function.
        ReLU (nn.ReLU): Rectified Linear Unit activation function.

    Methods:
        forward(X: torch.Tensor): Performs the forward pass of the DeepCNN model.
    """

    def __init__(self, N, grid_size):
        # input dim (N, T)
        super(DeepCNN, self).__init__()
        self.N = N
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2)
        # self.fc1 = nn.Linear(256 * (round((self.N -2) / 2) - 3) * (round((self.N -2) / 2) -3), 4096)
        self.fc1 = nn.Linear(256 * (self.N - 5) * (self.N - 5), 4096)
        self.BatchNorm = nn.BatchNorm2d(256)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, self.grid_size)
        self.DropOut = nn.Dropout(0.3)
        self.Sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, X):
        # X shape == [Batch size, N, N, 3]
        X = X.view(X.size(0), X.size(3), X.size(
            2), X.size(1))  # [Batch size, 3, N, N]
        ## Architecture flow ##
        # CNN block #1: 3xNxN-->256x(N-2)x(N-2)
        X = self.conv1(X)
        X = self.ReLU(X)
        # CNN block #2: 256x(N-2)x(N-2)-->256x(N-3)x(N-3)
        X = self.conv2(X)
        X = self.ReLU(X)
        # CNN block #3: 256x(N-3)x(N-3)-->256x(N-4)x(N-4)
        X = self.conv2(X)
        X = self.ReLU(X)
        # CNN block #4: 256x(N-4)x(N-4)-->256x(N-5)x(N-5)
        X = self.conv2(X)
        X = self.ReLU(X)
        # FC BLOCK
        # Reshape Output shape: [Batch size, 256 * (self.N - 5) * (self.N - 5)]
        X = X.view(X.size(0), -1)
        X = self.DropOut(self.ReLU(self.fc1(X)))  # [Batch size, 4096]
        X = self.DropOut(self.ReLU(self.fc2(X)))  # [Batch size, 2048]
        X = self.DropOut(self.ReLU(self.fc3(X)))  # [Batch size, 1024]
        X = self.fc4(X)  # [Batch size, grid_size]
        X = self.Sigmoid(X)
        return X


class DeepCNNMUSIC(DeepCNN):
    def __init__(self, N, grid_size):
        # input dim (N, T)
        super(DeepCNN, self).__init__(N, grid_size)

    def forward(self, X):
        # X shape == [Batch size, N, N, 3]
        X = X.view(X.size(0), X.size(3), X.size(
            2), X.size(1))  # [Batch size, 3, N, N]
        ## Architecture flow ##
        # CNN block #1: 3xNxN-->256x(N-2)x(N-2)
        X = self.conv1(X)
        X = self.ReLU(X)
        # CNN block #2: 256x(N-2)x(N-2)-->256x(N-3)x(N-3)
        X = self.conv2(X)
        X = self.ReLU(X)
        # CNN block #3: 256x(N-3)x(N-3)-->256x(N-4)x(N-4)
        X = self.conv2(X)
        X = self.ReLU(X)
        # CNN block #4: 256x(N-4)x(N-4)-->256x(N-5)x(N-5)
        X = self.conv2(X)
        X = self.ReLU(X)
        # FC BLOCK
        # Reshape Output shape: [Batch size, 256 * (self.N - 5) * (self.N - 5)]
        X = X.view(X.size(0), -1)
        X = self.DropOut(self.ReLU(self.fc1(X)))  # [Batch size, 4096]
        X = self.DropOut(self.ReLU(self.fc2(X)))  # [Batch size, 2048]
        X = self.DropOut(self.ReLU(self.fc3(X)))  # [Batch size, 1024]
        X = self.fc4(X)  # [Batch size, grid_size]
        X = self.Sigmoid(X)
        return X

if __name__ == "__main__":
    model = DeepCNN(N=16, grid_size=121)
    summary(model, (16, 16, 3))
    model = DeepCNN(N=9, grid_size=121)
    summary(model, (9, 9, 3))
    