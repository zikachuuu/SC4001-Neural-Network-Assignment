### THIS FILE CONTAINS COMMON FUNCTIONS, CLASSSES

import tqdm
import time
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy.io import wavfile as wav

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# =================================
# Provided functions & classes
# =================================

def split_dataset(df, columns_to_drop, test_size, random_state):
    label_encoder = preprocessing.LabelEncoder()

    df['label'] = label_encoder.fit_transform(df['label'])

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    df_train2 = df_train.drop(columns_to_drop,axis=1)
    y_train2 = df_train['label'].to_numpy()

    df_test2 = df_test.drop(columns_to_drop,axis=1)
    y_test2 = df_test['label'].to_numpy() 

    return df_train2, y_train2, df_test2, y_test2

def preprocess_dataset(df_train, df_test):

    standard_scaler = preprocessing.StandardScaler()
    df_train_scaled = standard_scaler.fit_transform(df_train)

    df_test_scaled = standard_scaler.transform(df_test)

    return df_train_scaled, df_test_scaled

def set_seed(seed = 0):
    '''
    set random seed
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



# ================================
# Copied from Question A1
# ================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) class that defines the architecture of the MLP model.
    """

    def __init__(
            self                            , 
            num_features        : int       ,   # Num of input features (dimensionality of input data)
            num_hidden_layers   : int       ,   # Num of hidden layers (i.e. depth) in the MLP (excluding the output layer)
            hidden_widths       : list[int] ,   # List of widths for each hidden layer (length of list should be equal to num_hidden_layers)
            num_labels          : int       ,   # Num of output labels (dimensionality of output)
            dropout             : float         # Dropout probability to apply after each hidden layer (between 0 and 1)
        ):
        """
        Initializes the MLP model architecture based on the specified parameters.
        Args:
            num_features: Number of input features (dimensionality of input data).
            num_hidden_layers: Number of hidden layers (i.e. depth) in the MLP (excluding the output layer).
            hidden_widths: List of widths for each hidden layer (length of list should be equal to num_hidden_layers).
            num_labels: Number of output labels (dimensionality of output).
            dropout: Dropout probability to apply after each hidden layer (between 0 and 1).
        """

        super().__init__()

        if num_hidden_layers != len(hidden_widths):
            raise ValueError("Length of hidden_widths list must be equal to num_hidden_layers")

        # Input layer + first hidden layer
        layers = [
            nn.Linear(num_features, hidden_widths[0]),  # Creates a fully connected layer that performs U = W*X + B
            nn.ReLU(),                                  # Apply ReLU activation function Y = f(U)
            nn.Dropout(dropout),                        # Apply Dropout with probability of dropout
        ]

        # Additional hidden layers (if num_hidden_layers > 1)
        for i in range(0, num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_widths[i], hidden_widths[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        # Output layer
        layers.extend([
            nn.Linear(hidden_widths[-1], num_labels),
            nn.Sigmoid(),
        ])

        self.mlp_stack = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        """
        Defines the forward pass of the MLP model, which takes input data x and produces output predictions.
        Args:
            x: Input data tensor of shape (batch_size, num_features).
        Returns:
            torch.Tensor: Output predictions tensor of shape (batch_size, num_labels) after passing through the MLP architecture.
        """                
        return self.mlp_stack(x)


def preprocess(df):
    """
    Process the input DataFrame by splitting it into training and testing sets, 
    and applying preprocessing steps such as feature scaling.
    """
    X_train, y_train, X_test, y_test = split_dataset(
        df              = df,
        columns_to_drop = ['filename', 'label'],    # Columns to drop for training set
        test_size       = 0.25,
        random_state    = 42
    )
    X_train_scaled, X_test_scaled = preprocess_dataset(X_train, X_test)
    return X_train_scaled, y_train, X_test_scaled, y_test


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset class that takes in features X and labels y (dataframes), 
    converts them to PyTorch tensors, and implements the necessary methods for data loading.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def intialise_loaders(X_train_scaled, y_train, X_test_scaled, y_test, batch_size):
    """
    Initializes PyTorch DataLoaders for the training and testing datasets.
    Args:
        X_train_scaled: Scaled training features.
        y_train: Training labels.
        X_test_scaled: Scaled testing features.
        y_test: Testing labels.
        batch_size: Batch size to use for the DataLoaders.
    Returns:
        train_dataloader: DataLoader for the training dataset.
        test_dataloader: DataLoader for the testing dataset.
    """
    train_dataset   : CustomDataset = CustomDataset(X_train_scaled, y_train)
    test_dataset    : CustomDataset = CustomDataset(X_test_scaled, y_test)

    train_dataloader: DataLoader    = DataLoader(
        dataset     = train_dataset, 
        batch_size  = batch_size,
        shuffle     = True  # Shuffle the training data at the beginning of each epoch
    )
    test_dataloader : DataLoader    = DataLoader(
        dataset     = test_dataset, 
        batch_size  = batch_size,
        shuffle     = False # No need to shuffle test data (Of course global metrics like MSE will be the same regardless, but good practise for consistency)
    )

    return train_dataloader, test_dataloader


def train_one_epoch(
        model       , 
        dataloader  , 
        loss_fn     , 
        optimizer
    ):
    """Train for one epoch and return average loss and accuracy"""
    model.train()               # Enable dropout

    epoch_train_loss    = 0     # Sum of losses (binary cross entropy) across all batches (one epoch)
    epoch_correct       = 0     # Total number of correct predictions across all batches (one epoch)
    
    size                = len(dataloader.dataset)     # Total number of input patterns
    num_batches         = len(dataloader)             # Number of batches
    
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        batch_size = X_batch.size(0)              # Number of samples in the current batch

        # Forward pass
        predictions = model(X_batch)
        loss        = loss_fn(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        epoch_train_loss += loss.item() * batch_size                # Add the loss of the current batch to the total loss for the epoch
                                                                    # loss.item() gives the average loss per sample in the batch, so we multiply by batch_size to get the total loss for the batch and accumulate it for the epoch

        predicted_labels = (predictions > 0.5).float()              # Convert probabilities to binary predictions (0 or 1) using a threshold of 0.5
        epoch_correct += (predicted_labels == y_batch).sum().item() # Count the number of correct predictions in the batch and add to total_correct
    
    epoch_avg_train_loss    = epoch_train_loss / size   # Average loss per sample for the epoch
    epoch_accuracy          = epoch_correct / size      # Percentage of samples correctly classified in the epoch
    
    return epoch_avg_train_loss, epoch_accuracy


def evaluate(
        model       , 
        dataloader  , 
        loss_fn
    ):
    """Evaluate model and return average loss and accuracy"""
    model.eval()  # Disable dropout

    test_loss   = 0
    correct     = 0

    size            = len(dataloader.dataset)
    num_batches     = len(dataloader)
    
    with torch.no_grad():   # stop PyTouch from calculating gradients during evaluation (saves memory and computations)
        
        for X_batch, y_batch in dataloader:
            batch_size = X_batch.size(0)

            predictions = model(X_batch)
            loss        = loss_fn(predictions, y_batch)

            test_loss += loss.item() * batch_size   

            predicted_labels = (predictions > 0.5).float()
            correct += (predicted_labels == y_batch).sum().item()
    
    avg_test_loss   = test_loss / size
    accuracy        = correct / size
    return avg_test_loss, accuracy