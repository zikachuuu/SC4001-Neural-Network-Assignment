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
    

##############################################################################


def preprocess(df):
    """From Part A1"""
    
    X_train, y_train, X_test, y_test = split_dataset(
        df              = df,
        columns_to_drop = ['filename', 'label'],    # Columns to drop for training set
        test_size       = 0.25,
        random_state    = 42
    )
    X_train_scaled, X_test_scaled = preprocess_dataset(X_train, X_test)
    return X_train_scaled, y_train, X_test_scaled, y_test


class CustomDataset(Dataset):
    """From Part A1"""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def intialise_loaders(X_train_scaled, y_train, X_test_scaled, y_test, batch_size=64):
    """From Part A1"""

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


class MLP(nn.Module):

    def __init__(
            self                    , 
            num_features    = 128   , 
            num_hidden      = 256   , 
            num_labels      = 1     ,
            dropout         = 0.3
        ):
        super().__init__()
        self.mlp_stack = nn.Sequential(
            nn.Linear(num_features, num_hidden),    # Creates a fully connected layer that performs U = W*X + B
            nn.ReLU(),                              # Apply ReLU activation function Y = f(U)
            nn.Dropout (dropout),                   # Apply Dropout with probability of dropout

            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout (dropout),

            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout (dropout),

            nn.Linear(num_hidden, num_labels),
            nn.Sigmoid()                            
        )

    def forward(self, x):                
        return self.mlp_stack(x)
    

def train_one_epoch(
        model       , 
        dataloader  , 
        loss_fn     , 
        optimizer
    ):
    """Train for one epoch and return average loss and accuracy"""
    model.train()  # Enable dropout

    epoch_train_loss= 0     # Sum of losses (binary cross entropy) across all batches (one epoch)
    epoch_correct   = 0     # Total number of correct predictions across all batches (one epoch)
    
    size            = len(dataloader.dataset)     # Total number of input patterns
    num_batches     = len(dataloader)             # Number of batches
    
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
    
    with torch.no_grad():   # stop PyTouchh from calculating gradients during evaluation (saves memory and computations)
        
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