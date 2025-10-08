from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_features():
    
    features = ["Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
        "eFGPct", "TOPct", "Adjusted Tempo", "ORPct", "FTRate", "OffFT", "DefFT"]
    
    return features
    


def create_train_test_set(dataset, split=0.2, rand=42):

    features = get_features()

    #Select Features to be used for training
    X = dataset[features].values
    
    #Target to predict. (Rather or not the team made the post season tournment. Will eith be 0 or 1)
    Y = dataset["Tournament Target"].values
    
    #Split into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split, random_state=rand)

    return X_train, X_test, Y_train, Y_test


def load_data_as_tensor(path):
    """Load dataset and split into train/test tensors."""
    dataset = pd.read_csv(path)
    X_train, X_test, Y_train, Y_test = create_train_test_set(dataset)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

    return X_train, X_test, Y_train, Y_test


def get_dataset_loader(X_data, Y_data, batch_size):
    """Return DataLoader for train set."""

    loader = DataLoader(TensorDataset(X_data, Y_data), batch_size=batch_size, shuffle=True)
    return loader


