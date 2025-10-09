from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from NN.BracketPredictionModel import BracketPredictor
import pickle

def get_features():
    
    features = ["Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
        "eFGPct", "TOPct", "Adjusted Tempo", "Efficiency_Ratio"]
    
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


def save_model_as_pkl(model, params, path, input_size, hidden_layers):
    """Save model and configuration as a .pkl file."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_layers": hidden_layers,
        "params": params
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_model_from_pkl(path, device):
    """Load model and metadata from a .pkl file."""
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    model = BracketPredictor(
        input_size=checkpoint["input_size"],
        hidden_layers=checkpoint["hidden_layers"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["params"]


