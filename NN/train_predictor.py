from LogReg.logistic_reg import create_train_test_set
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .BracketPredictionModel import BracketPredictor


from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim

def train_and_evaluate(X_train, Y_train, X_test, Y_test, model, lr=0.01, epochs=100, batch_size=64, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, Y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == Y_batch).sum().item()
            train_total += Y_batch.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                logits = model(X_batch)
                loss = criterion(logits, Y_batch)
                test_loss += loss.item() * X_batch.size(0)

                preds = (torch.sigmoid(logits) > 0.5).float()
                test_correct += (preds == Y_batch).sum().item()
                test_total += Y_batch.size(0)

        avg_test_loss = test_loss / test_total
        test_acc = test_correct / test_total

        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  "
                  f"Train Loss: {avg_train_loss:.4f}  Train Acc: {train_acc:.4f}  "
                  f"Test Loss: {avg_test_loss:.4f}  Test Acc: {test_acc:.4f}")




if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = pd.read_csv("Datasets/dataset.csv")
    X_train, X_test, Y_train, Y_test = create_train_test_set(dataset)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1).to(device)
    Y_test  = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1).to(device)
   
    model = BracketPredictor()
    model.to(device)

    train_and_evaluate(X_train, Y_train, X_test, Y_test, model, lr=0.7, epochs=125, batch_size=16, device=device)
