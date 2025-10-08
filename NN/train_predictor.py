from utils.utility_functions import load_data_as_tensor, get_dataset_loader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import optuna
from .BracketPredictionModel import BracketPredictor



def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return running_loss / total, correct / total


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model and return (loss, accuracy, AUROC)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, Y_batch)
            total_loss += loss.item() * X_batch.size(0)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            labels = Y_batch.cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels)
            correct += (preds == labels).sum()
            total += len(labels)

    avg_loss = total_loss / total
    accuracy = correct / total
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.0

    return avg_loss, accuracy, auroc


def save_best_model(model, params, path, input_size, hidden_layers):
    """Save model weights and configuration."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_layers": hidden_layers,
        "params": params
    }, path)


def load_best_model(path, device):
    """Load model and metadata from file."""
    checkpoint = torch.load(path, map_location=device)
    model = BracketPredictor(
        input_size=checkpoint["input_size"],
        hidden_layers=checkpoint["hidden_layers"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["params"]


def objective(trial):
    lr = trial.suggest_float("lr", 0.00001, 1.0, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0000001, 0.1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    epochs = trial.suggest_int("epochs", 20, 30)

    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_layers = [trial.suggest_int(f"n_units_l{i}", 8, 256, log=True) for i in range(n_layers)]
    input_size = X_train.shape[1]

    model = BracketPredictor(input_size, hidden_layers).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = get_dataset_loader(X_train, Y_train, batch_size)
    test_loader = get_dataset_loader(X_test, Y_test, batch_size)
    best_auroc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, auroc = evaluate_model(model, test_loader, criterion, device)

        trial.report(auroc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if auroc > best_auroc:

            best_auroc = auroc
            path = "NN/Trained Models/best_model.pt"
            params={"lr": lr, "weight_decay": weight_decay, "batch_size": batch_size, "epochs": epochs}

            save_best_model(model, params, path, input_size, hidden_layers)


        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"[Trial {trial.number}] Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | AUROC: {auroc:.4f}")

        

    return best_auroc


def evaluate_final_model(X_test, Y_test, device):
    print("\n--- Evaluating Best Saved Model ---")
    model, params = load_best_model("NN/Trained Models/best_model.pt", device)

    test_loader = get_dataset_loader(X_test, Y_test, batch_size=params["batch_size"])

    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_acc, auroc = evaluate_model(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUROC: {auroc:.4f}")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test, Y_train, Y_test = load_data_as_tensor("Datasets/primary/primary_dataset.csv")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2, timeout=None)

    print("\nBest Hyperparameters Found:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    print(f"\nBest AUROC Score: {study.best_value:.4f}")

    evaluate_final_model(X_test, Y_test, device)
