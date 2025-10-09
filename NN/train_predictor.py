from utils.utility_functions import load_data_as_tensor, get_dataset_loader, load_model_from_pkl, save_model_as_pkl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import optuna
from .BracketPredictionModel import BracketPredictor


def train_model(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
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




def objective(trial):
    """Optuna objective: return best AUROC for this trial."""
    lr = trial.suggest_float("lr", 0.00001, 1.0, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0000001, 0.1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    epochs = trial.suggest_int("epochs", 20, 300)

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

        if (epoch + 1) % 5 == 0:
            print(f"[Trial {trial.number}] Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | AUROC: {auroc:.4f}")

        best_auroc = max(best_auroc, auroc)

    trial.set_user_attr("hidden_layers", hidden_layers)
    trial.set_user_attr("input_size", input_size)
    return best_auroc


def train_final_model(X_train, Y_train, best_params, input_size, hidden_layers, device):
    """Train the final model using the best hyperparameters."""
    print("\n--- Training Best Model from Scratch ---")

    model = BracketPredictor(input_size, hidden_layers).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    train_loader = get_dataset_loader(X_train, Y_train, best_params["batch_size"])

    for epoch in range(best_params["epochs"]):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        if (epoch + 1) % 5 == 0:
            print(f"Final Model Epoch {epoch+1}/{best_params['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    return model


def evaluate_final_model(X_test, Y_test, device, model_path="NN/Trained Models/best_model.pkl"):
    """Evaluate the final saved model."""
    print("\n--- Evaluating Best Saved Model ---")
    model, params = load_model_from_pkl(model_path, device)
    test_loader = get_dataset_loader(X_test, Y_test, batch_size=params["batch_size"])
    criterion = nn.BCEWithLogitsLoss()

    test_loss, test_acc, auroc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUROC: {auroc:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    X_train, X_test, Y_train, Y_test = load_data_as_tensor("Datasets/primary/primary_dataset.csv")

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=None)

    # Print best results
    print("\nBest Hyperparameters Found:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
    print(f"\nBest AUROC Score: {study.best_value:.4f}")

    # Train and save final model
    best_params = study.best_params
    input_size = X_train.shape[1]
    hidden_layers = study.best_trial.user_attrs["hidden_layers"]
    model_path = "NN/Trained Models/best_model.pkl"

    best_model = train_final_model(X_train, Y_train, best_params, input_size, hidden_layers, device)
    save_model_as_pkl(best_model, best_params, model_path, input_size, hidden_layers)
    print(f"\nBest model saved to {model_path}")

    # Evaluate saved model
    evaluate_final_model(X_test, Y_test, device, model_path)
