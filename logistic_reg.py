import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, log_loss
import matplotlib.pyplot as plt
import pickle



def create_train_test_set(dataset, split=0.2, rand=42):

    features = ["Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
                "eFGPct", "TOPct", "Adjusted Tempo"]
    
    #Select Features to be used for training
    X = dataset[features].values
    
    #Target to predict. (Rather or not the team made the post season tournment. Will eith be 0 or 1)
    Y = dataset["Tournament Target"].values

    
    #Split into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split, random_state=rand)

    return X_train, X_test, Y_train, Y_test


def train_log_reg(X_train, X_test, Y_train, iter=500):

    #Train the model
    LR_model = LR(max_iter=iter)
    LR_model.fit(X_train,Y_train)

    #Predict
    pred_probs = LR_model.predict_proba(X_test)[:,1]
    pred = LR_model.predict(X_test)


    with open("Trained Models/trained_log_reg.pkl", "wb") as f:
        pickle.dump(LR_model, f)
    
    return pred, pred_probs, LR_model


def eval_metrics(pred, pred_probs, Y_test):

    #Evaluate Model based on predictions and prediciton probs
    acc = accuracy_score(Y_test, pred)
    roc_sc = roc_auc_score(Y_test, pred_probs)
    fpr, tpr, thresholds = roc_curve(Y_test, pred_probs)
    loss = log_loss(Y_test, pred_probs)
    
    plt.title("Log Reg ROC Curve on Training Data")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr, tpr)
    plt.savefig("Graphs/log_reg_roc.png")
    
    print("Performance metrics:\n")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC Score: {roc_sc:.4f}")
    print(f"Loss {loss:.4f}")



if __name__ =="__main__":
    
    dataset = pd.read_csv("dataset.csv")
    val = pd.read_csv("validation_dataset.csv")

    #Feature selection and split dataset
    X_train, X_test, Y_train, Y_test = create_train_test_set(dataset)

    #Train model and predict
    pred, pred_probs, LR_model = train_log_reg(X_train, X_test, Y_train)

    
    #Evaluate model
    eval_metrics(pred, pred_probs, Y_test)



    



