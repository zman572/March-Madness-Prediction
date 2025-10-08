import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, log_loss
from utils.utility_functions import create_train_test_set
import matplotlib.pyplot as plt
import pickle



def train_log_reg(X_train, X_test, Y_train, iter=500):

    #Train the model
    LR_model = LR(max_iter=iter)
    LR_model.fit(X_train,Y_train)

    #Predict
    pred_probs = LR_model.predict_proba(X_test)[:,1]
    pred = LR_model.predict(X_test)


    with open("LogReg/Trained Models/trained_log_reg.pkl", "wb") as f:
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
    plt.savefig("LogReg/Graphs/log_reg_roc.png")
    
    print("Performance metrics:\n")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC Score: {roc_sc:.4f}")
    print(f"Loss {loss:.4f}")



if __name__ =="__main__":
    
    primary_dataset = pd.read_csv("Datasets/primary/primary_dataset.csv")
    
    #Feature selection and split dataset
    X_train, X_test, Y_train, Y_test = create_train_test_set(primary_dataset)

    #Train model and predict
    pred, pred_probs, LR_model = train_log_reg(X_train, X_test, Y_train)

    
    #Evaluate model
    eval_metrics(pred, pred_probs, Y_test)



    



