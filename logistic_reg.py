import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import MinMaxScaler



def create_train_test_set(dataset, split=0.2, rand=35):
    

    X = dataset[["Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency", "eFGPct", "TOPct", "FTRate","OffFT", "DefFT", "Adjusted Tempo"]].values
    Y = dataset[["Post-Season Tournament"]].values.ravel()

    scale = MinMaxScaler()
    scale.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split, random_state=rand, shuffle=True)

    return X_train, X_test, Y_train, Y_test


def run_log_reg(X_train, X_test, Y_train, iter=300):

    LR_model = LR(max_iter=iter)
    LR_model.fit(X_train,Y_train)

    pred_probs = LR_model.predict_proba(X_test)
    pred = LR_model.predict(X_test)

    return pred, pred_probs


def eval_metrics(pred, Y_test):

    acc = accuracy_score(Y_test, pred)
    roc = roc_auc_score(Y_test, pred)
    loss = log_loss(Y_test, pred)

    print("Performance metrics:\n")
    print(f"Accuracy: {acc}")
    print(f"ROC AUC Score: {roc}")
    print(f"Loss {loss}")



if __name__ =="__main__":
    
    dataset = pd.read_csv("dataset.csv")

    X_train, X_test, Y_train, Y_test = create_train_test_set(dataset)
    pred, pred_probs = run_log_reg(X_train, X_test, Y_train)
    eval_metrics(pred, Y_test)



    



