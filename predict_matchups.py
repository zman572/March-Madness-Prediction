import pandas as pd
import pickle


def run_matchup(teamA, teamB, val_dataset):

    with open("Trained Models/trained_log_reg.pkl", "rb") as f:
        log_reg_model = pickle.load(f)

    teamA_data = val_dataset[val_dataset["Mapped ESPN Team Name"] == teamA]
    teamA_data = teamA_data[["Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency", "Adjusted Tempo"]].values[0].reshape(1, -1)
    
    teamB_data = val_dataset[val_dataset["Mapped ESPN Team Name"] == teamB]
    teamB_data = teamB_data[["Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency", "Adjusted Tempo"]].values[0].reshape(1, -1)
    
    teamA_prob = log_reg_model.predict_proba(teamA_data)
    teamB_prob = log_reg_model.predict_proba(teamB_data)

    print(teamA_prob)
    print(teamB_prob)

    '''teamA_win_prob = teamA_prob / (teamA_prob + teamB_prob)
    teamB_win_prob = 1 - teamA_win_prob

    print(f"\n Probability {teamA} wins: {(teamA_win_prob * 100):.4f}%")
    print(f"Probablility {teamB} wins: {(teamB_win_prob * 100):.4f}%")

    winner = teamA if teamA_win_prob >= 0.5 else teamB

    print(f"\nThe predicted winner is: {winner}")'''








if __name__ == "__main__":

    val_dataset = pd.read_csv("validation_dataset.csv")

    teamA = input("\nEnter the first team: ")
    teamB = input("\nEnter the second team: ")
    
    

    run_matchup(teamA, teamB, val_dataset)




    
