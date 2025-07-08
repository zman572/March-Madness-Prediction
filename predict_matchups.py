import pandas as pd
import pickle


def run_matchup(teamA, teamB, val_dataset):

    with open("Trained Models/trained_log_reg.pkl", "rb") as f:
        log_reg_model = pickle.load(f)

    features = ["Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
                "eFGPct", "TOPct", "Adjusted Tempo"]
    
    teamA_data = val_dataset[val_dataset["Mapped ESPN Team Name"] == teamA]
    teamA_data = teamA_data[features].values
    
    teamB_data = val_dataset[val_dataset["Mapped ESPN Team Name"] == teamB]
    teamB_data = teamB_data[features].values
    
    teamA_prob = log_reg_model.predict_proba(teamA_data)[0][1]
    teamB_prob = log_reg_model.predict_proba(teamB_data)[0][1]

    teamA_win_prob = teamA_prob / (teamA_prob + teamB_prob)
    teamB_win_prob = 1 - teamA_win_prob

    print(f"\n Probability {teamA} wins: {(teamA_win_prob * 100):.4f}%")
    print(f"Probablility {teamB} wins: {(teamB_win_prob * 100):.4f}%")

    winner = teamA if teamA_win_prob >= 0.5 else teamB

    print(f"\nThe predicted winner is: {winner}")








if __name__ == "__main__":

    val_dataset = pd.read_csv("validation_dataset.csv")

    teamA = input("\nEnter the first team: ")
    teamB = input("\nEnter the second team: ")
    
    

    run_matchup(teamA, teamB, val_dataset)




    
