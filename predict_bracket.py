import pandas as pd
import pickle
from collections import defaultdict

def nested_dict():
    return defaultdict(list)

def readMatchups(matchup_path):

    '''Crates a nested dict Where each key is the reagion. Each inner dict uses rounds as the key 
    (for now only round 1). Each round contains a list of the matchups for that round'''

    matches = defaultdict(nested_dict)

    # Read CSV with pandas
    df = pd.read_csv(matchup_path)

    # Iterate over DataFrame rows
    for _, row in df.iterrows():
        matchup = row["Matchup"]
        region = row["Region"]

        # Split into [team1, team2]
        teams = [team.strip() for team in matchup.split("vs")]

        # Append to nested dict
        matches[region]["round_1"].append(teams)

    return matches

def run_matchup(teamA, teamB, dataset):

    '''Uses the trained log reg model to predict a matchup given two teams and their features'''

    with open("Trained Models/trained_log_reg.pkl", "rb") as f:
        log_reg_model = pickle.load(f)

    features = ["Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
                "eFGPct", "TOPct", "Adjusted Tempo"]
    
    
    teamA_data = dataset[dataset["Mapped ESPN Team Name"] == teamA]
    teamA_data = teamA_data[features].values
    
    teamB_data = dataset[dataset["Mapped ESPN Team Name"] == teamB]
    teamB_data = teamB_data[features].values
    
    teamA_prob = log_reg_model.predict_proba(teamA_data)[0][1]
    teamB_prob = log_reg_model.predict_proba(teamB_data)[0][1]

    teamA_win_prob = teamA_prob / (teamA_prob + teamB_prob)
    #teamB_win_prob = 1 - teamA_win_prob

    winner = teamA if teamA_win_prob >= 0.5 else teamB

    return winner


def predict_bracket(regions, dataset):

    '''Predicts the entire bracket for each round of each region. Once 
    each region has a champion it predicts the matchups in the final_four and the national championship and then finally predicts the national champion.'''
  
    region_brackets = {}
    region_champions = {}

    for region, rounds in regions.items():
        bracket = defaultdict(list)
        bracket["round_1"] = rounds["round_1"]
        r = 2
        matchups = bracket["round_1"]
        next_match = []

        while len(matchups) > 0:
            for match in matchups:
                teamA, teamB = match
                winner = run_matchup(teamA, teamB, dataset)
                next_match.append(winner)

                if len(next_match) == 2:
                    bracket[f"round_{r}"].append(next_match)
                    matchups = bracket[f"round_{r}"]
                    next_match = []

            r += 1
          
            if r > 5:
                if next_match:
                    bracket["champion"].append(next_match[0])
                    region_champions[region] = next_match[0]
                break
        
        region_brackets[region] = bracket


    final_four = []
    final_four.append([region_champions["South"], region_champions["West"]])
    final_four.append([region_champions["East"], region_champions["Midwest"]])

    national_championship = []
    national_championship.append(run_matchup(region_champions["South"], region_champions["West"], dataset))
    national_championship.append(run_matchup(region_champions["East"], region_champions["Midwest"], dataset))

    national_champion = run_matchup(national_championship[0], national_championship[1], dataset)

    region_brackets["Final_Four"] = final_four
    region_brackets["National_Championship"] = national_championship
    region_brackets["National_Champion"] = national_champion

    return region_brackets
    

        
if __name__ == "__main__":

    matchup_path = "Matchups/init_matches_round1.csv"
    init_matchups = readMatchups(matchup_path) #Read in the intial round one matchups from CSV
    dataset = pd.read_csv("Datasets/validation/validation_dataset.csv") #Get validation data to be used as model features

    bracket = predict_bracket(init_matchups, dataset) #Create the complete bracket

    





    

    

    




