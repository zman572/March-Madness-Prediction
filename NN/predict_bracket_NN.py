import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from utils.utility_functions import get_features, load_model_from_pkl


def nested_dict():
    return defaultdict(list)


def readMatchups(matchup_path):
    """
    Creates a nested dict where each key is the region.
    Each inner dict uses rounds as the key (for now only round 1).
    Each round contains a list of the matchups for that round.
    """
    matches = defaultdict(nested_dict)
    df = pd.read_csv(matchup_path)

    for _, row in df.iterrows():
        matchup = row["Matchup"]
        region = row["Region"]
        teams = [team.strip() for team in matchup.split("vs")]
        matches[region]["round_1"].append(teams)

    return matches


def run_matchup(teamA, teamB, dataset, model, features, device):
    """
    Uses the trained NN model to predict a matchup given two teams and their features.
    """

    # Get team data
    teamA_data = dataset[dataset["Mapped ESPN Team Name"] == teamA][features].values
    teamB_data = dataset[dataset["Mapped ESPN Team Name"] == teamB][features].values

    if teamA_data.size == 0 or teamB_data.size == 0:
        raise ValueError(f"Missing data for one of the teams: {teamA}, {teamB}")

    # Convert to tensors
    teamA_tensor = torch.tensor(teamA_data, dtype=torch.float32).to(device)
    teamB_tensor = torch.tensor(teamB_data, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        teamA_prob = torch.sigmoid(model(teamA_tensor)).squeeze().mean().item()
        teamB_prob = torch.sigmoid(model(teamB_tensor)).squeeze().mean().item()

    # Compute win probability
    teamA_win_prob = teamA_prob / (teamA_prob + teamB_prob)
    winner = teamA if teamA_win_prob >= 0.5 else teamB

    return winner


def predict_bracket(regions, dataset, model, device):
    """
    Predicts the entire bracket for each round of each region.
    Once each region has a champion, it predicts the matchups in
    the Final Four and National Championship, then determines the National Champion.
    """

    features = get_features()
    region_brackets = {}
    region_champions = {}

    # Predict each region
    for region, rounds in regions.items():
        bracket = defaultdict(list)
        bracket["round_1"] = rounds["round_1"]
        r = 2
        matchups = bracket["round_1"]
        next_match = []

        while len(matchups) > 0:
            for match in matchups:
                teamA, teamB = match
                winner = run_matchup(teamA, teamB, dataset, model, features, device)
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

    # Final Four
    final_four = [
        [region_champions["South"], region_champions["West"]],
        [region_champions["East"], region_champions["Midwest"]],
    ]

    # National Championship
    national_championship = [
        run_matchup(region_champions["South"], region_champions["West"], dataset, model, features, device),
        run_matchup(region_champions["East"], region_champions["Midwest"], dataset, model, features, device),
    ]

    # National Champion
    national_champion = run_matchup(
        national_championship[0], national_championship[1], dataset, model, features, device
    )

    region_brackets["Final_Four"] = final_four
    region_brackets["National_Championship"] = national_championship
    region_brackets["National_Champion"] = national_champion

    return region_brackets


if __name__ == "__main__":
    matchup_path = "Matchups/init_matches_round1.csv"
    dataset_path = "Datasets/validation/validation_dataset.csv"
    model_path = "NN/Trained Models/best_model.pkl"  # adjust if necessary

    # Load matchups and dataset
    init_matchups = readMatchups(matchup_path)
    dataset = pd.read_csv(dataset_path)

    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, params = load_model_from_pkl(model_path, device)

    # Predict entire bracket
    bracket = predict_bracket(init_matchups, dataset, model, device)
    print(bracket["South"]["round_2"])
    print(bracket["South"]["round_3"])
    print(bracket["South"]["round_4"])
    print(bracket["South"]["champion"])

    print("\n",bracket["East"]["round_2"])
    print(bracket["East"]["round_3"])
    print(bracket["East"]["round_4"])
    print(bracket["East"]["champion"])

    print("\n",bracket["West"]["round_2"])
    print(bracket["West"]["round_3"])
    print(bracket["West"]["round_4"])
    print(bracket["West"]["champion"])

    print("\n",bracket["Midwest"]["round_2"])
    print(bracket["Midwest"]["round_3"])
    print(bracket["Midwest"]["round_4"])
    print(bracket["Midwest"]["champion"])

    print("\n",bracket["Final_Four"])
    print(bracket["National_Championship"])
    print(bracket["National_Champion"])
