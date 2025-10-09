import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.utility_functions import get_features

primary_dataset = pd.read_csv("NCAA Data\DEV _ March Madness.csv")

#Dataset will be made up of these columns.
primary_dataset = primary_dataset[["Season", "Short Conference Name", "Adjusted Tempo",
        "Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
        "eFGPct", "TOPct", "ORPct", "FTRate", "OffFT", "DefFT", "Mapped Conference Name", "Mapped ESPN Team Name",
        "Full Team Name", "Seed", "Region", "Post-Season Tournament"]]


primary_dataset["Tournament Target"] = (primary_dataset["Post-Season Tournament"] == "March Madness").astype(int)
primary_dataset["Efficiency_Ratio"] = primary_dataset["Adjusted Offensive Efficiency"] / primary_dataset["Adjusted Defensive Efficiency"]

features = get_features()

scale = MinMaxScaler()
primary_dataset[features] = scale.fit_transform(primary_dataset[features])

validation_dataset = primary_dataset[(primary_dataset["Season"] == 2025) & (primary_dataset["Tournament Target"] == 1)]
primary_dataset = primary_dataset[(primary_dataset["Season"] <= 2024)]

primary_dataset.to_csv("Datasets/primary/primary_dataset.csv")
validation_dataset.to_csv("Datasets/validation/validation_dataset.csv")



