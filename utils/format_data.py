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

features = get_features()

scale = MinMaxScaler()
primary_dataset[features] = scale.fit_transform(primary_dataset[features])

primary_dataset.to_csv("Datasets/primary/primary_dataset.csv")

validation_dataset = primary_dataset[(primary_dataset["Season"] == 2025) & (primary_dataset["Tournament Target"] == 1)]
validation_dataset.to_csv("Datasets/validation/validation_dataset.csv")



