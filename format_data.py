import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("NCAA Data\DEV _ March Madness.csv")

#Dataset will be made up of these columns.
df = df[["Season", "Short Conference Name", "Adjusted Tempo",
        "Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
        "eFGPct", "TOPct", "ORPct", "FTRate", "OffFT", "DefFT", "Mapped Conference Name", "Mapped ESPN Team Name",
        "Full Team Name", "Seed", "Region", "Post-Season Tournament"]]


df["Tournament Target"] = (df["Post-Season Tournament"] == "March Madness").astype(int)

features = ["Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
                "eFGPct", "TOPct", "Adjusted Tempo"]


scale = MinMaxScaler()
df[features] = scale.fit_transform(df[features])

df.to_csv("dataset.csv")

this_season = df[(df["Season"] == 2025) & (df["Tournament Target"] == 1)]
this_season.to_csv("validation_dataset.csv")

