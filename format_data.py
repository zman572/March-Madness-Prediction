import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("NCAA Data\DEV _ March Madness.csv")

#Dataset will be made up of these columns.
df = df[["Season", "Short Conference Name", "Adjusted Tempo",
        "Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
        "eFGPct", "TOPct", "ORPct", "FTRate", "OffFT", "DefFT", "Mapped Conference Name", "Mapped ESPN Team Name",
        "Full Team Name", "Seed", "Region", "Post-Season Tournament"]]



#Set teams that were in march madness to 1, otherwise set to 0.

'''df = df[df["Season"] == 2025]
df["Efficiency_Ratio"] = df["Adjusted Offensive Efficiency"] / df["Adjusted Defensive Efficiency"]

# Define major conferences (this can be controversial lol)
big_conferences = ["ACC", "SEC", "B12"]
median_eff_ratio = df["Efficiency_Ratio"].median()

def compute_upset_propensity(row):
    if row["Short Conference Name"] not in big_conferences and row["Efficiency_Ratio"] > median_eff_ratio:
        return 1
    else:
        return 0

df["Upset_Propensity"] = df.apply(compute_upset_propensity, axis=1)'''

df["Tournament Target"] = (df["Post-Season Tournament"] == "March Madness").astype(int)

features = ["Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
                "eFGPct", "TOPct", "Adjusted Tempo"]


scale = MinMaxScaler()
df[features] = scale.fit_transform(df[features])

df.to_csv("dataset.csv")

this_season = df[(df["Season"] == 2025) & (df["Tournament Target"] == 1)]
this_season.to_csv("validation_dataset.csv")

