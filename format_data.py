import pandas as pd

df = pd.read_csv("NCAA Data\DEV _ March Madness.csv")

#Dataset will be made up of these columns.
df = df[["Season", "Short Conference Name", "Adjusted Tempo",
        "Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency",
        "eFGPct", "TOPct", "ORPct", "FTRate", "OffFT", "DefFT", "Mapped Conference Name", "Mapped ESPN Team Name",
        "Full Team Name", "Seed", "Region", "Post-Season Tournament"]]

#Set teams that were not in the post season tournament to seed -1.
df.loc[df["Seed"] == "Not In a Post-Season Tournament", "Seed"] = -1

#Set teams that were in march madness to 1, otherwise set to 0.
df["Post-Season Tournament"] = (df["Post-Season Tournament"] == "March Madness").astype(int)

df = df[df["Season"] != 2025]

df.to_csv("dataset.csv")

'''this_season = df[(df["Season"] == 2025) & (df["Post-Season Tournament"] == 1)]
this_season.to_csv("validation_dataset.csv")'''

