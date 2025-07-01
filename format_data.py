import pandas as pd

df = pd.read_csv("NCAA Data\DEV _ March Madness.csv")
df = df[["Season", "Mapped ESPN Team Name", "Short Conference Name", "Adjusted Tempo", "Adjusted Offensive Efficiency", "eFGPct", "TOPct", "FTRate","OffFT", 
         "DefFT", "Adjusted Defensive Efficiency", "Post-Season Tournament"]]
df["Post-Season Tournament"] = (df["Post-Season Tournament"] == "March Madness").astype(int)
df.to_csv("dataset.csv")

