import pandas as pd

df = pd.read_csv("NCAA Data\DEV _ March Madness.csv")
df = df[["Season", "Mapped ESPN Team Name", "Short Conference Name", "Adjusted Tempo", "Adjusted Offensive Efficiency", "Adjusted Defensive Efficiency"]]
df.to_csv("dataset.csv")

