import json
import pandas as pd

df = pd.read_csv("data/creditcard.csv").drop_duplicates()
fraud = df[df["Class"] == 1].drop(columns="Class").head(5).to_dict(orient="records")
normal = df[df["Class"] == 0].drop(columns="Class").head(5).to_dict(orient="records")

with open("samples.json", "w") as f:
    json.dump({"fraud": fraud, "normal": normal}, f)
print("saved samples.json")