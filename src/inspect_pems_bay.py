import pandas as pd
from pathlib import Path

files = [
    Path("data/raw/pems_bay/speed.csv"),
    Path("data/raw/pems_bay/occupancy.csv")
]

for file in files:
    print("\n" + "=" * 60)
    print("FILE:", file)

    df = pd.read_csv(file, nrows=5)
    print("Sample shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nSample records:")
    print(df.to_string(index=False))

    full = pd.read_csv(file, nrows=0)
    print("\nNumber of columns/sensors:", len(full.columns))
