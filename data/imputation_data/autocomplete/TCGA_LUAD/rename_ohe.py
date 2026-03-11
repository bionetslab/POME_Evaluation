from pathlib import Path
import pandas as pd

data_dir = Path("simulated_ohe_data/")

for file_path in data_dir.glob("masked_values_*.tsv"):
    # load TSV
    df = pd.read_csv(file_path, sep="\t", index_col=0)

    # add ID column from original index
    df.insert(0, "ID", df.index)

    # build output path with .csv extension
    out_path = file_path.with_suffix(".csv")

    # save as CSV
    df.to_csv(out_path, sep=",", index=False)

