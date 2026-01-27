import pandas as pd
from io import StringIO

def load_sites_data(files) -> pd.DataFrame:
    """Parse a TSV file and return its contents as a dictionary."""
    dfs = []
    for f in files:
        df = pd.read_csv(StringIO(f.get("f", "")), sep="\t")
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df
