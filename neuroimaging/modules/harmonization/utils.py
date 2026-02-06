import pandas as pd
from io import StringIO

PALETTE = [
    "#000000",
    "#de3838",
    "#5ecbf3",
    "#6ce66c",
    "#cc971b",
    "#9400D3",
    "#a5bc0e",
    "#39618f",
    "#31984d",
    "#b75c50",
    "#c08f07",
    "#5f6ca7",
    "#3CB371",
    "#FF5640",
    "#FFC640",
    "#4965D6",
    "#00FF7F",
    "#FF8373",
    "#FFD573",
    "#6F83D6",
    "#64DF85",
    "#FF5600",
    "#FF7C00",
    "#04859D",
    "#00AA72",
    "#BF4030",
    "#BF6030",
    "#5F9EA0",
    "#20B2AA",
    "#193dcc",
    "#A63800",
    "#A65100",
    "#015666",
    "#006E4A",
    "#FF8040",
    "#37B6CE",
    "#556B2F",
    "#FFB773",
    "#6A5ACD",
] * 10


def to_plotly_rgb(rgb_values):
    """From a list of 3 integers representing RGB values, return a Plotly-compatible RGB string."""
    return f"rgb({rgb_values[0]}, {rgb_values[1]}, {rgb_values[2]})"


def load_sites_data(files) -> pd.DataFrame:
    """Parse a TSV file and return its contents as a dictionary."""
    dfs = []
    for f in files:
        df = pd.read_csv(StringIO(f.get("f", "")), sep="\t")
        dfs.append(df)
    if len(dfs) > 1:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = dfs[0]

    return df
