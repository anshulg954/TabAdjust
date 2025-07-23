import pandas as pd
from IPython.display import display
import sys

def in_jupyter():
    return 'ipykernel' in sys.modules

def display_diagnostics(df: pd.DataFrame, caption: str):
    """
    Displays a styled preview and summary of the DataFrame in Jupyter;
    falls back to print if run in scripts.
    """
    print(f"************* {caption.upper()} *********")
    if in_jupyter():
        _ = display(df.head(10).style.set_caption(f"{caption}").set_table_styles([
            {"selector": "caption", "props": [("font-size", "16px"), ("font-weight", "bold")]}
        ]))
    else:
        print(df.head(10))

    # print(f"************* {caption.upper()} SUMMARY *********")
    # if in_jupyter():
    #     _ = display(df.describe(include='all').style.set_caption(f"{caption} Summary").set_table_styles([
    #         {"selector": "caption", "props": [("font-size", "16px"), ("font-weight", "bold")]}
    #     ]))
    # else:
    #     print(df.describe(include='all'))
