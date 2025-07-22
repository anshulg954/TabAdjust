from IPython.display import display

def display_diagnostics(df, caption: str):
    """
    Display the first few rows and summary statistics of a DataFrame
    with styled captions for visual inspection in notebooks.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    caption : str
        Caption label to annotate output.
    """
    print(f"************* {caption.upper()} *********")
    display(df.head(10).style.set_caption(f"{caption}").set_table_styles([
        {"selector": "caption", "props": [("font-size", "16px"), ("font-weight", "bold")]}
    ]))

    print(f"************* {caption.upper()} SUMMARY *********")
    display(df.describe(include='all').style.set_caption(f"{caption} Summary").set_table_styles([
        {"selector": "caption", "props": [("font-size", "16px"), ("font-weight", "bold")]}
    ]))
