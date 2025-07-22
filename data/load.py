import pandas as pd

def load_data(csv_path='adjuster_dataset_gsoc_v1.csv'):
    """
    Loads the dataset from a CSV file.

    Parameters:
    ----------
    csv_path : str
        Path to the input CSV file.

    Returns:
    -------
    pd.DataFrame
        Loaded data as a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df