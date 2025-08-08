"""
Dataset loading and management module.

This module provides access to various sample datasets including:
- Titanic passenger data
- Iris flower measurements
- Restaurant tips
- Diamond prices
- Retail sales data
- And more

The main function is `load_dataset()` which loads datasets by name.
"""

from pathlib import Path
from typing import Union

import pandas as pd


def load_dataset(name: str) -> Union[pd.DataFrame, None]:
    """
    Load example datasets used in frameon examples.

    Parameters
    ----------
    name : str
        Name of the dataset to load. Available options:
        - 'titanic': Titanic passenger data
        - 'iris': Iris flower dataset
        - 'tips': Restaurant tips data
        - 'diamonds': Diamond prices
        - 'superstore': Retail sales data

    Returns
    -------
    pandas.DataFrame
        Requested dataset as a DataFrame
    """
    data_dir = Path(__file__).parent
    dataset_map = {
        "titanic": lambda: pd.read_csv(data_dir / "titanic.csv"),
        "iris": lambda: pd.read_csv(data_dir / "iris.csv"),
        "tips": lambda: pd.read_csv(data_dir / "tips.csv"),
        "diamonds": lambda: pd.read_csv(data_dir / "diamonds.csv"),
        "superstore": lambda: pd.read_csv(
            data_dir / "superstore.csv",
            parse_dates=["Order Date"],
            date_format="%m/%d/%Y",
        ),
        "penguins": lambda: pd.read_csv(data_dir / "penguins.csv"),
        "taxis": lambda: pd.read_csv(data_dir / "taxis.csv", parse_dates=["pickup"]),
        "reviews": lambda: pd.read_csv(data_dir / "reviews.csv"),
    }

    if name not in dataset_map:
        raise ValueError(
            f"Unknown dataset: {name}. Available datasets: {list(dataset_map.keys())}"
        )

    return dataset_map[name]()
