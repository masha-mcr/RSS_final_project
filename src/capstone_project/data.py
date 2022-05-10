import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import click
from pathlib import Path
from typing import Tuple


@click.command()
@click.option(
    "-d",
    "--data-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
def get_report(data_path: Path)-> pd.DataFrame:
    df = pd.read_csv(data_path)
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file("profiling_report.html")

def get_dataset(data_path):
    df = pd.read_csv(data_path)
    return df


def train_val_split(df, ratio, random_state)-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    #click.echo(f"Dataset shape: {dataset.shape}.")
    features = df.drop("target", axis=1)
    target = df["target"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=ratio, random_state=random_state
    )
    return features_train, features_val, target_train, target_val



    
