import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import click
from pathlib import Path
from typing import Tuple

from sklearn.model_selection import train_test_split

@click.command()
@click.option(
    "-d",
    "--data-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
def get_report(data_path: Path)-> pd.DataFrame:
    df = pd.read_csv(data_path,index_col='Id')
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file("profiling_report.html")

def get_dataset(data_path: Path)-> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df


def data_train_test_split(df: pd.DataFrame, ratio: float, random_state: int)-> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = df.drop('Cover_Type', axis=1)
    target = df['Cover_Type']
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=ratio, random_state=random_state
    )
    return features_train, features_test, target_train, target_test




def get_columns(columns: list)-> Tuple[list, list]:
    soils = [s for s in columns if s.startswith('Soil_Type')]
    wilderness = [w for w in columns if w.startswith('Wilderness_Area')]
    binary_cols = wilderness + soils
    numeric_cols = list(set(columns) - set(binary_cols))
    return numeric_cols, binary_cols


def preprocess(df)-> pd.DataFrame:
    cols_to_drop = [] 
    for col in df.columns:
        unique,counts = np.unique(df[col],return_counts=True) 
        distinct = pd.DataFrame({'value': unique, 'percent': counts/df[col].shape[0]*100})
        if (distinct['percent'] > 99.6).any():
            cols_to_drop.append(col)
    new_df = df.drop(labels=cols_to_drop,axis=1)
    return new_df


    
