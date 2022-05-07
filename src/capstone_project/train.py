import click

from .data import get_dataset, get_report
from pathlib import Path

@click.command()
@click.option(
    "-d",
    "--data-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)

def train(data_path):
    click.echo('here')
    df = get_dataset(data_path)
    profile = get_report(df)
    
