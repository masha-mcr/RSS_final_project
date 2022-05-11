import click
import mlflow
import numpy as np
import pandas as pd

from .data import get_dataset, preprocess, data_train_test_split, get_columns
from .pipeline import create_pipeline
from pathlib import Path
from joblib import dump
from sklearn.model_selection import (
    KFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.metrics import accuracy_score, recall_score, precision_score
from typing import Tuple
from sklearn.pipeline import Pipeline
import mlflow.sklearn


@click.command()
@click.option(
    "-d",
    "--data-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "-m",
    "--model-alg",
    default="tree",
    type=str,
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--tuning",
    default="default",
    type=str,
    show_default=True,
)
@click.option(
    "-r",
    "--random_state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
def train(
    data_path: Path,
    model_alg: str,
    save_model_path: Path,
    random_state: int,
    use_scaler: bool,
    tuning: str,
    test_split_ratio: float,
) -> None:
    click.echo("Loading and preprocessing data.")
    df = preprocess(get_dataset(data_path))
    X_train, X_test, y_train, y_test = data_train_test_split(
        df, test_split_ratio, random_state
    )
    numeric_cols, _ = get_columns(X_train.columns)

    if tuning == "default":
        best_pipeline = run_default(
            X_train,
            y_train,
            X_test,
            y_test,
            use_scaler,
            numeric_cols,
            model_alg,
            random_state,
        )
    elif tuning in ["grid", "random"]:
        best_pipeline = run_nestedCV(
            X_train,
            y_train,
            X_test,
            y_test,
            use_scaler,
            numeric_cols,
            model_alg,
            random_state,
            tuning,
        )
    else:
        raise ValueError(f"Unknown tuning method {tuning}")

    dump(best_pipeline, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")


def get_search_grid(model_alg: str) -> dict:
    if model_alg == "log-reg":
        return {
            "estimator__penalty": ["l1", "l2"],
            "estimator__solver": ["saga"],
            "estimator__C": [0.1, 0.5, 1, 5, 20],
            "estimator__max_iter": [400, 500],
        }
    elif model_alg == "tree":
        return {
            "estimator__criterion": ["gini", "entropy"],
            "estimator__max_depth": [1, 3, 4, 5, 10, 20, None],
            "estimator__max_features": ["auto", "sqrt", "log2", None],
            "estimator__min_samples_split": [2, 5, 10, 20],
            "estimator__min_samples_leaf": [1, 3, 6, 10],
        }
    elif model_alg == "forest":
        return {
            "estimator__criterion": ["gini", "entropy"],
            "estimator__n_estimators": [50, 100, 300, 450, 700],
            "estimator__min_samples_split": [2, 5, 10, 15, 20],
            "estimator__min_samples_leaf": [1, 3, 6, 10],
            "estimator__max_depth": [1, 3, 4, 5, 10, 20, None],
            "estimator__max_features": ["auto", "sqrt", "log2", 2, 4, 6],
        }


def run_default(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_scaler: bool,
    numeric_cols: list,
    model_alg: str,
    random_state: int,
) -> Pipeline:
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, numeric_cols, model_alg, random_state)
        click.echo("Starting CV with default hyperparameters.")
        cv_result = cross_validate(
            pipeline, X_train, y_train, cv=5, return_estimator=True, scoring="accuracy"
        )
        accuracy = np.mean(cv_result["test_score"])
        closest_pipeline_idx = np.argmin(
            np.abs(np.array(cv_result["test_score"]) - accuracy)
        )
        closest_pipeline = cv_result["estimator"][closest_pipeline_idx]
        y_pred = closest_pipeline.predict(X_test)
        acc, rec, pres = get_metrics(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall_weighted", rec)
        mlflow.log_metric("precision_weighted", pres)
        click.echo(f"Average accuracy: {accuracy}.")
        click.echo(f"Accuracy on holdout test set: {acc}.")

        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("HP tuning", "default")
        mlflow.log_param("model", model_alg)

        closest_model = closest_pipeline["estimator"]
        log_model_params(closest_model, model_alg)

        click.echo("Experiment recorded via mlflow")
        return closest_pipeline


def run_nestedCV(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_scaler: bool,
    numeric_cols: list,
    model_alg: str,
    random_state: int,
    tuning: str,
) -> Pipeline:
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, numeric_cols, model_alg, random_state)
        click.echo("Starting nested CV with HP tuning.")
        cv_outer = KFold(n_splits=5, shuffle=True, random_state=random_state)
        outer_results, outer_estimators = [], []
        for train_idx, test_idx in cv_outer.split(X_train):
            X_train_outer, X_test_outer = (
                X_train.iloc[train_idx],
                X_train.iloc[test_idx],
            )
            y_train_outer, y_test_outer = (
                y_train.iloc[train_idx],
                y_train.iloc[test_idx],
            )

            cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)
            param_grid = get_search_grid(model_alg)
            if tuning == "grid":
                search = GridSearchCV(
                    pipeline,
                    param_grid,
                    scoring="accuracy",
                    cv=cv_inner,
                    refit=True,
                    n_jobs=-1,
                    error_score=0,
                )
                click.echo("Starting GridSearchCV on outer fold.")
            elif tuning == "random":
                search = RandomizedSearchCV(
                    pipeline,
                    param_grid,
                    scoring="accuracy",
                    cv=cv_inner,
                    refit=True,
                    n_jobs=-1,
                    error_score=0,
                )
                click.echo("Starting RandomizedSearchCV on outer fold.")

            try:
                result = search.fit(X_train_outer, y_train_outer)
                best_model = result.best_estimator_
                y_pred_outer = best_model.predict(X_test_outer)
                outer_results.append(accuracy_score(y_test_outer, y_pred_outer))
                outer_estimators.append(best_model)
            except ValueError:
                click.echo("Fit failed due to hyperparameters incompatibility")
            finally:
                pass

        accuracy = np.mean(outer_results)
        closest_pipeline_idx = np.argmin(np.abs(np.array(outer_results) - accuracy))
        closest_pipeline = outer_estimators[closest_pipeline_idx]
        y_pred = closest_pipeline.predict(X_test)
        acc, rec, pres = get_metrics(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall_weighted", rec)
        mlflow.log_metric("precision_weighted", pres)
        click.echo(f"Average accuracy: {accuracy}.")
        click.echo(f"Accuracy on holdout test set: {acc}.")

        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("HP tuning", tuning)
        mlflow.log_param("model", model_alg)

        closest_model = closest_pipeline["estimator"]
        log_model_params(closest_model, model_alg)

        click.echo("Experiment recorded via mlflow")
        return closest_pipeline


def get_metrics(y_test: pd.Series, y_pred: pd.Series) -> Tuple[float, float, float]:
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    return accuracy, recall, precision


def log_model_params(model: Pipeline, model_alg: str) -> None:
    if model_alg == "log-reg":
        mlflow.log_param("penalty", model.penalty)
        mlflow.log_param("solver", model.solver)
        mlflow.log_param("C", model.C)
        mlflow.log_param("max_iter", model.max_iter)
    elif model_alg == "tree":
        mlflow.log_param("criterion", model.criterion)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("max_features", model.max_features)
        mlflow.log_param("min_samples_leaf", model.min_samples_leaf)
        mlflow.log_param("min_samples_split", model.min_samples_split)
    elif model_alg == "forest":
        mlflow.log_param("criterion", model.criterion)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_features", model.max_features)
        mlflow.log_param("min_samples_leaf", model.min_samples_leaf)
        mlflow.log_param("min_samples_split", model.min_samples_split)
