import click
import mlflow
import numpy as np
import pandas as pd

from .data import get_dataset, preprocess, data_train_test_split, get_columns
from .pipeline import create_pipeline
from pathlib import Path
from joblib import dump
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score
from typing import Tuple
from sklearn.pipeline import Pipeline

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
    default='default',
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
def train(data_path: Path, model_alg: str, save_model_path: Path, random_state: int, use_scaler: bool, tuning: str):
    click.echo('Loading and preprocessing data.')
    df = preprocess(get_dataset(data_path))
    X_train, X_test, y_train, y_test  = data_train_test_split(df, 0.2, random_state) 
    numeric_cols, _ = get_columns(X_train.columns)

    best_pipeline = run_default(X_train, y_train, X_test, y_test, 
                                use_scaler, numeric_cols, model_alg, random_state)
    dump(best_pipeline, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")
    
    # with mlflow.start_run():

    #     cv_outer = KFold(n_splits=5, shuffle=True, random_state=random_state)
    #     outer_results = list()
    #     for train_idx, test_idx in cv_outer.split(X_train):
    #         X_train_outer, X_test_outer = X_train[train_idx, :], X_train[test_idx, :]
    #         y_train_outer, y_test_outer = y_train[train_idx], y_train[test_idx]

    #         cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)

    #         pipeline =  create_pipeline(use_scaler, numeric_cols, model_alg, random_state)
    #         param_grid = get_search_grid(model_alg)

    #         click.echo('Started GridSearchCV')
    #         search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=cv_inner, refit=True, n_jobs=-1)
    #         result = search.fit(X_train, y_train)
    #         best_model = result.best_estimator_
    #         y_pred = best_model.predict(X_test)

            #click.echo('Started RandomizedSearchCV')
            #random_search = RandomizedSearchCV(pipeline, param_grid, scoring='accuracy', cv=cv_inner, refit=True, n_jobs=-1)
            #random_result = random_search.fit(X_train, y_train)
            #best_model_random = random_result.best_estimator_
            #y_pred_random = best_model_random.predict(X_test)
            #acc = accuracy_score(y_test, yhat)



def get_search_grid(model_alg: str) -> dict:
    if model_name == 'log-reg':
        return {
            'penalty': ['l1','l2', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'C': [0.1, 0.5, 1, 1.5, 2, 5, 20],
            'max_iter': [200, 300, 450]
        }
    elif model_name == 'tree':
        depth = range(3, 15)
        depth.append(None)
        return {
            'criterion': ['gini', 'entropy'],
            'max_depth': depth,
            'max_features': ['auto', 'sqrt', 'log2', None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 3, 6, 10],
        }
    elif model_name == 'forest':
        depth = range(3, 15)
        depth.append(None)
        return {
            'criterion': ['gini', 'entropy'],
            'n_estimators': [20, 50, 100, 300, 450, 700],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 3, 6, 10],
            'max_depth': depth,
            'max_features': ['auto', 'sqrt', 'log2', 2, 4, 6],
        }
        
def run_default(X_train, y_train, X_test, y_test, use_scaler, 
                    numeric_cols, model_alg, random_state)-> Pipeline:
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, numeric_cols, model_alg, random_state)
        click.echo('Starting CV with default hyperparameters.')
        cv_result = cross_validate(pipeline, X_train, y_train, cv=5, return_estimator=True, scoring='accuracy')
        accuracy = np.mean(cv_result['test_score'])
        closest_pipeline_idx = np.argmin(np.abs(np.array(cv_result['test_score']) - accuracy))
        closest_pipeline = cv_result['estimator'][closest_pipeline_idx]
        y_pred = closest_pipeline.predict(X_test)
        acc, rec, pres = get_metrics(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall_weighted", rec)
        mlflow.log_metric("precision_weighted", pres)
        click.echo(f"Average accuracy: {accuracy}.")
        click.echo(f"Accuracy on holdout test set: {acc}.")
        
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("HP tuning", 'default')
        mlflow.log_param("model", model_alg)

        closest_model = closest_pipeline['estimator']
        if model_alg == 'log-reg':
            mlflow.log_param("penalty", closest_model.penalty)
            mlflow.log_param("solver", closest_model.solver)
            mlflow.log_param("C", closest_model.C)
            mlflow.log_param("max_iter", closest_model.max_iter)
        elif model_alg == 'tree':
            mlflow.log_param("criterion", closest_model.criterion)
            mlflow.log_param("max_depth", closest_model.max_depth)
            mlflow.log_param("max_features", closest_model.max_features)
            mlflow.log_param("min_samples_leaf", closest_model.min_samples_leaf)
            mlflow.log_param("min_samples_split", closest_model.min_samples_split)
        elif model_alg == 'forest':
            mlflow.log_param("criterion", closest_model.criterion)
            mlflow.log_param("max_depth", closest_model.max_depth)
            mlflow.log_param("n_estimators", closest_model.n_estimators)
            mlflow.log_param("max_features", closest_model.max_features)
            mlflow.log_param("min_samples_leaf", closest_model.min_samples_leaf)
            mlflow.log_param("min_samples_split", closest_model.min_samples_split)
        
        click.echo('Experiment recorded via mlflow')
        return closest_pipeline


def run_grid():
    pass

def run_random():
    pass

def get_metrics(y_test: pd.Series, y_pred: pd.Series)-> Tuple[float, float, float]:
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    return accuracy, recall, precision



        
