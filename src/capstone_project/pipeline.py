from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer


def create_pipeline(
    use_scaler: bool, numeric_cols:str, model_alg:str, random_state: int

) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        numeric_scaler = ColumnTransformer(transformers=[("num", StandardScaler(), numeric_cols)], remainder='passthrough')
        pipeline_steps.append(("scaler", numeric_scaler))
    if model_alg == 'log-reg':
        pipeline_steps.append(("estimator", LogisticRegression(random_state=random_state, n_jobs=-1, max_iter=350)))
    elif model_alg == 'tree':
        pipeline_steps.append(("estimator", DecisionTreeClassifier(random_state=random_state)))
    elif model_alg == 'forest':
        pipeline_steps.append(("estimator", RandomForestClassifier(random_state=random_state, n_jobs=-1)))
    else: 
        raise ValueError(f'Unknown model {model_alg}')
    
    return Pipeline(steps=pipeline_steps)