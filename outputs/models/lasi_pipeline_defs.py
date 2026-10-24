import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

def make_preprocessor(cont, cat):
    '''Imputation lives INSIDE the pipeline, so it is re-fit on training folds only.'''
    parts = []
    if cont:
        imp = IterativeImputer(max_iter=10, random_state=42) if len(cont) > 1 else SimpleImputer(strategy="median")
        parts.append(("continuous", Pipeline([("impute", imp), ("scale", StandardScaler())]), cont))
    if cat:
        parts.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return ColumnTransformer(parts)

def make_lr(cont, cat):
    return Pipeline([("pre", make_preprocessor(cont, cat)),
                     ("model", LogisticRegression(max_iter=3000))])

def make_gbm(cont, cat):
    return Pipeline([("pre", make_preprocessor(cont, cat)),
                     ("model", HistGradientBoostingClassifier(random_state=42))])
