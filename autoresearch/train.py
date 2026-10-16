"""Titanic baseline. Run `uv run train.py`; see program.md for the protocol."""

import argparse
import hashlib
import platform
import signal
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# === EXPERIMENT AREA: the agent may change this function and add transformers. ===
def build_pipeline():
    """Start from the four features used in the October 2019 hands-on.

    Input is a raw DataFrame (all predictor columns except PassengerId).
    Any learned feature engineering must be fitted inside this pipeline.
    """
    preprocessing = ColumnTransformer(
        [
            ("pclass", "passthrough", ["Pclass"]),
            ("age", SimpleImputer(strategy="mean"), ["Age"]),
            (
                "sex",
                OneHotEncoder(
                    categories=[["female", "male"]],
                    drop="if_binary",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["Sex"],
            ),
            (
                "embarked",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["Embarked"],
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", LogisticRegression(random_state=1, solver="liblinear", max_iter=100)),
        ]
    )


# === FIXED EVALUATION: do not edit during an experiment session. ===
DATA_DIR = Path(__file__).resolve().parent / "data"
SEED = 1
HOLDOUT_SIZE = 0.2
N_SPLITS = 5
TIME_LIMIT_SECONDS = 120


def time_limit_exceeded(signum, frame):
    raise TimeoutError(f"Experiment exceeded {TIME_LIMIT_SECONDS} seconds")


def load_training_data():
    path = DATA_DIR / "train.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run bash prepare.sh first.")
    frame = pd.read_csv(path)
    required = {"PassengerId", "Survived", "Age", "Pclass", "Sex", "Embarked"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not frame["Survived"].isin([0, 1]).all():
        raise ValueError("Survived must contain only 0 and 1, without missing values")
    if frame["PassengerId"].isna().any() or not frame["PassengerId"].is_unique:
        raise ValueError("PassengerId must be present and unique")
    print(f"data_sha256: {hashlib.sha256(path.read_bytes()).hexdigest()}")
    return frame.drop(columns=["Survived", "PassengerId"]), frame["Survived"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit", action="store_true", help="Evaluate the frozen winner on the reserved holdout once")
    mode.add_argument("--submission", type=Path, metavar="CSV", help="Refit on all labels and write a CSV; never uploads")
    args = parser.parse_args()
    if args.submission is not None and args.submission.exists():
        parser.error(f"Refusing to overwrite {args.submission}")

    started = time.perf_counter()
    signal.signal(signal.SIGALRM, time_limit_exceeded)
    signal.alarm(TIME_LIMIT_SECONDS)
    warnings.filterwarnings("error", category=ConvergenceWarning)
    print(f"python_version: {platform.python_version()}")
    print(f"sklearn_version: {sklearn.__version__}")
    print(f"numpy_version: {np.__version__}")
    print(f"pandas_version: {pd.__version__}")
    print(f"code_sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}")
    X, y = load_training_data()
    pipeline = build_pipeline()
    print(f"model: {type(pipeline.steps[-1][1]).__name__}")

    if args.submission is not None:
        test = pd.read_csv(DATA_DIR / "test.csv")
        if not set(X.columns).issubset(test.columns):
            raise ValueError("test.csv is missing predictor columns")
        if test["PassengerId"].isna().any() or not test["PassengerId"].is_unique:
            raise ValueError("test.csv PassengerId must be present and unique")
        pipeline.fit(X, y)
        predictions = pipeline.predict(test.loc[:, X.columns])
        if not np.isin(predictions, [0, 1]).all():
            raise ValueError("Submission predictions must be 0 or 1")
        # Exclusive creation also protects against a concurrent overwrite.
        with args.submission.open("x") as output:
            pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions.astype(int)}).to_csv(output, index=False)
        print(f"submission_rows: {len(predictions)}")
        print(f"submission_path: {args.submission.resolve()}")
    else:
        # Split raw rows before any learned preprocessing. The holdout is never
        # fitted or scored in the default exploration command.
        X_dev, X_holdout, y_dev, y_holdout = train_test_split(
            X, y, test_size=HOLDOUT_SIZE, stratify=y, random_state=SEED
        )
        print(f"development_rows: {len(X_dev)}")
        print(f"holdout_rows: {len(X_holdout)}")
        if args.audit:
            pipeline.fit(X_dev, y_dev)
            score = accuracy_score(y_holdout, pipeline.predict(X_holdout))
            print(f"holdout_accuracy: {score:.6f}")
        else:
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
            scores = []
            for fold, (train_index, val_index) in enumerate(cv.split(X_dev, y_dev), start=1):
                model = clone(pipeline)
                model.fit(X_dev.iloc[train_index], y_dev.iloc[train_index])
                score = accuracy_score(y_dev.iloc[val_index], model.predict(X_dev.iloc[val_index]))
                scores.append(score)
                print(f"fold_{fold}_accuracy: {score:.6f}")
            print(f"cv_accuracy: {np.mean(scores):.6f}")
            print(f"cv_std: {np.std(scores):.6f}")
    print(f"total_seconds: {time.perf_counter() - started:.3f}")
    signal.alarm(0)


if __name__ == "__main__":
    main()
