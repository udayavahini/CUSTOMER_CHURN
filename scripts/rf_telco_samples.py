from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42

COLORS = {
    "green": "#86BC25",
    "blue": "#007CB0",
    "orange": "#ED8B00",
    "red": "#DA291C",
    "gray": "#97999B",
    "navy": "#002D72",
    "sky": "#62B5E5",
}


def find_csv(data_dir: Path) -> Path:
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in: {data_dir.resolve()}")
    return csvs[0]


def safe_strip_objects(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        out[c] = out[c].astype(str).str.strip()
        out[c] = out[c].replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})
    return out


def coerce_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    return out


def get_target(df: pd.DataFrame) -> pd.Series:
    if "Churn" not in df.columns:
        raise ValueError("Expected 'Churn' column (Yes/No).")
    y = df["Churn"].map({"Yes": 1, "No": 0})
    if y.isna().any():
        bad = df["Churn"].dropna().unique()
        raise ValueError(f"Unexpected Churn values (expected Yes/No): {bad}")
    y = y.astype(int)
    y.name = "Churn"
    return y


def drop_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "customerID" in out.columns:
        out = out.drop(columns=["customerID"])
    return out


def build_full_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )
    return pre


def split_and_balance_train(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float,
    random_state: int = 42,
    method: str = "oversample",  # "oversample" or "downsample"
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_size,
        stratify=y,
        random_state=random_state
    )

    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)

    df0 = train_df[train_df["Churn"] == 0]
    df1 = train_df[train_df["Churn"] == 1]

    if len(df0) == 0 or len(df1) == 0:
        raise ValueError("Train split lost a class; check stratify and target values.")

    if method == "downsample":
        n = min(len(df0), len(df1))
        df0b = resample(df0, replace=False, n_samples=n, random_state=random_state)
        df1b = resample(df1, replace=False, n_samples=n, random_state=random_state)
    elif method == "oversample":
        n = max(len(df0), len(df1))
        df0b = resample(df0, replace=(len(df0) < n), n_samples=n, random_state=random_state)
        df1b = resample(df1, replace=(len(df1) < n), n_samples=n, random_state=random_state)
    else:
        raise ValueError("method must be 'oversample' or 'downsample'")

    train_bal = (
        pd.concat([df0b, df1b], axis=0)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    y_train_bal = train_bal["Churn"].astype(int)
    X_train_bal = train_bal.drop(columns=["Churn"])

    return X_train_bal, X_test, y_train_bal, y_test


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def plot_svd_errors(preprocessor, model, X_train, X_test, y_test, title: str, out_path: Path) -> None:
    # Fit SVD on TRAIN transformed (no leakage), project TEST, color by correct/incorrect.
    Xt_train = preprocessor.transform(X_train)
    Xt_test = preprocessor.transform(X_test)

    scaler = StandardScaler(with_mean=False)
    Xt_train_s = scaler.fit_transform(Xt_train)
    Xt_test_s = scaler.transform(Xt_test)

    svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    Z_train = svd.fit_transform(Xt_train_s)
    Z_test = svd.transform(Xt_test_s)

    y_pred = model.predict(X_test)
    correct = (y_pred == y_test.values)

    plt.figure(figsize=(6.4, 5.4))
    plt.scatter(Z_test[correct, 0], Z_test[correct, 1], s=10, alpha=0.65, c=COLORS["green"], label="Correct")
    plt.scatter(Z_test[~correct, 0], Z_test[~correct, 1], s=14, alpha=0.85, c=COLORS["red"], label="Incorrect")
    plt.title(title)
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.legend(frameon=False, loc="best")
    savefig(out_path)


def plot_top_feature_importances(pipe: Pipeline, out_path: Path, top_n: int = 20) -> None:
    pre = pipe.named_steps["pre"]
    rf = pipe.named_steps["rf"]

    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = np.array([f"f{i}" for i in range(len(rf.feature_importances_))])

    imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False).head(top_n)[::-1]

    plt.figure(figsize=(9.5, 6.2))
    plt.barh(imp.index.astype(str), imp.values, color=COLORS["blue"])
    plt.title(f"Top {top_n} feature importances (RF)")
    plt.xlabel("Importance")
    plt.ylabel("")
    savefig(out_path)


def evaluate_one_split(df_raw: pd.DataFrame, train_size: float, balance_method: str, out_dir: Path) -> dict:
    y = get_target(df_raw)
    df_nod = drop_id(df_raw)
    X = df_nod.drop(columns=["Churn"])

    X_train_bal, X_test, y_train_bal, y_test = split_and_balance_train(
        X, y, train_size=train_size, random_state=RANDOM_STATE, method=balance_method
    )

    pre = build_full_preprocessor(X_train_bal)

    pipe = Pipeline(steps=[
        ("pre", pre),
        ("rf", RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
        )),
    ])

    pipe.fit(X_train_bal, y_train_bal)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "train_size": train_size,
        "balance_method": balance_method,
        "train_n": int(len(y_train_bal)),
        "test_n": int(len(y_test)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    tag = f"train_{int(train_size*100)}_{balance_method}"
    (out_dir / tag).mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    plt.figure(figsize=(5.4, 4.6))
    disp.plot(values_format="d", cmap="Blues", colorbar=False)
    plt.title(f"Confusion matrix ({int(train_size*100)}% train)")
    savefig(out_dir / tag / "confusion_matrix.png")

    # ROC curve
    plt.figure(figsize=(5.8, 4.8))
    RocCurveDisplay.from_predictions(y_test, y_proba, name="RF", color=COLORS["blue"])
    plt.plot([0, 1], [0, 1], linestyle="--", color=COLORS["gray"], linewidth=1)
    plt.title(f"ROC curve ({int(train_size*100)}% train)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    savefig(out_dir / tag / "roc_curve.png")

    # SVD 2D scatter highlighting errors
    plot_svd_errors(
        preprocessor=pipe.named_steps["pre"],
        model=pipe,
        X_train=X_train_bal,
        X_test=X_test,
        y_test=y_test,
        title=f"SVD (test) errors — {int(train_size*100)}% train",
        out_path=out_dir / tag / "svd_errors.png",
    )

    # Feature importances
    plot_top_feature_importances(pipe, out_dir / tag / "feature_importances_top20.png", top_n=20)

    return metrics


def main() -> None:
    data_dir = Path("data")
    out_dir = Path("reports") / "rf"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = find_csv(data_dir)
    df0 = pd.read_csv(csv_path)

    df = safe_strip_objects(df0)
    df = coerce_total_charges(df)

    all_metrics = []
    for ts in (0.70, 0.75, 0.80):
        m = evaluate_one_split(df, train_size=ts, balance_method="oversample", out_dir=out_dir)
        all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics).sort_values("train_size")
    print("\nRandom Forest results (test set):")
    print(metrics_df[[
        "train_size",
        "train_n",
        "test_n",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    ]].to_string(index=False))

    metrics_path = out_dir / "rf_metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics CSV: {metrics_path.resolve()}")
    print(f"Saved figures under: {(out_dir).resolve()}")


if __name__ == "__main__":
    main()
