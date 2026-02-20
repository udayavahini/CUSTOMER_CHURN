
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD


RANDOM_STATE = 42


def load_web_csv() -> pd.DataFrame:
    url = os.getenv("https://raw.githubusercontent.com/Nas-virat/Telco-Customer-Churn/master/Telco-Customer-Churn.csv", "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(
            "Set TELCO_DATA_URL to an http(s) URL pointing to the Telco CSV."
        )
    return pd.read_csv(url)


def plot_before(df_raw: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 4))

    # (a) Churn distribution (raw)
    ax1 = plt.subplot(1, 3, 1)
    if "Churn" in df_raw.columns:
        df_raw["Churn"].value_counts(dropna=False).plot(kind="bar", color="#007CB0", ax=ax1)
        ax1.set_title("Churn distribution (raw)")
        ax1.set_xlabel("Churn")
        ax1.set_ylabel("Count")
    else:
        ax1.text(0.5, 0.5, "Missing 'Churn' column", ha="center", va="center")
        ax1.axis("off")

    # (b) Missing values (raw)
    ax2 = plt.subplot(1, 3, 2)
    missing = df_raw.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(15)
    if len(missing) > 0:
        missing.plot(kind="bar", color="#ED8B00", ax=ax2)
        ax2.set_title("Top missing columns (raw)")
        ax2.set_xlabel("Column")
        ax2.set_ylabel("Missing rows")
        ax2.tick_params(axis="x", rotation=75)
    else:
        ax2.text(0.5, 0.5, "No NA values detected", ha="center", va="center")
        ax2.axis("off")

    # (c) TotalCharges parseability snapshot (raw)
    ax3 = plt.subplot(1, 3, 3)
    if "TotalCharges" in df_raw.columns:
        tc_num = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")
        pd.Series(
            {"Numeric": tc_num.notna().sum(), "Non-numeric/blank": tc_num.isna().sum()}
        ).plot(kind="bar", color=["#43B02A", "#DA291C"], ax=ax3)
        ax3.set_title("TotalCharges parseability (raw)")
        ax3.set_xlabel("")
        ax3.set_ylabel("Rows")
    else:
        ax3.text(0.5, 0.5, "Missing 'TotalCharges'", ha="center", va="center")
        ax3.axis("off")

    plt.tight_layout()
    plt.show()

    # Numeric distributions (raw-ish, coercing TotalCharges only for plotting)
    cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df_raw.columns]
    if cols:
        tmp = df_raw.copy()
        if "TotalCharges" in tmp.columns:
            tmp["TotalCharges"] = pd.to_numeric(tmp["TotalCharges"], errors="coerce")
        tmp[cols].hist(bins=30, figsize=(12, 3), color="#62B5E5")
        plt.suptitle("Numeric distributions (raw / coerced)")
        plt.tight_layout()
        plt.show()


def preprocess_fit_transform(df_raw: pd.DataFrame):
    df = df_raw.copy()

    if "Churn" not in df.columns:
        raise ValueError("Expected a 'Churn' column with values 'Yes'/'No'.")

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    if df["Churn"].isna().any():
        raise ValueError("Churn has unexpected values; expected only 'Yes'/'No'.")

    # Coerce TotalCharges (common: blanks stored as strings)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop identifier if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features),
        ],
        remainder="drop"
    )

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocess.fit(X_train)
    X_train_proc = preprocess.transform(X_train)

    return preprocess, X_train_proc, y_train, numeric_features, categorical_features


def plot_after(preprocess, X_train_proc, y_train, numeric_features, categorical_features) -> None:
    print("Processed train shape:", X_train_proc.shape)

    try:
        nnz = X_train_proc.nnz
        total = X_train_proc.shape[0] * X_train_proc.shape[1]
        print(f"Processed train sparsity: {1.0 - (nnz / total):.3f}")
    except Exception:
        print("Processed matrix appears dense (or sparsity not available).")

    # 2D projection (works well with sparse one-hot matrices)
    svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    X_2d = svd.fit_transform(X_train_proc)

    plt.figure(figsize=(7, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train.values, cmap="coolwarm", s=12, alpha=0.7)
    plt.title("After preprocessing: 2D projection (train)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    cbar = plt.colorbar()
    cbar.set_label("Churn (0=No, 1=Yes)")
    plt.tight_layout()
    plt.show()

    # Top engineered feature prevalences (mean of processed columns)
    try:
        col_means = np.asarray(X_train_proc.mean(axis=0)).ravel()
        top_idx = np.argsort(col_means)[-20:][::-1]

        ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_features) if categorical_features else np.array([])
        feature_names = np.array(numeric_features + list(cat_names))

        top_features = feature_names[top_idx]
        top_values = col_means[top_idx]

        plt.figure(figsize=(10, 5))
        plt.barh(top_features[::-1], top_values[::-1], color="#86BC25")
        plt.title("After preprocessing: top 20 feature prevalences")
        plt.xlabel("Mean value in processed matrix")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Skipping top-feature plot:", e)


def main():
    df_raw = load_web_csv()
    print("Loaded rows, cols:", df_raw.shape)
    plot_before(df_raw)

    preprocess, X_train_proc, y_train, num_feats, cat_feats = preprocess_fit_transform(df_raw)
    plot_after(preprocess, X_train_proc, y_train, num_feats, cat_feats)


if __name__ == "__main__":
    main()
