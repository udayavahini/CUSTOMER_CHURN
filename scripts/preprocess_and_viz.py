from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD


RANDOM_STATE = 42

# Deloitte-ish solid palette (use for marks only, not text)
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


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def safe_strip_objects(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        out[c] = out[c].astype(str).str.strip()
        # turn literal "nan" into actual NaN
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
    return y.astype(int)


def drop_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "customerID" in out.columns:
        out = out.drop(columns=["customerID"])
    return out


def plot_missing_by_column(df: pd.DataFrame, title: str, out_path: Path, top_n: int = 30) -> None:
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    plt.figure(figsize=(11, 4.2))
    if len(miss) == 0:
        plt.title(title)
        plt.text(0.5, 0.5, "No missing values detected", ha="center", va="center")
        plt.axis("off")
        savefig(out_path)
        return

    miss_plot = miss.head(top_n)
    miss_plot.plot(kind="bar", color=COLORS["orange"])
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Missing rows")
    plt.xticks(rotation=75, ha="right")
    savefig(out_path)


def plot_churn_balance(y: pd.Series, title: str, out_path: Path) -> None:
    vc = y.value_counts().reindex([0, 1], fill_value=0)
    labels = ["No (0)", "Yes (1)"]
    plt.figure(figsize=(5.8, 4))
    plt.bar(labels, vc.values, color=[COLORS["blue"], COLORS["red"]])
    plt.title(title)
    plt.xlabel("Churn")
    plt.ylabel("Count")
    savefig(out_path)


def plot_churn_pie(y: pd.Series, title: str, out_path: Path) -> None:
    vc = y.value_counts().reindex([0, 1], fill_value=0)
    plt.figure(figsize=(5.2, 4.2))
    plt.pie(
        vc.values,
        labels=["No", "Yes"],
        colors=[COLORS["blue"], COLORS["red"]],
        autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
        startangle=90,
    )
    plt.title(title)
    savefig(out_path)


def plot_numeric_histograms(df: pd.DataFrame, cols: list[str], title: str, out_path: Path, color: str) -> None:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return
    ax = df[cols].hist(bins=30, figsize=(11.2, 3.2), color=color)
    plt.suptitle(title)
    # Improve layout for long labels
    for a in np.ravel(ax):
        a.set_xlabel(a.get_xlabel(), labelpad=6)
    savefig(out_path)


def plot_numeric_corr_heatmap(df: pd.DataFrame, cols: list[str], title: str, out_path: Path) -> None:
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return
    corr = df[cols].corr(numeric_only=True)
    plt.figure(figsize=(6.2, 5.2))
    im = plt.imshow(corr.values, cmap="Blues", vmin=-1, vmax=1)
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    savefig(out_path)


def plot_boxplots_by_churn(df: pd.DataFrame, y: pd.Series, cols: list[str], title: str, out_path: Path) -> None:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return

    # align indices
    tmp = df[cols].copy()
    tmp["Churn"] = y.values

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        groups = [tmp.loc[tmp["Churn"] == 0, c].dropna(), tmp.loc[tmp["Churn"] == 1, c].dropna()]
        ax.boxplot(groups, labels=["No", "Yes"], patch_artist=True,
                  boxprops=dict(facecolor=COLORS["sky"], color=COLORS["gray"]),
                  medianprops=dict(color=COLORS["navy"]))
        ax.set_title(c)
        ax.set_xlabel("Churn")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close()


def churn_rate_by_category(df: pd.DataFrame, y: pd.Series, col: str, title: str, out_path: Path, top_k: int = 10) -> None:
    if col not in df.columns:
        return

    tmp = df[[col]].copy()
    tmp["Churn"] = y.values
    # bucket rare categories for readability
    counts = tmp[col].value_counts(dropna=False)
    keep = counts.head(top_k).index
    tmp[col] = tmp[col].where(tmp[col].isin(keep), other="Other")

    agg = tmp.groupby(col, dropna=False).agg(rate=("Churn", "mean"), n=("Churn", "size")).sort_values("n", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(agg.index.astype(str), agg["rate"].values, color=COLORS["green"])
    axes[0].set_title("Churn rate")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Rate (0–1)")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(agg.index.astype(str), agg["n"].values, color=COLORS["blue"])
    axes[1].set_title("Count")
    axes[1].set_xlabel(col)
    axes[1].set_ylabel("Customers")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close()


def build_full_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
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
    return pre, num_cols, cat_cols


def make_imputed_X_dataframe(X: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """Impute in original column space (no one-hot), so we can plot 'missing by column after'."""
    X2 = X.copy()

    if num_cols:
        imp_num = SimpleImputer(strategy="median")
        X2[num_cols] = imp_num.fit_transform(X2[num_cols])

    if cat_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        X2[cat_cols] = imp_cat.fit_transform(X2[cat_cols])

    return X2


def svd_projection_numeric_only(X: pd.DataFrame, y: pd.Series, title: str, out_path: Path) -> None:
    # "Before" projection: numeric-only, minimal cleanup (impute + scale), then SVD -> 2D
    num = X.select_dtypes(include=["number"]).copy()
    if num.shape[1] < 2:
        return

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=True, with_std=True)

    Xn = imp.fit_transform(num)
    Xn = scaler.fit_transform(Xn)

    svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    X2 = svd.fit_transform(Xn)

    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(X2[:, 0], X2[:, 1], c=y.values, cmap="coolwarm", s=10, alpha=0.75)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    cb = plt.colorbar()
    cb.set_label("Churn (0=No, 1=Yes)")
    savefig(out_path)


def svd_projection_after_preprocess(X: pd.DataFrame, y: pd.Series, title: str, out_path: Path) -> None:
    # "After" projection: full preprocessing (impute + OHE), then SVD -> 2D
    pre, _, _ = build_full_preprocessor(X)
    # Fit on train split for stability and to mirror modeling practice
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    Xt = pre.fit_transform(X_train)

    # Standardize AFTER OHE (works for sparse/dense). with_mean=False is needed for sparse.
    scaler = StandardScaler(with_mean=False)
    Xt_s = scaler.fit_transform(Xt)

    svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    X2 = svd.fit_transform(Xt_s)

    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(X2[:, 0], X2[:, 1], c=y_train.values, cmap="coolwarm", s=10, alpha=0.75)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    cb = plt.colorbar()
    cb.set_label("Churn (0=No, 1=Yes)")
    savefig(out_path)


def write_report(df_raw: pd.DataFrame, X_raw: pd.DataFrame, X_imputed: pd.DataFrame, out_path: Path, csv_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig_dir = out_path.parent / "figures"
    figures = sorted([p.name for p in fig_dir.glob("*.png")])

    lines: list[str] = []
    lines.append("# Telco churn: preprocessing + visuals\n\n")
    lines.append(f"- Input CSV: `{csv_path.name}`\n")
    lines.append(f"- Raw shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols\n")
    lines.append(f"- Feature shape (raw, post-basic-clean): {X_raw.shape[0]} rows × {X_raw.shape[1]} cols\n")
    lines.append(f"- Feature shape (imputed, original columns): {X_imputed.shape[0]} rows × {X_imputed.shape[1]} cols\n")

    if "Churn" in df_raw.columns:
        vc = df_raw["Churn"].value_counts(dropna=False)
        lines.append("\n## Churn (raw values)\n")
        for k, v in vc.items():
            lines.append(f"- {k}: {v}\n")

    lines.append("\n## Figures (generated)\n")
    for f in figures:
        lines.append(f"- `reports/figures/{f}`\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    data_dir = Path("data")
    reports_dir = Path("reports")
    figs_dir = reports_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = find_csv(data_dir)
    df_raw0 = pd.read_csv(csv_path)

    # ---- "Basic cleaning" that we still treat as raw for plots (strip + TotalCharges coercion)
    df_raw = safe_strip_objects(df_raw0)
    df_raw = coerce_total_charges(df_raw)

    # Target + X
    y = get_target(df_raw)
    df_nod = drop_id(df_raw)
    X = df_nod.drop(columns=["Churn"])

    # ---- Churn balance (before/after)
    plot_churn_balance(y, "Churn balance (raw)", figs_dir / "churn_balance_before_bar.png")
    plot_churn_pie(y, "Churn balance (raw)", figs_dir / "churn_balance_before_pie.png")

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    plot_churn_balance(y_train, "Churn balance (train split)", figs_dir / "churn_balance_after_train_bar.png")
    plot_churn_pie(y_train, "Churn balance (train split)", figs_dir / "churn_balance_after_train_pie.png")

    # ---- Missingness by column (before vs after imputation)
    plot_missing_by_column(df_nod, "Missing values by column (before)", figs_dir / "missing_by_column_before.png", top_n=35)

    pre, num_cols, cat_cols = build_full_preprocessor(X)
    X_imputed = make_imputed_X_dataframe(X, num_cols, cat_cols)

    # Plot missingness in original column space after imputation (should be zero)
    df_after_impute = X_imputed.copy()
    df_after_impute["Churn"] = y.values
    df_after_impute = df_after_impute  # keep naming consistent
    plot_missing_by_column(
        pd.concat([X_imputed, y.rename("Churn")], axis=1),
        "Missing values by column (after imputation)",
        figs_dir / "missing_by_column_after.png",
        top_n=35,
    )

    # ---- Dataset distributions (before vs after)
    numeric_focus = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df_nod.columns]
    plot_numeric_histograms(df_nod, numeric_focus, "Numeric distributions (before)", figs_dir / "numeric_hist_before.png", COLORS["sky"])
    plot_numeric_histograms(pd.concat([X_imputed, y.rename("Churn")], axis=1), numeric_focus, "Numeric distributions (after imputation)", figs_dir / "numeric_hist_after.png", COLORS["green"])

    plot_numeric_corr_heatmap(df_nod, numeric_focus, "Numeric correlation (before)", figs_dir / "numeric_corr_before.png")
    plot_numeric_corr_heatmap(pd.concat([X_imputed, y.rename("Churn")], axis=1), numeric_focus, "Numeric correlation (after imputation)", figs_dir / "numeric_corr_after.png")

    # Boxplots by churn (before/after imputation—should look similar unless many missings)
    plot_boxplots_by_churn(df_nod, y, numeric_focus, "Numeric boxplots by churn (before)", figs_dir / "boxplots_by_churn_before.png")
    plot_boxplots_by_churn(pd.concat([X_imputed, y.rename("Churn")], axis=1), y, numeric_focus, "Numeric boxplots by churn (after imputation)", figs_dir / "boxplots_by_churn_after.png")

    # ---- SVD projections (before vs after)
    svd_projection_numeric_only(X, y, "SVD projection (before: numeric-only)", figs_dir / "svd_before_numeric_only.png")
    svd_projection_after_preprocess(X, y, "SVD projection (after: full preprocessing)", figs_dir / "svd_after_full_preprocess.png")

    # ---- Extra visuals: churn rate by category (top-k)
    for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "SeniorCitizen"]:
        churn_rate_by_category(
            X, y, col,
            f"Churn by {col} (rate + count)",
            figs_dir / f"churn_by_{col}.png",
            top_k=10,
        )

    # ---- Write markdown report with figure list
    write_report(df_raw0, X, X_imputed, reports_dir / "report.md", csv_path)

    print("Done.")
    print(f"CSV: {csv_path}")
    print(f"Report: {reports_dir / 'report.md'}")
    print(f"Figures: {figs_dir.resolve()}")


if __name__ == "__main__":
    main()
