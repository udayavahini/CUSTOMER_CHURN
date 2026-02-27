set -euo pipefail

DATASET_SLUG="blastchar/telco-customer-churn"
OUT_DIR="data"

mkdir -p "$OUT_DIR"
kaggle datasets download -d "$DATASET_SLUG" -p "$OUT_DIR" --unzip
ls -lah "$OUT_DIR"

from sklearn.utils import resample
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
    target_col = y.name if y.name is not None else "Churn"
    train_df = train_df.rename(columns={train_df.columns[-1]: target_col})

    df0 = train_df[train_df[target_col] == 0]
    df1 = train_df[train_df[target_col] == 1]

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

    y_train_bal = train_bal[target_col].astype(int)
    X_train_bal = train_bal.drop(columns=[target_col])

    return X_train_bal, X_test, y_train_bal, y_test

y = get_target(df_raw)
y.name = "Churn"  # so the helper keeps the label consistent

df_nod = drop_id(df_raw)
X = df_nod.drop(columns=["Churn"])

balanced_splits = {}
for ts in (0.70, 0.75, 0.80):
    X_trb, X_te, y_trb, y_te = split_and_balance_train(
        X, y, train_size=ts, random_state=RANDOM_STATE, method="oversample"
    )
    balanced_splits[ts] = (X_trb, X_te, y_trb, y_te)
    print(f"train_size={ts} balanced train counts:\n{y_trb.value_counts()}\n")