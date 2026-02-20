set -euo pipefail

DATASET_SLUG="blastchar/telco-customer-churn"
OUT_DIR="data"

mkdir -p "$OUT_DIR"
kaggle datasets download -d "$DATASET_SLUG" -p "$OUT_DIR" --unzip
ls -lah "$OUT_DIR"
