import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================
# ARGUMENT PARSER
# =====================
parser = argparse.ArgumentParser(description="Train RandomForest for Obesity Classification")
parser.add_argument(
    "--data_path",
    type=str,
    default="obesity_clean.csv",
    help="Path ke dataset CSV"
)
args = parser.parse_args()
DATA_PATH = args.data_path

# =====================
# MLFLOW CONFIG
# =====================
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("Obesity_Classification_RF")

# =====================
# LOAD DATASET
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
full_data_path = os.path.join(BASE_DIR, DATA_PATH)
print("Mencari dataset di:", full_data_path)

df = pd.read_csv(full_data_path)

# =====================
# SPLIT FEATURE & TARGET
# =====================
TARGET_COLUMN = "Label"
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# =====================
# SPLIT TRAIN - TEST
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================
# TRAINING
# =====================
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)

# =====================
# PREDICTION
# =====================
y_pred = rf_model.predict(X_test)

# =====================
# METRICS
# =====================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted")
rec = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# =====================
# LOGGING (TANPA start_run)
# =====================
mlflow.log_param("model_type", "RandomForest")
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 10)
mlflow.log_param("data_path", DATA_PATH)

mlflow.log_metric("accuracy", acc)
mlflow.log_metric("precision", prec)
mlflow.log_metric("recall", rec)
mlflow.log_metric("f1_score", f1)

mlflow.sklearn.log_model(
    rf_model,
    artifact_path="random_forest_model"
)

print("Training selesai & logged ke MLflow")
