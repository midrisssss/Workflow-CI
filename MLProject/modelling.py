import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "winequality_preprocessing.csv")

df = pd.read_csv(csv_path)
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("CI_Retraining_Workflow")

with mlflow.start_run() as run:
    n_estimators = 100
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model", 
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print(f"CI Retraining Completed. Run ID: {run.info.run_id}, Accuracy: {acc:.4f}")