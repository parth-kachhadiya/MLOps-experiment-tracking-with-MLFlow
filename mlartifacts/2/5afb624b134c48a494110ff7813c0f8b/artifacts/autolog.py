"""
    Working locally + auto log
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # for local experiment tracking on mlflow

wine = load_wine()

X = wine.data
Y = wine.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 34)

max_depth = 1
n_estimator = 2

mlflow.set_experiment("from-youtube-experiments")

mlflow.autolog()
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimator)
    rf.fit(x_train, y_train)

    prediction = rf.predict(x_test)

    # Not part of autolog, requires manual logging
    mlflow.log_artifact(__file__)

    # Adding tags
    mlflow.set_tags({'Author' : "Parth Kachhadiya", 'Project' : 'MLOps-experiment-tracking-mlflow'})

    # Logging models
    mlflow.sklearn.log_model(rf, "Random Forest Cassifier (M)")
