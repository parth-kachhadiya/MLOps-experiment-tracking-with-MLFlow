"""
    Working with Dagshub + Manual logging
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub


# ------------- setup -------------
dagshub.init(repo_owner='parthkachhadiya04', repo_name='MLOps-experiment-tracking-with-MLFlow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/parthkachhadiya04/MLOps-experiment-tracking-with-MLFlow.mlflow")
# ---------------------------------

wine = load_wine()

X = wine.data
Y = wine.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 34)

max_depth = 4
n_estimator = 40

# Mention your `experiment location` here, or mention experiment_id in `with mlflow.start_run(experiment_id=<id>)`
# If mentioned experiment doesn't exist, it will automatically create new one.
mlflow.set_experiment("from-youtube-experiments")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimator)
    rf.fit(x_train, y_train)

    prediction = rf.predict(x_test)

    accuracy = accuracy_score(y_test, prediction)

    # Logging matrix
    mlflow.log_metric('accuracy', accuracy)

    # Logging parameters
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimator)

    # Logging a confusion matrix plot
    cm = confusion_matrix(y_test, prediction)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plt.savefig("Confusion-matrix.png")

    # Logging artifacts using mlflow
    mlflow.log_artifact("Confusion-matrix.png")

    # Logging current file code
    mlflow.log_artifact(__file__)

    # Adding tags
    mlflow.set_tags({'Author' : "Parth Kachhadiya", 'Project' : 'MLOps-experiment-tracking-mlflow'})

    # Logging models
    mlflow.sklearn.log_model(rf, "Random Forest Cassifier (M)")

    print(f"Accuracy : {accuracy}")
