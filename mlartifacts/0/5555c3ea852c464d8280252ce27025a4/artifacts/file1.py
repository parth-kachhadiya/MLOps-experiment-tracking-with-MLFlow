import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

wine = load_wine()

X = wine.data
Y = wine.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 34)

max_depth = 7
n_estimator = 20


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

    print(f"Accuracy : {accuracy}")
