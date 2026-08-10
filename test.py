import mlflow
print("Tracking old URI schema : ")
print(mlflow.get_tracking_uri(), end="\n")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

print("Tracking new URI schema : ")
print(mlflow.get_tracking_uri(), end="\n")