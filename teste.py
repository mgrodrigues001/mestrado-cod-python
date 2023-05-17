import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
#from river import metrics
#from mlflow.tracking import MlflowClient

experiment_name = "river_ARF"
metric_name = "metrics.roc_auc"
runs_df = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id])
best_run = runs_df.loc[runs_df[metric_name].idxmax()]
#model = mlflow.sklearn.load_model("runs:/" + str(best_run["run_id"]) + "/model_ARF")
model = mlflow.sklearn.load_model("/home/miguel/Projeto/fraud_detection/mlruns/" + str(best_run.experiment_id) + "/" +str(best_run["run_id"]) + "/artifacts/model_ARF")
#print("/home/miguel/Projeto/fraud_detection/mlruns/" + str(best_run["run_id"]) + "/model_ARF")

print('')
print("/home/miguel/Projeto/fraud_detection/mlruns/" + str(best_run.experiment_id) + "/" +str(best_run["run_id"]) + "/artifacts/model_ARF")
print('')
print(str(best_run.artifact_uri))

print(model)

mlflow.set_experiment(experiment_name)
with mlflow.start_run():

    print(mlflow.active_run().info.run_id)
    #mlflow.log_metric("roc_auc", auc_score.get())
    #mlflow.sklearn.log_model(model, "model_ARF")
    caminho = "/app/mlruns/" + str(best_run.experiment_id) + "/" +str(mlflow.active_run().info.run_id) + "/artifacts/model_ARF"
    mlflow.sklearn.save_model(model, caminho, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

