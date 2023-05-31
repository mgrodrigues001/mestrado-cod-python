import mlflow
import mlflow.sklearn

experiment_name = "river_ARF"

runs_df = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id])
last_run= runs_df.iloc[0]
print(last_run)
#model = mlflow.sklearn.load_model("/home/miguel/Projeto/fraud_detection/mlruns/927908605469128688/" + str(last_run["run_id"]) + "/artifacts/model_ARF")
