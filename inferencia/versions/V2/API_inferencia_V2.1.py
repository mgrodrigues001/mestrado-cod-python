import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify

experiment_name = "AGR_a"
contador = 0

runs_df = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id])
last_run= runs_df.iloc[0]
model = mlflow.sklearn.load_model("/app/mlruns/960876711757393375/" +str(last_run["run_id"]) + "/artifacts/model_agr_a")
#model = mlflow.sklearn.load_model("/home/miguel/Projeto/fraud_detection/mlruns/960876711757393375/" + str(last_run["run_id"]) + "/artifacts/model_agr_a")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    global contador, model, last_run

    data = request.get_json(force=True)
    contador += 1

    if contador%5000 == 0:        
        runs_df = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id])
        last_run= runs_df.iloc[0]
        model = mlflow.sklearn.load_model("/app/mlruns/960876711757393375/" +str(last_run["run_id"]) + "/artifacts/model_agr_a")
        #model = mlflow.sklearn.load_model("/home/miguel/Projeto/fraud_detection/mlruns/960876711757393375/" + str(last_run["run_id"]) + "/artifacts/model_agr_a")
        print(last_run["run_id"])

    y_pred = model.predict_one(data)
    
    return jsonify(float(y_pred))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)