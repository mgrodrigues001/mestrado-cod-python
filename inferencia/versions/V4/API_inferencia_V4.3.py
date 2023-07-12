import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
import threading
import requests

experiment_name = "AGR_a"
taxa_load = 1000
url = 'http://10.32.2.212:32002/update'
fila_envio = []

runs_df = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id])
last_run= runs_df.iloc[0]
model = mlflow.sklearn.load_model("/app/mlruns/960876711757393375/" +str(last_run["run_id"]) + "/artifacts/model_agr_a")


def envia():
    global fila_envio, url
    while True:
        if len(fila_envio) != 0:
            item = fila_envio.pop(0)
            feature = item[0]
            label = item[1]
            r = requests.post(url,json=(feature, label))

def carrega():
    global model, last_run
    while True:
        runs = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id])
        run= runs.iloc[0]
        if (last_run["run_id"]) != (run["run_id"]):
            model = mlflow.sklearn.load_model("/app/mlruns/960876711757393375/" + str(run["run_id"]) + "/artifacts/model_agr_a")
            last_run = run


threading.Thread(target=envia).start()
threading.Thread(target=carrega).start()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    global model, fila_envio, taxa_load

    data = request.get_json(force=True)
    Xi = (data[0])
    yi = (data[1])

    y_pred = model.predict_one(Xi)

    fila_envio.append([Xi,yi])

    return jsonify(float(y_pred))

@app.route('/load', methods=['POST'])
def load():
    global model

    run_id = request.get_json(force=True)
    model = mlflow.sklearn.load_model("/app/mlruns/960876711757393375/" + str(run_id) + "/artifacts/model_agr_a")

    return jsonify(f'{model}')

@app.route('/taxa', methods=['POST'])
def taxa():
    global taxa_load

    taxa_load = request.get_json(force=True)

    return jsonify(taxa_load)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)