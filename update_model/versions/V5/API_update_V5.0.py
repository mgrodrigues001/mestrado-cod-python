import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
import threading
import requests

# Declaracao das variaveis
contador = 0
experiment_name = "AGR_a"
taxa_load = 1000


mlflow.set_experiment(experiment_name)

# Carrega o modelo
runs_df = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id])
first_run= runs_df.iloc[-1]
model = mlflow.sklearn.load_model("/app/mlruns/960876711757393375/" + str(first_run["run_id"]) + "/artifacts/model_agr_a")

def salva(modelo):
    global experiment_name, list_pods
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(nested=True):
        caminho = "/app/mlruns/960876711757393375/" +str(mlflow.active_run().info.run_id) + "/artifacts/model_agr_a"
        mlflow.sklearn.save_model(modelo, caminho, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
        
        modelo_id = str(mlflow.active_run().info.run_id)
        for i in list_pods:
            r = requests.post(i,json=(modelo_id))

app = Flask(__name__)

@app.route('/update', methods=['POST'])
def update():
    global contador, model, taxa_load

    # carrega os dados da requisicao e separa em X e y
    data = request.get_json(force=True)
    Xi = (data[0])
    yi = (data[1])

    model = model.learn_one(Xi, yi)

    contador += 1

    if contador%taxa_load == 0:
        threading.Thread(target=salva, args=(model,)).start()

    return jsonify('ok')

@app.route('/taxa', methods=['POST'])
def taxa():
    global taxa_load
    taxa_load = request.get_json(force=True)
    return jsonify(taxa_load)

@app.route('/cont', methods=['POST'])
def cont():
    global contador
    contador = request.get_json(force=True)
    return jsonify(contador)

@app.route('/pods', methods=['POST'])
def pods():
    global list_pods
    list_pods = request.get_json(force=True)
    return jsonify(list_pods)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)