import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify

# Declaracao das variaveis
contador = 0
experiment_name = "AGR_a"

mlflow.set_experiment(experiment_name)

# Carrega o modelo
runs_df = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id])
first_run= runs_df.iloc[-1]
model = mlflow.sklearn.load_model("/app/mlruns/960876711757393375/" +str(first_run["run_id"]) + "/artifacts/model_agr_a")
#model = mlflow.sklearn.load_model("/home/miguel/Projeto/fraud_detection/mlruns/960876711757393375/" + str(first_run["run_id"]) + "/artifacts/model_agr_a")

app = Flask(__name__)

@app.route('/update', methods=['POST'])
def update():

    global contador, model

    # carrega os dados da requisicao e separa em X e y
    data = request.get_json(force=True)
    Xi = (data[0])
    yi = (data[1])

    # Faz a predicao e update modelo
    y_pred = model.predict_one(Xi)
    model = model.learn_one(Xi, yi)

    contador += 1

    if contador%10 == 0:        
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            #caminho = "/home/miguel/Projeto/fraud_detection/mlruns/960876711757393375/" +str(mlflow.active_run().info.run_id) + "/artifacts/model_agr_a"
            caminho = "/app/mlruns/960876711757393375/" +str(mlflow.active_run().info.run_id) + "/artifacts/model_agr_a"
            mlflow.sklearn.save_model(model, caminho, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

    return jsonify(float(y_pred))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)