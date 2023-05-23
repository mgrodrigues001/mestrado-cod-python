import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify

# Declaracao das variaveis
contador = 0
experiment_name = "river_ARF"

mlflow.set_experiment(experiment_name)

# Carrega o modelo
runs_df = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id])
first_run= runs_df.iloc[-1]
model = mlflow.sklearn.load_model("/app/mlruns/927908605469128688/" +str(first_run["run_id"]) + "/artifacts/model_ARF")
#model = mlflow.sklearn.load_model("/home/miguel/Projeto/fraud_detection/mlruns/927908605469128688/" + str(first_run["run_id"]) + "/artifacts/model_ARF")

app = Flask(__name__)

@app.route('/update', methods=['POST'])
def update():

    global contador, model

    # carrega os dados da requisicao e separa em X e y
    data = request.get_json(force=True)
    Xi = (data[0])
    yi = (data[1])

    # Faz o update modelo
    model = model.learn_one(Xi, yi)
    contador += 1
    
    return jsonify(f'Contador: {contador} -- {str(first_run["run_id"])}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)