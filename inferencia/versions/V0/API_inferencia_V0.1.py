import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify

experiment_name = "river_ARF"
contador = 0

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    global contador, model, last_run

    data = request.get_json(force=True)

    if contador == 0:
        runs_df = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id])
        last_run= runs_df.iloc[0]
        model = mlflow.sklearn.load_model("/app/mlruns/927908605469128688/" +str(last_run["run_id"]) + "/artifacts/model_ARF")
        #model = mlflow.sklearn.load_model("/home/miguel/Projeto/fraud_detection/mlruns/927908605469128688/" + str(last_run["run_id"]) + "/artifacts/model_ARF")
        
    y_pred = model.predict_one(data)
    contador += 1
    
    return jsonify(int(y_pred))
    #return jsonify(f'Pred: {y_pred} -- Run_id: {last_run["run_id"]} -- Contador: {contador}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)