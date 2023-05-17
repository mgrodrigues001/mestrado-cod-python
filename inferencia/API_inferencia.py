import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify

contador = 0

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    global contador, model, best_run

    data = request.get_json(force=True)

    if contador == 0:
        # Seleciona o melhor modelo
        experiment_name = "river_ARF"
        metric_name = "metrics.roc_auc"
        runs_df = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id])
        best_run = runs_df.loc[runs_df[metric_name].idxmax()]
        model = mlflow.sklearn.load_model("/app/mlruns/" + str(best_run.experiment_id) + "/" +str(best_run["run_id"]) + "/artifacts/model_ARF")
        #model = mlflow.sklearn.load_model("runs:/" + str(best_run["run_id"]) + "/model_ARF")
        
    y_pred = model.predict_one(data)
    contador += 1

    # Apos n iteracoes zera o contador carregar o novo modelo na proxima iteracao
    if contador > 20000:
        contador = 0
    
    #return jsonify(int(y_pred))
    return jsonify(f'Pred: {y_pred} -- Run_id: {best_run["run_id"]} -- Contador: {contador}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)