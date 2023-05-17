import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
from river import metrics
from mlflow.tracking import MlflowClient

# Declaracao das variaveis
contador = 0
experiment_name = "river_ARF"
metric_name = "metrics.roc_auc"
metric_ref = 0

app = Flask(__name__)

@app.route('/', methods=['POST'])
def update():

    global contador, metric_ref, auc_score, model, best_run

    # carrega os dados da requisicao e separa em X e y
    data = request.get_json(force=True)
    Xi = (data[0])
    yi = (data[1])

    # Com o contador == 0 deve-se buscar no MLFlow o modelo com o melhor valor da metrica ROCAUC
    if contador == 0:
        runs_df = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id])
        best_run = runs_df.loc[runs_df[metric_name].idxmax()]

        # Carrega o valor da metrica ROCAUC para ser usada como referencia
        cliente = MlflowClient()
        metricas = cliente.get_metric_history(run_id=best_run.run_id, key="roc_auc")
        metric_ref = metricas[-1].value 

        # Cria uma metrica zerada e carrega o modelo
        auc_score = metrics.ROCAUC()
        model = mlflow.sklearn.load_model("/app/mlruns/" + str(best_run.experiment_id) + "/" +str(best_run["run_id"]) + "/artifacts/model_ARF")
        #model = mlflow.sklearn.load_model("runs:/" + str(best_run["run_id"]) + "/model_ARF")

    # Faz a predicao e update modelo
    y_pred = model.predict_one(Xi)
    auc_score = auc_score.update(yi, y_pred)

    model = model.learn_one(Xi, yi)

    contador += 1

    ### Condicoes para salvar um novo modelo atualizado no MLFlow
    # O modelo no pode ser salvo no MLFlow antes de n atualizacoes
    # Quando o modelo atingir uma valor maior que a metrica de referencia
    # Apos um numero muito grande de interacoes e mesmo assim nao superou o valor da metrica de referencia
    if (auc_score.get() > metric_ref or contador >= 10000) and contador >= 1000:        
        # Salva o modelo e metrica no MLFlow
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_metric("roc_auc", auc_score.get())
            #mlflow.sklearn.log_model(model, "model_ARF")
            caminho = "/app/mlruns/" + str(best_run.experiment_id) + "/" +str(mlflow.active_run().info.run_id) + "/artifacts/model_ARF"
            mlflow.sklearn.save_model(model, caminho, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

        contador = 0
    
    return jsonify(f'Ref: {metric_ref:,.6f} -- Auc_score: {auc_score.get():,.6f} -- Contador: {contador} -- ID: {best_run.run_id}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)