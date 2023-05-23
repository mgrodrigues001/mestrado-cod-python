import pandas as pd
from river.ensemble import AdaptiveRandomForestClassifier
from river import metrics
from river import stream
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import pickle

# Carregar dataset
df = pd.read_csv("/home/miguel/Projeto/fraud_detection/datasets/df_bal_train.csv")

# Separando DataFrame em X e y
X = pd.DataFrame(df)
y = X.pop('Class')

# treinado o modelo
auc_score = metrics.ROCAUC(n_thresholds=20)
metric_dist_plot = []
n = 0
numero = []

# Cria um objeto de classificador ARF
model = AdaptiveRandomForestClassifier(n_models=5, seed=42)

for xi, yi in stream.iter_pandas(X, y):
    y_pred = model.predict_one(xi)
    model = model.learn_one(xi, yi)
    if y_pred != None:
        print(f'y_pred: {y_pred} -- yi: {yi}')
        auc_score = auc_score.update(yi, y_pred)
        metric_dist_plot.append(auc_score.get())
        n += 1
        numero.append(n)

# Plot da metrica AUC
plt.plot(numero, metric_dist_plot)
plt.title("Evolucao da Metrica", fontsize = 10)
plt.xlabel("N instancias")
plt.ylabel("AUC")
plt.show()

# Salvando o modelo e a metrica no MLFlow
mlflow.set_experiment("river_ARF")
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model_ARF")
    mlflow.log_metric("roc_auc", auc_score.get())
    #mlflow.log_artifact("roc_auc.pkl")
    print("Modelo: ", mlflow.active_run().info.run_uuid) 
mlflow.end_run()