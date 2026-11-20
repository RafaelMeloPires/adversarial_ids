import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Caminhos do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTADOS_DIR = BASE_DIR / "resultados"

RESULTADOS_DIR.mkdir(exist_ok=True)

# 1. Carregamento dos dados
print("Carregando dataset...")

treino = pd.read_csv(DATA_DIR / "UNSW_NB15_training-set.csv")
teste = pd.read_csv(DATA_DIR / "UNSW_NB15_testing-set.csv")

# 2. Separação das características e rótulos
colunas_remover = ["id", "attack_cat", "label"]

X_treino = treino.drop(columns=colunas_remover)
y_treino = treino["label"]

X_teste = teste.drop(columns=colunas_remover)
y_teste = teste["label"]

# 3. Identificação das variáveis categóricas
categoricas = X_treino.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

print("Colunas categóricas:", categoricas)

# 4. Pré-processamento
preprocessador = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            categoricas,
        )
    ],
    remainder="passthrough",
)

# 5. Criação do modelo
modelo = Pipeline(
    steps=[
        ("preprocessamento", preprocessador),
        (
            "classificador",
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        ),
    ]
)

# 6. Treinamento
print("\nIniciando treinamento...")
inicio = time.time()

modelo.fit(X_treino, y_treino)

tempo = time.time() - inicio
print(f"Treinamento concluído em {tempo:.2f} segundos.")

# 7. Previsões no conjunto de teste
print("\nAvaliando o modelo...")
previsoes = modelo.predict(X_teste)

# 8. Cálculo das métricas
acuracia = accuracy_score(y_teste, previsoes)
precisao = precision_score(y_teste, previsoes)
recall = recall_score(y_teste, previsoes)
f1 = f1_score(y_teste, previsoes)

print("\n=== RESULTADOS DO MODELO ===")
print(f"Acurácia: {acuracia:.4f}")
print(f"Precisão: {precisao:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nRelatório de classificação:")
print(classification_report(y_teste, previsoes, target_names=["Normal", "Ataque"]))

# 9. Matriz de confusão
matriz = confusion_matrix(y_teste, previsoes, labels=[0, 1])

visualizacao = ConfusionMatrixDisplay(
    confusion_matrix=matriz, display_labels=["Normal", "Ataque"]
)

visualizacao.plot(cmap="Blues", values_format="d")
plt.title("Matriz de Confusão - Random Forest")
plt.tight_layout()

plt.savefig(RESULTADOS_DIR / "matriz_confusao.png", dpi=300)
plt.close()

# 10. Salvamento das métricas
metricas = {
    "acuracia": float(acuracia),
    "precisao": float(precisao),
    "recall": float(recall),
    "f1_score": float(f1),
    "tempo_treinamento_segundos": float(tempo),
    "matriz_confusao": matriz.tolist(),
}

with open(RESULTADOS_DIR / "metricas.json", "w", encoding="utf-8") as arquivo:
    json.dump(metricas, arquivo, indent=4)

# 11. Salvamento do modelo completo
joblib.dump(modelo, RESULTADOS_DIR / "modelo_random_forest.pkl")

print("\nModelo e resultados salvos com sucesso!")
