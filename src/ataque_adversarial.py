import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTADOS_DIR = BASE_DIR / "resultados"

RESULTADOS_DIR.mkdir(exist_ok=True)

# Configurações do experimento
SEMENTE = 42
AMOSTRAS = 2000
INTENSIDADES = [0.05, 0.10, 0.20, 0.30]

# Características numéricas que serão manipuladas
CARACTERISTICAS = ["dur", "sbytes", "dbytes", "rate", "sload", "dload"]

print("=== ATAQUE ADVERSARIAL ===")

# 1. Carregar modelo treinado
modelo = joblib.load(RESULTADOS_DIR / "modelo_random_forest.pkl")

# 2. Carregar dados
treino = pd.read_csv(DATA_DIR / "UNSW_NB15_training-set.csv")

teste = pd.read_csv(DATA_DIR / "UNSW_NB15_testing-set.csv")

colunas_remover = ["id", "attack_cat", "label"]

# Selecionar somente registros de ataques reais
ataques = teste[teste["label"] == 1].copy()

# Amostragem reproduzível
ataques = ataques.sample(n=min(AMOSTRAS, len(ataques)), random_state=SEMENTE)

X_original = ataques.drop(columns=colunas_remover)

# 3. Previsões originais
previsoes_originais = modelo.predict(X_original)

detectados = previsoes_originais == 1

print(f"Registros maliciosos: {len(X_original)}")
print(f"Detectados inicialmente: {detectados.sum()}")

# Taxa de detecção antes do ataque
recall_original = np.mean(previsoes_originais == 1)

# 4. Calcular limites a partir do treinamento
# Não utilizamos o conjunto de teste para definir
# os intervalos de manipulação.
limites = {}

for coluna in CARACTERISTICAS:
    valores = treino[coluna]

    limites[coluna] = {
        "min": float(valores.quantile(0.01)),
        "max": float(valores.quantile(0.99)),
        "escala": float(valores.quantile(0.75) - valores.quantile(0.25)),
    }


# 5. Função de manipulação adversarial
def modificar_registros(dados, intensidade, rng):
    modificados = dados.copy()

    for coluna in CARACTERISTICAS:
        info = limites[coluna]

        # Perturbação proporcional ao intervalo
        # interquartil da característica
        escala = info["escala"]

        if escala == 0:
            continue

        ruido = rng.uniform(-intensidade, intensidade, size=len(modificados)) * escala

        valores = modificados[coluna].to_numpy(dtype=float)

        novos_valores = valores + ruido

        # Restringir a valores não negativos
        novos_valores = np.maximum(novos_valores, 0)

        # Limitar a alteração por amostra
        # e não o valor original absoluto
        novos_valores = np.clip(
            novos_valores,
            valores - intensidade * escala,
            valores + intensidade * escala,
        )

        modificados[coluna] = novos_valores

    return modificados


# 6. Executar experimentos
resultados = []
rng = np.random.default_rng(SEMENTE)

for intensidade in INTENSIDADES:
    print(f"\nTestando intensidade: {intensidade:.0%}")

    X_modificado = modificar_registros(X_original, intensidade, rng)

    previsoes = modelo.predict(X_modificado)

    # Recall entre todos os registros maliciosos
    recall = np.mean(previsoes == 1)

    # Sucesso entre os ataques inicialmente detectados
    sucesso = np.mean(previsoes[detectados] == 0) if detectados.sum() > 0 else 0.0

    print(f"Recall após manipulação: {recall:.4f}")
    print(f"Taxa de evasão: {sucesso:.4f}")

    resultados.append(
        {
            "intensidade": intensidade,
            "recall": float(recall),
            "taxa_evasao": float(sucesso),
        }
    )

# 7. Salvar resultados
dados_resultados = {
    "amostras": len(X_original),
    "detectados_inicialmente": int(detectados.sum()),
    "recall_original": float(recall_original),
    "caracteristicas": CARACTERISTICAS,
    "experimentos": resultados,
}

with open(RESULTADOS_DIR / "resultados_ataque.json", "w", encoding="utf-8") as arquivo:
    json.dump(dados_resultados, arquivo, indent=4)

# 8. Gerar gráfico
intensidades = [r["intensidade"] * 100 for r in resultados]

taxas = [r["taxa_evasao"] * 100 for r in resultados]

plt.figure(figsize=(8, 5))
plt.plot(intensidades, taxas, marker="o")
plt.xlabel("Intensidade da perturbação (%)")
plt.ylabel("Taxa de evasão (%)")
plt.title("Evasão adversarial - Random Forest")
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(RESULTADOS_DIR / "grafico_ataque.png", dpi=300)
plt.close()

print("\nExperimentos concluídos!")
print("Resultados salvos na pasta resultados.")
