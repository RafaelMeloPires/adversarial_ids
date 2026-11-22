"""Etapa 4: treinamento adversarial e avaliação comparativa (UNSW-NB15).

Treina um segundo Random Forest com exemplos maliciosos perturbados,
utilizando a MESMA família de perturbações da etapa 3. Compara o modelo
original com o defendido em dados limpos e em ataques sintéticos idênticos.

Os CSVs de teste não são usados no treinamento nem nos limites das alterações.
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Permite salvar gráficos sem janela gráfica.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTADOS_DIR = BASE_DIR / "resultados"

# Configurações fixas para permitir reprodução do experimento.
SEMENTE_TESTE = 42                   # Igual à etapa 3.
SEMENTE_TREINO = 2026               # Separada da semente do teste.
AMOSTRAS_TESTE = 2000               # Igual à etapa 3.
FRACAO_ATAQUES_TREINO = 0.20         # 20% dos ataques do treino são selecionados.
INTENSIDADES_AUMENTO = [0.10, 0.30] # Duas novas cópias por ataque selecionado.
INTENSIDADES_TESTE = [0.05, 0.10, 0.20, 0.30]  # Igual à etapa 3.
CARACTERISTICAS = ["dur", "sbytes", "dbytes", "rate", "sload", "dload"]
COLUNAS_REMOVER = ["id", "attack_cat", "label"]


def calcular_escalas(treino: pd.DataFrame) -> dict[str, float]:
    """Calcula IQRs a partir do TREINO, assim como na etapa 3."""
    return {
        coluna: float(treino[coluna].quantile(0.75) - treino[coluna].quantile(0.25))
        for coluna in CARACTERISTICAS
    }


def modificar_registros(
    dados: pd.DataFrame,
    intensidade: float,
    rng: np.random.Generator,
    escalas: dict[str, float],
) -> pd.DataFrame:
    """Replica a modificação de atributos numéricos da etapa 3.

    A perturbação é aleatória (não otimizada contra o classificador).
    É um experimento no espaço de características, não tráfego executável.
    """
    modificados = dados.copy()

    for coluna in CARACTERISTICAS:
        escala = escalas[coluna]
        if escala == 0:
            continue

        valores = modificados[coluna].to_numpy(dtype=float)
        ruido = rng.uniform(-intensidade, intensidade, size=len(modificados)) * escala
        novos_valores = valores + ruido
        novos_valores = np.maximum(novos_valores, 0)
        novos_valores = np.clip(
            novos_valores,
            valores - intensidade * escala,
            valores + intensidade * escala,
        )
        modificados[coluna] = novos_valores

    return modificados


def medir_classificacao(y_verdadeiro: pd.Series, previsoes: np.ndarray) -> dict:
    """Métricas binárias: 0 = normal, 1 = ataque."""
    return {
        "acuracia": float(accuracy_score(y_verdadeiro, previsoes)),
        "precisao": float(precision_score(y_verdadeiro, previsoes, zero_division=0)),
        "recall": float(recall_score(y_verdadeiro, previsoes, zero_division=0)),
        "f1_score": float(f1_score(y_verdadeiro, previsoes, zero_division=0)),
        "matriz_confusao": confusion_matrix(y_verdadeiro, previsoes, labels=[0, 1]).tolist(),
    }


def salvar_grafico(pontos: list[dict], destino: Path) -> None:
    intensidades = [0] + [r["intensidade"] * 100 for r in pontos]
    original = [pontos[0]["recall_amostra_original"] * 100] + [
        r["recall_modelo_original"] * 100 for r in pontos
    ]
    defesa = [pontos[0]["recall_amostra_defesa"] * 100] + [
        r["recall_modelo_defesa"] * 100 for r in pontos
    ]

    plt.figure(figsize=(9, 5.5))
    plt.plot(intensidades, original, marker="o", label="Modelo original")
    plt.plot(intensidades, defesa, marker="s", label="Modelo com defesa")
    plt.xlabel("Intensidade de perturbação (%)")
    plt.ylabel("Recall dos registros maliciosos (%)")
    plt.title("Detecção de ataques: modelo original vs. defesa")
    plt.xticks(intensidades)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destino, dpi=300)
    plt.close()


def main() -> None:
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    caminho_modelo = RESULTADOS_DIR / "modelo_random_forest.pkl"
    if not caminho_modelo.is_file():
        raise SystemExit(
            f"Modelo original não encontrado: {caminho_modelo}\n"
            "Execute antes: python src/treinar_modelo.py\n"
            "O modelo não foi incluído no ZIP porque a pasta resultados estava ignorada."
        )

    treino_csv = DATA_DIR / "UNSW_NB15_training-set.csv"
    teste_csv = DATA_DIR / "UNSW_NB15_testing-set.csv"
    for caminho in (treino_csv, teste_csv):
        if not caminho.is_file():
            raise SystemExit(f"Dataset ausente: {caminho}")

    print("=== ETAPA 4: TREINAMENTO ADVERSARIAL ===", flush=True)
    # joblib/pickle: carregue apenas modelos gerados por você ou fontes confiáveis.
    modelo_original = joblib.load(caminho_modelo)
    treino = pd.read_csv(treino_csv)
    teste = pd.read_csv(teste_csv)
    X_treino = treino.drop(columns=COLUNAS_REMOVER)
    y_treino = treino["label"]
    X_teste = teste.drop(columns=COLUNAS_REMOVER)
    y_teste = teste["label"]

    escalas = calcular_escalas(treino)

    # Somente exemplos maliciosos do treinamento geram exemplos adicionais.
    ataques_treino = X_treino.loc[y_treino == 1]
    qtd_augmentacao = max(1, round(len(ataques_treino) * FRACAO_ATAQUES_TREINO))
    amostra_treino = ataques_treino.sample(n=qtd_augmentacao, random_state=SEMENTE_TREINO)
    rng_treino = np.random.default_rng(SEMENTE_TREINO)

    partes_X = [X_treino]
    partes_y = [y_treino]
    for intensidade in INTENSIDADES_AUMENTO:
        partes_X.append(modificar_registros(amostra_treino, intensidade, rng_treino, escalas))
        partes_y.append(pd.Series(np.ones(qtd_augmentacao, dtype=np.int8)))

    X_aumentado = pd.concat(partes_X, ignore_index=True)
    y_aumentado = pd.concat(partes_y, ignore_index=True)
    print(f"Treinamento limpo: {len(X_treino):,} registros")
    print(f"Ataques do treino selecionados: {qtd_augmentacao:,}")
    print(f"Treinamento aumentado: {len(X_aumentado):,} registros")
    print(f"Intensidades de aumento: {INTENSIDADES_AUMENTO}")

    # Clonagem preserva a arquitetura/hyperparâmetros do modelo da etapa 2,
    # mas cria um novo estimador SEM reutilizar o ajuste anterior.
    modelo_defendido = clone(modelo_original)
    print("\nTreinando modelo com defesa...", flush=True)
    inicio = time.perf_counter()
    modelo_defendido.fit(X_aumentado, y_aumentado)
    tempo = time.perf_counter() - inicio
    print(f"Treinamento concluído em {tempo:.1f} s.", flush=True)

    # Medição limpa: MESMO conjunto completo de teste para ambos os modelos.
    print("\nAvaliando os dois modelos no conjunto de teste limpo...", flush=True)
    metricas_limpo_original = medir_classificacao(y_teste, modelo_original.predict(X_teste))
    metricas_limpo_defesa = medir_classificacao(y_teste, modelo_defendido.predict(X_teste))

    # Mesmos 2 mil ataques da etapa 3; mesma ordem de ruídos (semente 42).
    amostra_ataques = teste.loc[teste["label"] == 1].sample(
        n=min(AMOSTRAS_TESTE, int((teste["label"] == 1).sum())),
        random_state=SEMENTE_TESTE,
    )
    X_ataques = amostra_ataques.drop(columns=COLUNAS_REMOVER)
    pred_original_limpo = modelo_original.predict(X_ataques)
    pred_defesa_limpo = modelo_defendido.predict(X_ataques)
    detectados_inicial_original = pred_original_limpo == 1
    detectados_inicial_defesa = pred_defesa_limpo == 1
    recall_amostra_original = float(np.mean(detectados_inicial_original))
    recall_amostra_defesa = float(np.mean(detectados_inicial_defesa))

    rng_teste = np.random.default_rng(SEMENTE_TESTE)
    pontos = []
    print("\nComparação sob perturbações iguais:", flush=True)
    for intensidade in INTENSIDADES_TESTE:
        X_modificado = modificar_registros(X_ataques, intensidade, rng_teste, escalas)
        pred_original = modelo_original.predict(X_modificado)
        pred_defesa = modelo_defendido.predict(X_modificado)

        # Ambos os recalls usam EXATAMENTE os mesmos ataques verdadeiros.
        recall_original = float(np.mean(pred_original == 1))
        recall_defesa = float(np.mean(pred_defesa == 1))
        # Taxa de evasão restrita às detecções iniciais DE CADA modelo.
        evasao_original = (
            float(np.mean(pred_original[detectados_inicial_original] == 0))
            if detectados_inicial_original.any() else 0.0
        )
        evasao_defesa = (
            float(np.mean(pred_defesa[detectados_inicial_defesa] == 0))
            if detectados_inicial_defesa.any() else 0.0
        )

        pontos.append({
            "intensidade": intensidade,
            "recall_amostra_original": recall_amostra_original,
            "recall_amostra_defesa": recall_amostra_defesa,
            "recall_modelo_original": recall_original,
            "recall_modelo_defesa": recall_defesa,
            "delta_recall_pontos_percentuais": float((recall_defesa - recall_original) * 100),
            "taxa_evasao_modelo_original": evasao_original,
            "taxa_evasao_modelo_defesa": evasao_defesa,
        })
        print(
            f"  {intensidade:>4.0%} | recall original: {recall_original:.4f}"
            f" | recall defesa: {recall_defesa:.4f}"
            f" | diferença: {(recall_defesa-recall_original)*100:+.2f} pp",
            flush=True,
        )

    print("\nConjunto de teste limpo:")
    for nome, met in [("Original", metricas_limpo_original), ("Defesa", metricas_limpo_defesa)]:
        print(
            f"  {nome}: acurácia={met['acuracia']:.4f}"
            f" precisão={met['precisao']:.4f} recall={met['recall']:.4f}"
            f" F1={met['f1_score']:.4f}"
        )

    dados_resultados = {
        "descricao": "Comparação empírica da robustez a ruídos sintéticos no espaço de características",
        "configuracao": {
            "semente_teste": SEMENTE_TESTE,
            "semente_treino": SEMENTE_TREINO,
            "amostras_teste_maliciosas": len(X_ataques),
            "registros_treino_originais": len(X_treino),
            "ataques_treino_selecionados": qtd_augmentacao,
            "registros_treino_aumentados": len(X_aumentado),
            "fracao_ataques_treino": FRACAO_ATAQUES_TREINO,
            "intensidades_aumento": INTENSIDADES_AUMENTO,
            "intensidades_teste": INTENSIDADES_TESTE,
            "caracteristicas_manipuladas": CARACTERISTICAS,
            "tempo_treinamento_segundos": float(tempo),
        },
        "teste_limpo_completo": {
            "modelo_original": metricas_limpo_original,
            "modelo_defesa": metricas_limpo_defesa,
        },
        "teste_ataques_manipulados": {
            "recall_inicial_modelo_original": recall_amostra_original,
            "recall_inicial_modelo_defesa": recall_amostra_defesa,
            "detectados_inicialmente_modelo_original": int(detectados_inicial_original.sum()),
            "detectados_inicialmente_modelo_defesa": int(detectados_inicial_defesa.sum()),
            "experimentos": pontos,
        },
        "limitacoes": [
            "Perturbações aleatórias em seis atributos; não há otimização contra a predição.",
            "Não se garante consistência entre atributos correlacionados nem viabilidade como pacotes reais.",
            "A taxa de evasão usa apenas os ataques inicialmente detectados por cada modelo; os denominadores podem diferir.",
        ],
    }

    caminho_json = RESULTADOS_DIR / "resultados_defesa.json"
    with caminho_json.open("w", encoding="utf-8") as arq:
        json.dump(dados_resultados, arq, indent=4, ensure_ascii=False)

    caminho_grafico = RESULTADOS_DIR / "comparacao_defesa.png"
    salvar_grafico(pontos, caminho_grafico)
    caminho_defesa = RESULTADOS_DIR / "modelo_random_forest_defesa.pkl"
    joblib.dump(modelo_defendido, caminho_defesa)

    print("\nArquivos gerados:")
    for p in (caminho_json, caminho_grafico, caminho_defesa):
        print(" -", p.relative_to(BASE_DIR))
    print("Etapa 4 concluída.")


if __name__ == "__main__":
    main()
