"""Métricas com denominadores explícitos e resumos de repetições."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def medir_classificacao(y, pred):
    matriz = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in matriz.ravel())
    return {
        "amostras": len(y), "acuracia": float(accuracy_score(y, pred)),
        "precisao": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "taxa_falsos_positivos": fp / (fp + tn) if fp + tn else None,
        "taxa_falsos_negativos": fn / (fn + tp) if fn + tp else None,
        "matriz_confusao": matriz.tolist(),
    }


def medir_evasao(pred_limpo, pred_ataque, elegiveis=None):
    pred_limpo, pred_ataque = np.asarray(pred_limpo), np.asarray(pred_ataque)
    if pred_limpo.shape != pred_ataque.shape or not len(pred_limpo):
        raise ValueError("Predições devem ser pareadas e não vazias")
    detectados = pred_limpo == 1
    if elegiveis is None:
        elegiveis = detectados
    elegiveis = np.asarray(elegiveis, dtype=bool)
    if elegiveis.shape != detectados.shape or (elegiveis & ~detectados).any():
        raise ValueError("Elegíveis devem ser ataques inicialmente detectados")
    denominador = int(elegiveis.sum())
    evadiram = int(((pred_ataque == 0) & elegiveis).sum())
    return {
        "amostras_maliciosas": len(pred_limpo),
        "detectados_antes": int(detectados.sum()), "detectados_depois": int((pred_ataque == 1).sum()),
        "recall_antes": float(detectados.mean()), "recall": float((pred_ataque == 1).mean()),
        "queda_recall_pp": float((detectados.mean() - (pred_ataque == 1).mean()) * 100),
        "evadiram": evadiram, "denominador_evasao": denominador,
        "taxa_evasao": evadiram / denominador if denominador else None,
        "recuperados": int(((pred_limpo == 0) & (pred_ataque == 1)).sum()),
    }


def resumir_repeticoes(experimentos, campos):
    resumos = []
    for intensidade in sorted({e["intensidade"] for e in experimentos}):
        grupo = [e for e in experimentos if e["intensidade"] == intensidade]
        item = {"intensidade": intensidade, "repeticoes": len(grupo)}
        for campo in campos:
            valores = np.array([e[campo] for e in grupo if e[campo] is not None], dtype=float)
            item[campo] = None if not len(valores) else {
                "media": float(valores.mean()), "desvio_padrao": float(valores.std(ddof=1)) if len(valores) > 1 else 0.0,
                "minimo": float(valores.min()), "maximo": float(valores.max()),
            }
        resumos.append(item)
    return resumos
