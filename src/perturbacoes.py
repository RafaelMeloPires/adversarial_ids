"""Ruído de referência e busca de evasão com operações limitadas por registro.

Modelo de ameaça: consulta às probabilidades do próprio IDS; pode dilatar o
tempo de um fluxo e acrescentar padding nos pacotes de origem. Protocolo,
serviço, estado, TTLs, contagens de pacotes e bytes de destino são imutáveis.
As relações derivadas são preservadas por escala, inclusive os resíduos de
arredondamento já existentes no CSV. Não constitui validação em pacotes reais.
"""

import numpy as np

COLUNAS_RUIDO = ["dur", "sbytes", "dbytes", "rate", "sload", "dload"]
TEMPOS = ["sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat"]
MUTAVEIS = ["dur", "sbytes", "rate", "sload", "dload", "smean"] + TEMPOS
TENTATIVAS = 16
INTENSIDADES = [0.05, 0.10, 0.20, 0.30]
SEMENTES = [42, 43, 44]
AMOSTRAS = 2000


def calcular_escalas(treino):
    return {c: float(treino[c].quantile(.75) - treino[c].quantile(.25)) for c in COLUNAS_RUIDO}


def ruido_iqr(dados, intensidade, rng, escalas):
    """Referência histórica: intensidade em fração do IQR, sem consistência física."""
    modificados = dados.copy()
    for c in COLUNAS_RUIDO:
        ruido = rng.uniform(-intensidade, intensidade, len(dados)) * escalas[c]
        modificados[c] = np.maximum(dados[c].to_numpy(dtype=float) + ruido, 0)
    return modificados


def transformar(dados, atraso, padding):
    """Intensidades relativas aos valores originais; operações apenas aditivas.

    Atraso e padding são vetores em [0, 1]. Padding respeita bytes inteiros e
    teto médio de 1500 bytes/pacote; fluxos que já excedem isso não ganham bytes.
    Fluxos de duração zero são preservados, pois sua temporização é indefinida.
    """
    atraso = np.broadcast_to(np.asarray(atraso, dtype=float), (len(dados),))
    padding = np.broadcast_to(np.asarray(padding, dtype=float), (len(dados),))
    if not np.isfinite(atraso).all() or not np.isfinite(padding).all() or (atraso < 0).any() or (padding < 0).any() or (atraso > 1).any() or (padding > 1).any():
        raise ValueError("Orçamento relativo deve estar em [0, 1]")
    saida = dados.copy()
    dur = dados.dur.to_numpy(dtype=float)
    bytes_orig = dados.sbytes.to_numpy(dtype=float)
    pkts = dados.spkts.to_numpy(dtype=float)
    ativo = (dur > 0) & (pkts > 0)
    fator = 1 + np.where(ativo, atraso, 0)
    capacidade = np.maximum(1500 * pkts - bytes_orig, 0)
    extra = np.floor(np.minimum(bytes_orig * padding, capacidade))
    extra = np.where(ativo, extra, 0)
    novos_bytes = (bytes_orig + extra).astype(np.int64)
    fator_bytes = np.divide(novos_bytes, bytes_orig, out=np.ones(len(dados)), where=bytes_orig > 0)
    saida["dur"] = dur * fator
    saida["sbytes"] = novos_bytes
    saida["rate"] = dados.rate.to_numpy(dtype=float) / fator
    saida["sload"] = dados.sload.to_numpy(dtype=float) * fator_bytes / fator
    saida["dload"] = dados.dload.to_numpy(dtype=float) / fator
    medias = np.floor(np.divide(novos_bytes, pkts, out=np.zeros(len(dados)), where=pkts > 0) + .5)
    saida["smean"] = np.where(extra > 0, medias, dados.smean).astype(np.int64)
    for c in TEMPOS:
        saida[c] = dados[c].to_numpy(dtype=float) * fator
    return saida


def probabilidade_ataque(modelo, dados):
    posicao = list(modelo.classes_).index(1)
    return modelo.predict_proba(dados)[:, posicao]


def ruido_restrito(dados, intensidade, rng):
    """Uma tentativa aleatória com as mesmas operações/orçamento da busca."""
    if not 0 <= intensidade <= 1:
        raise ValueError("Intensidade deve estar em [0, 1]")
    return transformar(dados, rng.uniform(0, intensidade, len(dados)), rng.uniform(0, intensidade, len(dados)))


def busca_evasao(modelo, dados, intensidade, semente=42, tentativas=TENTATIVAS):
    """Busca aleatória direcionada a classe normal, retendo o melhor candidato.

    O original é candidato: a probabilidade de ataque nunca aumenta. Cada
    amostra recebe o mesmo orçamento: uma consulta inicial + N candidatos.
    Os três primeiros candidatos exploram os extremos das duas operações.
    """
    if not 0 <= intensidade <= 1 or tentativas < 1:
        raise ValueError("Intensidade deve estar em [0, 1] e tentativas >= 1")
    melhor = dados.copy()
    probabilidades = probabilidade_ataque(modelo, dados)
    inicial = probabilidades.copy()
    rng = np.random.default_rng(semente)
    extremos = [(intensidade, 0), (0, intensidade), (intensidade, intensidade)]
    for tentativa in range(tentativas):
        if tentativa < len(extremos):
            atraso, padding = extremos[tentativa]
        else:
            atraso = rng.uniform(0, intensidade, len(dados))
            padding = rng.uniform(0, intensidade, len(dados))
        candidato = transformar(dados, atraso, padding)
        proba = probabilidade_ataque(modelo, candidato)
        melhores = proba < probabilidades
        if melhores.any():
            # Atribuir por coluna mantém bytes/contagens com seus tipos originais.
            for c in MUTAVEIS:
                melhor.loc[melhores, c] = candidato.loc[melhores, c]
            probabilidades[melhores] = proba[melhores]
    return melhor, {
        "consultas_por_amostra": tentativas + 1,
        "probabilidade_media_antes": float(inicial.mean()),
        "probabilidade_media_depois": float(probabilidades.mean()),
        "amostras_com_probabilidade_reduzida": int((probabilidades < inicial).sum()),
    }
