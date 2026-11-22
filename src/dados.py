"""Contrato dos CSVs oficiais; nenhuma transformação é ajustada no teste."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTADOS_DIR = BASE_DIR / "resultados"
COLUNAS_REMOVER = ["id", "attack_cat", "label"]
CATEGORICAS = ["proto", "service", "state"]
COLUNAS = """id dur proto service state spkts dpkts sbytes dbytes rate sttl dttl
sload dload sloss dloss sinpkt dinpkt sjit djit swin stcpb dtcpb dwin tcprtt
synack ackdat smean dmean trans_depth response_body_len ct_srv_src ct_state_ttl
ct_dst_ltm ct_src_dport_ltm ct_dst_sport_ltm ct_dst_src_ltm is_ftp_login ct_ftp_cmd
ct_flw_http_mthd ct_src_ltm ct_srv_dst is_sm_ips_ports attack_cat label""".split()
ARQUIVOS = {"treino": "UNSW_NB15_training-set.csv", "teste": "UNSW_NB15_testing-set.csv"}
CONTINUAS = "dur rate sload dload sinpkt dinpkt sjit djit tcprtt synack ackdat".split()


def sha256(caminho):
    with Path(caminho).open("rb") as arquivo:
        return hashlib.file_digest(arquivo, "sha256").hexdigest()


def salvar_json(caminho, conteudo):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(conteudo, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def validar(dados, nome="dataset"):
    if list(dados.columns) != COLUNAS:
        raise ValueError(f"{nome}: esquema/ordem de colunas diferente dos CSVs oficiais")
    if dados.empty or dados.isna().any().any():
        raise ValueError(f"{nome}: dataset vazio ou com valores ausentes")
    numericas = [c for c in COLUNAS if c not in CATEGORICAS + ["attack_cat"]]
    if not all(pd.api.types.is_numeric_dtype(dados[c]) for c in numericas):
        raise ValueError(f"{nome}: atributo numérico com tipo inválido")
    valores = dados[numericas].to_numpy(dtype=float)
    if not np.isfinite(valores).all() or (valores < 0).any():
        raise ValueError(f"{nome}: valores numéricos negativos ou não finitos")
    inteiras = [c for c in numericas if c not in CONTINUAS]
    if (dados[inteiras].to_numpy(dtype=float) % 1 != 0).any():
        raise ValueError(f"{nome}: contagens/identificadores fracionários")
    if not dados.label.isin([0, 1]).all() or dados.label.nunique() != 2:
        raise ValueError(f"{nome}: label deve conter as duas classes 0 e 1")
    if dados.id.duplicated().any() or (dados.id % 1 != 0).any():
        raise ValueError(f"{nome}: identificadores repetidos ou fracionários")
    for coluna in CATEGORICAS + ["attack_cat"]:
        if not dados[coluna].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{nome}: categoria vazia/inválida em {coluna}")
    normal = dados.attack_cat.str.strip().str.lower().eq("normal")
    if not normal.eq(dados.label.eq(0)).all():
        raise ValueError(f"{nome}: attack_cat incompatível com label")


def carregar_dados(pasta=DATA_DIR):
    conjuntos = []
    for nome, arquivo in ARQUIVOS.items():
        caminho = Path(pasta) / arquivo
        if not caminho.is_file():
            raise FileNotFoundError(f"Dataset ausente: {caminho}. Execute python src/baixar_dados.py")
        dados = pd.read_csv(caminho)
        validar(dados, nome)
        conjuntos.append(dados)
    return tuple(conjuntos)


def separar(dados):
    return dados.drop(columns=COLUNAS_REMOVER), dados.label.copy()


def proveniencia():
    return {
        "python": platform.python_version(),
        "dependencias": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib"]},
        "datasets": {nome: {"arquivo": arquivo, "sha256": sha256(DATA_DIR / arquivo)} for nome, arquivo in ARQUIVOS.items()},
    }
