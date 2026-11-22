"""Defesa por exclusão dos atributos controláveis no modelo de ameaça declarado."""

import json
import time

import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dados import RESULTADOS_DIR, proveniencia, salvar_json, separar, sha256
from perturbacoes import MUTAVEIS


def construir_modelo(original, colunas):
    imutaveis = [c for c in colunas if c not in MUTAVEIS]
    if not imutaveis:
        raise ValueError('Não há atributos imutáveis para treinar')
    selecao = ColumnTransformer(
        [('imutaveis', 'passthrough', imutaveis)], remainder='drop',
        verbose_feature_names_out=False,
    ).set_output(transform='pandas')
    return Pipeline([('selecao_robusta', selecao)] + clone(original).steps)


def treinar(original, treino, reutilizar=False):
    X, y = separar(treino)
    caminho = RESULTADOS_DIR / 'modelo_random_forest_selecao.pkl'
    meta_path = RESULTADOS_DIR / 'treinamento_selecao_robusta.json'
    assinatura = {'versao': 1, 'modelo_original_sha256': sha256(RESULTADOS_DIR / 'modelo_random_forest.pkl'),
        'dataset_treino': proveniencia()['datasets']['treino'], 'excluidas': MUTAVEIS,
        'codigo_sha256': sha256(__file__)}
    if reutilizar and meta_path.is_file() and caminho.is_file():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['assinatura'] == assinatura and meta['sha256'] == sha256(caminho):
            print('Reutilizando seleção robusta com hashes conferidos', flush=True)
            return joblib.load(caminho), meta
    modelo = construir_modelo(original, X.columns)
    print(f'Treinando seleção robusta: {len(X.columns)-len(MUTAVEIS)} atributos, {len(X)} registros', flush=True)
    inicio = time.perf_counter()
    modelo.fit(X, y)
    segundos = time.perf_counter()-inicio
    joblib.dump(modelo, caminho)
    meta = {'assinatura': assinatura, 'arquivo': caminho.name, 'sha256': sha256(caminho),
        'tempo_treinamento_segundos': segundos, 'registros': len(X),
        'classes': {str(k): int(v) for k,v in y.value_counts().items()},
        'atributos_retidos': [c for c in X.columns if c not in MUTAVEIS],
        'garantia': 'Invariância às alterações apenas nos atributos excluídos, sem garantia contra outros ataques.'}
    salvar_json(meta_path, meta)
    return modelo, meta
