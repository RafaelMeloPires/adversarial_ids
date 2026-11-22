"""Etapa 4: treinamento adversarial, controle por duplicação e ataques adaptativos.

Os hiperparâmetros abaixo são fixos. O teste não seleciona configurações.
O controle e a defesa recebem os mesmos registros e rótulos adicionais;
só a transformação dos registros difere.
"""

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from avaliacao import medir_classificacao, medir_evasao, resumir_repeticoes
from dados import COLUNAS_REMOVER, RESULTADOS_DIR, carregar_dados, proveniencia, salvar_json, separar, sha256
from perturbacoes import AMOSTRAS, INTENSIDADES, SEMENTES, TENTATIVAS, busca_evasao

SEMENTE_TREINO = 2026
FRACAO_TREINO = .20
INTENSIDADES_TREINO = [.10, .30]
NOMES = ['original', 'controle_duplicacao', 'defesa']
ARQUIVOS_MODELOS = {
    'original': 'modelo_random_forest.pkl',
    'controle_duplicacao': 'modelo_random_forest_controle.pkl',
    'defesa': 'modelo_random_forest_defesa.pkl',
}


def montar_aumentacao(X, y, selecionados, copias):
    """Concatenação posicional evita desalinhamento de índices/rótulos."""
    partes = [X] + list(copias)
    for copia in copias:
        if not copia.index.equals(selecionados.index) or not copia.columns.equals(X.columns):
            raise ValueError('Cópias devem preservar índices/colunas da seleção')
    novo_X = pd.concat(partes, ignore_index=True)
    novo_y = pd.concat([y] + [pd.Series(np.ones(len(c), dtype=np.int64)) for c in copias], ignore_index=True)
    return novo_X, novo_y


def treinar_defesas(modelo, treino, tentativas, reutilizar):
    X, y = separar(treino)
    ataques = X.loc[y == 1]
    selecionados = ataques.sample(n=max(1, round(len(ataques)*FRACAO_TREINO)), random_state=SEMENTE_TREINO)
    assinatura = {
        'versao_protocolo': 2, 'modelo_original_sha256': sha256(RESULTADOS_DIR / ARQUIVOS_MODELOS['original']),
        'datasets': proveniencia()['datasets'], 'semente_treino': SEMENTE_TREINO,
        'fracao_ataques_treino': FRACAO_TREINO, 'intensidades_aumento': INTENSIDADES_TREINO,
        'tentativas': tentativas,
    }
    meta_path = RESULTADOS_DIR / 'treinamento_defesa.json'
    if reutilizar and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['assinatura'] == assinatura and all((RESULTADOS_DIR / ARQUIVOS_MODELOS[n]).is_file() and sha256(RESULTADOS_DIR / ARQUIVOS_MODELOS[n]) == meta['modelos'][n]['sha256'] for n in NOMES[1:]):
            print('Reutilizando modelos com hashes/configuração verificados', flush=True)
            return {'original': modelo, **{n: joblib.load(RESULTADOS_DIR / ARQUIVOS_MODELOS[n]) for n in NOMES[1:]}}, meta
        raise ValueError('Modelos em cache incompatíveis; execute sem --reutilizar')
    modelos, registros = {'original': modelo}, {}
    for nome in NOMES[1:]:
        copias, geracao = [], []
        for i, intensidade in enumerate(INTENSIDADES_TREINO):
            if nome == 'controle_duplicacao':
                copia, info = selecionados.copy(), {'transformacao': 'identidade'}
            else:
                copia, info = busca_evasao(modelo, selecionados, intensidade, SEMENTE_TREINO+i, tentativas)
            copias.append(copia)
            geracao.append({'intensidade': intensidade, **info})
        aumentado_X, aumentado_y = montar_aumentacao(X, y, selecionados, copias)
        print(f'Treinando {nome}: {len(aumentado_X)} registros', flush=True)
        estimador = clone(modelo)
        inicio = time.perf_counter()
        estimador.fit(aumentado_X, aumentado_y)
        segundos = time.perf_counter()-inicio
        caminho = RESULTADOS_DIR / ARQUIVOS_MODELOS[nome]
        joblib.dump(estimador, caminho)
        modelos[nome] = estimador
        registros[nome] = {'arquivo': caminho.name, 'sha256': sha256(caminho), 'tempo_treinamento_segundos': segundos,
            'registros': len(aumentado_X), 'classes': {str(k): int(v) for k,v in aumentado_y.value_counts().items()}, 'geracao': geracao}
    meta = {'assinatura': assinatura, 'ataques_selecionados': len(selecionados), 'registros_originais': len(X), 'modelos': registros}
    salvar_json(meta_path, meta)
    return modelos, meta


def medir_grupos(dados, pred):
    grupos = {}
    for categoria in sorted(dados.attack_cat.unique()):
        mask = dados.attack_cat.eq(categoria).to_numpy()
        grupos[categoria] = {'suporte': int(mask.sum()), 'taxa_prevista_ataque': float((pred[mask] == 1).mean())}
    return grupos


def executar(amostras=AMOSTRAS, tentativas=TENTATIVAS, reutilizar=False):
    treino, teste = carregar_dados()
    caminho_original = RESULTADOS_DIR / ARQUIVOS_MODELOS['original']
    original = joblib.load(caminho_original)
    modelos, treinamento = treinar_defesas(original, treino, tentativas, reutilizar)
    X_teste, y_teste = separar(teste)
    teste_limpo, por_categoria, teste_sem_coincidencias = {}, {}, {}
    # Diagnóstico adicional: coincidência das 42 entradas, sem consultar rótulos.
    X_treino, _ = separar(treino)
    hash_treino = pd.util.hash_pandas_object(X_treino, index=False)
    novos = ~pd.util.hash_pandas_object(X_teste, index=False).isin(hash_treino).to_numpy()
    for nome, modelo in modelos.items():
        pred = modelo.predict(X_teste)
        teste_limpo[nome] = medir_classificacao(y_teste, pred)
        por_categoria[nome] = medir_grupos(teste, pred)
        teste_sem_coincidencias[nome] = medir_classificacao(y_teste.to_numpy()[novos], pred[novos])
        print(f'Limpo {nome}: F1={teste_limpo[nome]["f1_score"]:.4f}, FPR={teste_limpo[nome]["taxa_falsos_positivos"]:.4f}', flush=True)
    ataques = teste.loc[teste.label == 1].sample(n=min(amostras, int(teste.label.sum())), random_state=42)
    X_ataques, _ = separar(ataques)
    pred_limpo = {n: m.predict(X_ataques) for n,m in modelos.items()}
    comum = np.logical_and.reduce([p == 1 for p in pred_limpo.values()])
    experimentos = []
    for semente in SEMENTES:
        for intensidade in INTENSIDADES:
            transferido, info_origem = busca_evasao(original, X_ataques, intensidade, semente, tentativas)
            for nome, modelo in modelos.items():
                if nome == 'original':
                    adaptativo, info = transferido, info_origem
                else:
                    adaptativo, info = busca_evasao(modelo, X_ataques, intensidade, semente, tentativas)
                ponto = {'semente': semente, 'intensidade': intensidade, 'modelo': nome, 'consultas': info}
                for tipo, entrada in [('transferencia', transferido), ('adaptativo', adaptativo)]:
                    pred = modelo.predict(entrada)
                    met = medir_evasao(pred_limpo[nome], pred)
                    met_comum = medir_evasao(pred_limpo[nome], pred, comum)
                    ponto[tipo] = {**met, 'taxa_evasao_comum': met_comum['taxa_evasao'], 'evadiram_comum': met_comum['evadiram'], 'denominador_comum': met_comum['denominador_evasao']}
                experimentos.append(ponto)
                print(f'{semente} {intensidade:.0%} {nome}: recall adaptativo {ponto["adaptativo"]["recall"]:.4f}', flush=True)
    resumos = {}
    for nome in NOMES:
        resumos[nome] = {}
        for tipo in ['transferencia', 'adaptativo']:
            pontos = [dict(intensidade=e['intensidade'], **e[tipo]) for e in experimentos if e['modelo'] == nome]
            resumos[nome][tipo] = resumir_repeticoes(pontos, ['recall', 'queda_recall_pp', 'taxa_evasao', 'taxa_evasao_comum'])
    saida = {
        'versao_protocolo': 2, 'proveniencia': proveniencia(), 'treinamento': treinamento,
        'configuracao': {'amostras': len(X_ataques), 'semente_amostra': 42, 'sementes_busca': SEMENTES, 'tentativas': tentativas, 'intensidades': INTENSIDADES,
            'ids_amostra': ataques.id.tolist(), 'denominador_comum': int(comum.sum()), 'modelo_original_sha256': sha256(caminho_original),
            'politica_parametros': 'Fixos antes da execução; sem seleção pelo desempenho de teste. Ajustes futuros exigem validação separada do treino.'},
        'teste_limpo_completo': teste_limpo, 'teste_limpo_por_categoria': por_categoria,
        'teste_sem_entradas_identicas_ao_treino': {'excluidos': int((~novos).sum()), 'modelos': teste_sem_coincidencias},
        'recall_amostra_limpa': {n: float((p == 1).mean()) for n,p in pred_limpo.items()},
        'experimentos': experimentos, 'resumos': resumos,
        'limitacoes': ['Mesmo modelo/mesma amostra em todas as sementes; desvio descreve apenas a aleatoriedade da busca.',
            'Operações coerentes parcialmente no espaço de características; sem validação do tráfego/payload.',
            'Transferência reutiliza entradas do ataque ao original; ataque adaptativo consulta cada modelo com orçamento igual.',
            'Hiperparâmetros do modelo e limiar padrão preservados; FPR alto pode permanecer ou piorar.',
            'Os resultados de teste do protocolo anterior já eram conhecidos na revisão; não há alegação de avaliação cega.'],
    }
    salvar_json(RESULTADOS_DIR / 'resultados_defesa.json', saida)
    print('Defesa e controle avaliados; execute python src/comparar_resultados.py', flush=True)
    return saida


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amostras', type=int, default=AMOSTRAS)
    parser.add_argument('--tentativas', type=int, default=TENTATIVAS)
    parser.add_argument('--reutilizar', action='store_true', help='Reutiliza apenas modelos com configuração e hashes conferidos')
    args = parser.parse_args()
    if args.amostras < 1 or args.tentativas < 1:
        parser.error('Amostras e tentativas devem ser positivas')
    executar(args.amostras, args.tentativas, args.reutilizar)
