"""Confere duas execuções do treinamento sem substituir o modelo em uso."""

import argparse
import json
from pathlib import Path

from dados import RESULTADOS_DIR, proveniencia, salvar_json, sha256


def comparar(original, replica):
    original, replica = Path(original), Path(replica)
    a = json.loads((original / 'metricas.json').read_text(encoding='utf-8'))
    b = json.loads((replica / 'metricas.json').read_text(encoding='utf-8'))
    campos = ['acuracia', 'precisao', 'recall', 'f1_score', 'matriz_confusao', 'taxa_falsos_positivos']
    divergentes = [c for c in campos if a[c] != b[c]]
    if divergentes:
        raise ValueError(f'Métricas divergentes: {divergentes}')
    if a.get('proveniencia') != b.get('proveniencia'):
        raise ValueError('Ambientes/datasets das execuções são diferentes')
    ha = sha256(original / 'modelo_random_forest.pkl')
    hb = sha256(replica / 'modelo_random_forest.pkl')
    for met, esperado in [(a, ha), (b, hb)]:
        if 'modelo_sha256' in met and met['modelo_sha256'] != esperado:
            raise ValueError('Modelo não corresponde ao hash registrado nas métricas')
    return {'metricas_identicas': campos, 'modelo_em_uso_sha256': ha,
        'modelo_retreinado_sha256': hb, 'serializacao_identica': ha == hb,
        'tempo_segundos_retreinamento': b['tempo_treinamento_segundos'],
        'configuracao': b['configuracao']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diretorio', type=Path, default=RESULTADOS_DIR / 'verificacao_baseline')
    args = parser.parse_args()
    resultado = {'proveniencia': proveniencia(), **comparar(RESULTADOS_DIR, args.diretorio)}
    salvar_json(RESULTADOS_DIR / 'reprodutibilidade_modelo.json', resultado)
    print(f'Métricas idênticas. Serialização idêntica: {resultado["serializacao_identica"]}')
