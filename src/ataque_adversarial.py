"""Etapa 3: referência de ruído e evasão direcionada sob restrições."""

import argparse
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from avaliacao import medir_evasao, resumir_repeticoes
from dados import RESULTADOS_DIR, carregar_dados, proveniencia, salvar_json, separar, sha256
from perturbacoes import AMOSTRAS, INTENSIDADES, SEMENTES, TENTATIVAS, busca_evasao, calcular_escalas, ruido_iqr, ruido_restrito


def executar(amostras=AMOSTRAS, tentativas=TENTATIVAS, sementes=SEMENTES):
    treino, teste = carregar_dados()
    caminho = RESULTADOS_DIR / 'modelo_random_forest.pkl'
    modelo = joblib.load(caminho)
    ataques = teste.loc[teste.label == 1].sample(n=min(amostras, int(teste.label.sum())), random_state=42)
    X, _ = separar(ataques)
    pred_limpo = modelo.predict(X)
    escalas = calcular_escalas(treino)
    experimentos = []
    for semente in sementes:
        rng = np.random.default_rng(semente)
        for intensidade in INTENSIDADES:
            ruido = ruido_iqr(X, intensidade, rng, escalas)
            met_ruido = medir_evasao(pred_limpo, modelo.predict(ruido))
            referencia = ruido_restrito(X, intensidade, np.random.default_rng(semente))
            met_restrito = medir_evasao(pred_limpo, modelo.predict(referencia))
            adversarial, consultas = busca_evasao(modelo, X, intensidade, semente, tentativas)
            met_busca = medir_evasao(pred_limpo, modelo.predict(adversarial))
            experimentos.append({'semente': semente, 'intensidade': intensidade, 'ruido_iqr': met_ruido, 'aleatorio_restrito': met_restrito, 'busca_restrita': met_busca, 'consultas': consultas})
            print(f'Semente {semente}, intensidade {intensidade:.0%}: recall {met_busca["recall"]:.4f}, evasao {met_busca["taxa_evasao"]:.4%}', flush=True)
    resumos = {}
    for metodo in ['ruido_iqr', 'aleatorio_restrito', 'busca_restrita']:
        resumos[metodo] = resumir_repeticoes([dict(intensidade=e['intensidade'], **e[metodo]) for e in experimentos], ['recall', 'taxa_evasao', 'queda_recall_pp'])
    saida = {
        'versao_protocolo': 2, 'proveniencia': proveniencia(), 'modelo_sha256': sha256(caminho),
        'configuracao': {'amostras': len(X), 'semente_amostra': 42, 'sementes_busca': list(sementes), 'tentativas': tentativas, 'intensidades': INTENSIDADES, 'ids_amostra': ataques.id.tolist(),
            'orcamento_busca': 'fracao do valor original: atraso e padding; inclui atualizacao de atributos derivados',
            'orcamento_ruido': 'fracao do IQR do treino em seis atributos independentes; referencia historica, nao equivalente ao orcamento da busca'},
        'recall_original': float((pred_limpo == 1).mean()), 'experimentos': experimentos, 'resumos': resumos,
        'limitacoes': ['Espaco de caracteristicas; sem demonstracao de preservacao do ataque em pacotes reais.', 'Ruido IQR e busca usam orcamentos diferentes; referencia restrita e busca compartilham operacoes e limites, com 1 e N candidatos, respectivamente.', 'Desvio padrao descreve sementes da busca na mesma amostra e no mesmo modelo, nao variabilidade de treinamento.'],
    }
    salvar_json(RESULTADOS_DIR / 'resultados_ataque.json', saida)
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4.5))
    for eixo, metrica, titulo in [(eixos[0], 'recall', 'Recall (%)'), (eixos[1], 'taxa_evasao', 'Taxa de evasao (%)')]:
        for metodo, rotulo in [('ruido_iqr', 'Ruido IQR (historico)'), ('aleatorio_restrito', 'Aleatorio restrito (1 candidato)'), ('busca_restrita', f'Busca restrita ({tentativas} candidatos)')]:
            pontos = resumos[metodo]
            eixo.errorbar([p['intensidade'] * 100 for p in pontos], [p[metrica]['media'] * 100 for p in pontos], yerr=[p[metrica]['desvio_padrao'] * 100 for p in pontos], marker='o', capsize=3, label=rotulo)
        if metrica == 'recall':
            eixo.axhline(saida['recall_original'] * 100, color='gray', linestyle='--', label='Amostra limpa')
        eixo.set(xlabel='Orcamento (%) — ver definicao de cada metodo', ylabel=titulo)
        eixo.grid(alpha=.3)
        eixo.legend(fontsize=8)
    fig.suptitle('Ruido de referencia e busca de evasao: media e desvio entre sementes')
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / 'grafico_ataque.png', dpi=180)
    plt.close(fig)
    return saida


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--amostras', type=int, default=AMOSTRAS)
    parser.add_argument('--tentativas', type=int, default=TENTATIVAS)
    args = parser.parse_args()
    if args.amostras < 1 or args.tentativas < 1:
        parser.error('Amostras e tentativas devem ser positivas')
    executar(args.amostras, args.tentativas)
