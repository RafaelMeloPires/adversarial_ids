"""Etapa 5: compara artefatos da mesma execução e exporta tabelas/gráficos."""

import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dados import RESULTADOS_DIR, salvar_json

ROTULOS = {'original': 'Original', 'controle_duplicacao': 'Controle: duplicação', 'defesa': 'Treino adversarial', 'selecao_robusta': 'Seleção de atributos'}


def conferir_compatibilidade(ataque, defesa):
    if ataque.get('versao_protocolo') != 2 or defesa.get('versao_protocolo') != 2:
        raise ValueError('Execute as etapas 3 e 4 com o protocolo 2')
    pares = [(ataque['modelo_sha256'], defesa['configuracao']['modelo_original_sha256']),
        (ataque['proveniencia']['datasets'], defesa['proveniencia']['datasets'])]
    for c in ['ids_amostra', 'tentativas', 'intensidades']:
        pares.append((ataque['configuracao'][c], defesa['configuracao'][c]))
    pares.append((ataque['configuracao']['sementes_busca'], defesa['configuracao']['sementes_busca']))
    if any(a != b for a,b in pares):
        raise ValueError('Artefatos incompatíveis: modelo, dados, amostra ou orçamento diferentes')


def gravar_csv(caminho, linhas):
    with caminho.open('w', newline='', encoding='utf-8-sig') as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        writer.writeheader()
        writer.writerows(linhas)


def executar():
    ataque = json.loads((RESULTADOS_DIR / 'resultados_ataque.json').read_text(encoding='utf-8'))
    defesa = json.loads((RESULTADOS_DIR / 'resultados_defesa.json').read_text(encoding='utf-8'))
    conferir_compatibilidade(ataque, defesa)
    nomes = list(defesa['teste_limpo_completo'])
    limpos = []
    for nome in nomes:
        m = defesa['teste_limpo_completo'][nome]
        tn, fp, fn, tp = [v for linha in m['matriz_confusao'] for v in linha]
        limpos.append({'modelo': nome, **{c: m[c] for c in ['amostras', 'acuracia', 'precisao', 'recall', 'f1_score', 'taxa_falsos_positivos']}, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
    gravar_csv(RESULTADOS_DIR / 'comparacao_metricas.csv', limpos)
    linhas = []
    for e in defesa['experimentos']:
        for tipo in ['transferencia', 'adaptativo']:
            linhas.append({'modelo': e['modelo'], 'cenario': tipo, 'semente': e['semente'], 'intensidade': e['intensidade'], **e[tipo]})
    gravar_csv(RESULTADOS_DIR / 'comparacao_ataques.csv', linhas)
    fig, eixos = plt.subplots(1, 2, figsize=(12, 5))
    for eixo, tipo in zip(eixos, ['transferencia', 'adaptativo']):
        inferiores = []
        for nome in nomes:
            pontos = defesa['resumos'][nome][tipo]
            x = [0] + [p['intensidade']*100 for p in pontos]
            y = [defesa['recall_amostra_limpa'][nome]*100] + [p['recall']['media']*100 for p in pontos]
            sd = [0] + [p['recall']['desvio_padrao']*100 for p in pontos]
            inferiores.extend(np.asarray(y)-sd)
            eixo.errorbar(x, y, yerr=sd, capsize=3, marker='o', label=ROTULOS[nome])
        eixo.set(xlabel='Orçamento relativo de atraso/padding (%)', ylabel='Recall na mesma amostra maliciosa (%)',
            title='Mesmas entradas: ataque ao original' if tipo == 'transferencia' else 'Ataque adaptativo a cada modelo')
        eixo.set_ylim(max(0, min(inferiores)-.3), min(100, max(99.5, max(inferiores)+.3)))
        eixo.grid(alpha=.3)
        eixo.legend(fontsize=8)
    fig.suptitle('Recall sob ataque — média e desvio padrão de 3 sementes; eixo ampliado')
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / 'comparacao_defesa.png', dpi=180)
    plt.close(fig)
    fig, eixos = plt.subplots(1, 2, figsize=(12, 5))
    campos = ['acuracia', 'precisao', 'recall', 'f1_score']
    x = np.arange(len(campos))
    largura = .8/len(nomes)
    for i,nome in enumerate(nomes):
        valores = [defesa['teste_limpo_completo'][nome][c]*100 for c in campos]
        eixos[0].bar(x+(i-(len(nomes)-1)/2)*largura, valores, largura, label=ROTULOS[nome])
    eixos[0].set(xticks=x, xticklabels=['Acurácia', 'Precisão', 'Recall', 'F1'], ylim=(0,100), ylabel='Percentual (%)', title='Teste limpo completo: 82.332 registros')
    eixos[0].legend(fontsize=8)
    valores = [defesa['teste_limpo_completo'][n]['taxa_falsos_positivos']*100 for n in nomes]
    barras = eixos[1].bar([ROTULOS[n] for n in nomes], valores)
    eixos[1].bar_label(barras, fmt='%.2f%%', padding=3)
    eixos[1].set(ylim=(0,max(valores)*1.25), ylabel='Normais classificados como ataque (%)', title='Custo em alarmes falsos')
    eixos[1].tick_params(axis='x', labelrotation=15)
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / 'metricas_limpo.png', dpi=180)
    plt.close(fig)
    fig, eixo = plt.subplots(figsize=(9, 5))
    for nome in nomes:
        pontos = defesa['resumos'][nome]['adaptativo']
        eixo.errorbar([p['intensidade']*100 for p in pontos], [p['taxa_evasao_comum']['media']*100 for p in pontos], yerr=[p['taxa_evasao_comum']['desvio_padrao']*100 for p in pontos], marker='o', capsize=3, label=ROTULOS[nome])
    eixo.set(xlabel='Orçamento relativo de atraso/padding (%)', ylabel='Evasão entre detecções comuns (%)', title=f'Evasão adaptativa — mesmo denominador: {defesa["configuracao"]["denominador_comum"]} registros')
    eixo.set_ylim(bottom=0)
    eixo.legend()
    eixo.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(RESULTADOS_DIR / 'evasao_denominador_comum.png', dpi=180)
    plt.close(fig)
    resumo = {'versao_protocolo': 2, 'modelo_original_sha256': ataque['modelo_sha256'], 'amostras_ataque': defesa['configuracao']['amostras'],
        'sementes': defesa['configuracao']['sementes_busca'], 'denominador_comum': defesa['configuracao']['denominador_comum'],
        'teste_limpo': limpos, 'ataque_orcamento_maximo': {},
        'interpretacao': ['Comparar modelos na mesma amostra e intensidade; o teste completo e a amostra maliciosa têm populações diferentes.',
            'Taxa de evasão comum usa a interseção das detecções limpas; a taxa própria de cada modelo tem outro denominador.',
            'Desvio padrão das sementes não é intervalo de confiança nem mede variabilidade de treinamento.',
            'Benefício sob ataque deve ser apresentado junto com F1, falsos positivos e controle por duplicação.']}
    for nome in nomes:
        resumo['ataque_orcamento_maximo'][nome] = defesa['resumos'][nome]['adaptativo'][-1]
    salvar_json(RESULTADOS_DIR / 'resumo_comparacao.json', resumo)
    print('Comparação conferida; 2 tabelas CSV, 3 gráficos e resumo JSON gerados', flush=True)
    return resumo


if __name__ == '__main__':
    executar()
