# Aprendizado de Máquina Adversarial contra Detecção de Intrusão

Experimento do tema 3 de Segurança em Sistemas Computacionais (UFJF), com
UNSW-NB15 e Random Forest. Mede evasão no espaço de características e o custo
das defesas em detecção e alarmes falsos. Não gera ataques contra redes reais.

## Reprodução no Windows

Ambiente verificado: Python 3.14.3. `requirements.txt` lista as dependências;
`requirements-lock.txt` fixa todas as versões utilizadas nesta execução.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
$env:MPLCONFIGDIR = "$PWD/resultados/.matplotlib"
python src/baixar_dados.py
python src/preparar_dados.py
python src/treinar_modelo.py
python src/ataque_adversarial.py
python src/defesa.py
python src/comparar_resultados.py
python -m unittest discover -s tests -v
```

Os scripts usam caminhos relativos à localização do projeto. Os CSVs e modelos
ficam fora do Git; resultados JSON, CSV e PNG são versionados. Carregue modelos
joblib apenas se tiverem sido gerados por você ou vierem de fonte confiável.
O treinamento do modelo original levou aproximadamente 123 s nesta máquina;
a duração depende do hardware. A defesa inclui outros três treinamentos e buscas.

`python src/defesa.py --reutilizar` reutiliza os modelos adicionais somente
quando configuração e hashes dos modelos e datasets conferem. `--amostras` e
`--tentativas` permitem execuções menores de ataque e defesa; para comparar,
ambos devem usar os mesmos valores. O padrão é 2.000 registros e 16 candidatos.

## Dados e verificação da implementação anterior

A [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) fornece as divisões
com 175.341 registros de treino e 82.332 de teste. O download automático usa um
espelho público fixado por commit e SHA-256, registrado em `data/origem.json`.
Os hashes identificam a cópia usada; não são assinatura de autenticidade da UNSW.
O script recusa sobrescrever CSVs existentes com hashes diferentes.

`preparar_dados.py` valida as 45 colunas, categorias, rótulos binários, domínios
inteiros, ausentes, valores negativos e não finitos. As 42 entradas excluem `id`,
`attack_cat` e `label`. Duplicações são reportadas, sem modificar a divisão oficial:
67.601 registros repetidos no treino, 26.387 no teste, desconsiderando `id`;
4.247 registros de teste coincidem com registros completos de treino sem `id`.
Considerando apenas as 42 entradas, há 8.541 coincidências de teste com treino.
A defesa também reporta métricas nos 73.791 registros sem essas coincidências.
Essa verificação não elimina todas as dependências entre os fluxos do dataset.

Os scripts anteriores das etapas 2, 3 e 4 foram executados antes da revisão.
As matrizes de confusão e métricas publicadas foram reproduzidas. O registro
completo está em `resultados/reproducao_inicial.json`. Os protocolos têm operações
e orçamentos diferentes; não se deve atribuir suas diferenças só à otimização.

## Protocolo atual

**Modelo original.** Pipeline com OneHotEncoder ajustado no treino e Random
Forest com 100 árvores, `random_state=42`, `n_jobs=-1`. Mantém o limiar padrão
e as divisões oficiais, sem usar o teste para ajustar o pré-processamento.

**Ataque.** Há três condições em `resultados_ataque.json`:

- Ruído histórico: perturbação independente de seis atributos em 5–30% do IQR
  do treinamento; pode criar registros inconsistentes e serve de referência.
- Aleatório restrito: um candidato com as mesmas operações e limites da busca.
- Busca restrita: 16 candidatos por registro, mais a consulta ao original.
  Minimiza a probabilidade prevista de ataque e mantém o original se nenhum
  candidato reduzir essa probabilidade. Não aumenta o recall por recuperação
  de ataques originalmente ignorados.

O atacante conhece as probabilidades do modelo. Pode acrescentar atraso e
padding nos pacotes de origem; orçamento de 5%, 10%, 20% ou 30% dos valores
originais, em cada operação. Os bytes são inteiros e aditivos, limitados à média
de 1.500 bytes por pacote. Registros que já excedem esse teto não ganham bytes.
Fluxos com duração zero são preservados. TTLs, protocolo, serviço, estado,
contagens de pacotes, bytes de destino e demais atributos são imutáveis.
Taxas, média de bytes de origem e atributos temporais relacionados são atualizados.
As razões são escaladas preservando os resíduos de arredondamento originais.
Isso é uma hipótese de manipulação com consistência parcial, sem prova de
preservação do payload ou viabilidade de reprodução em pacotes reais.

**Defesa e controle.** Seleciona 20% dos ataques de treinamento, semente 2026:
23.868 registros. A defesa gera duas cópias por registro com a busca direcionada,
intensidades 10% e 30%, usando somente o modelo original e o treino. O controle
recebe cópias idênticas, sem alteração. Ambos são clones do modelo original e
recebem 223.077 registros, com a mesma distribuição de rótulos. Assim é possível
separar o efeito das alterações do efeito de adicionar exemplos maliciosos.

**Avaliação.** Teste limpo completo para acurácia, precisão, recall, F1, matriz de
confusão e taxa de falsos positivos. Nos ataques, a mesma amostra maliciosa de
2.000 registros, semente 42, é avaliada com sementes de busca 42, 43 e 44:

- Transferência: os quatro modelos recebem exatamente os registros gerados contra
  o original, para medir generalização da defesa a um ataque fixo.
- Adaptativo: cada modelo é atacado diretamente, com as mesmas operações e o
  mesmo orçamento de consultas. As entradas resultantes podem ser diferentes.

A evasão própria usa detecções iniciais de cada modelo. A evasão comum usa a
interseção das detecções iniciais dos quatro, garantindo denominadores iguais.
Os JSONs e CSVs incluem contagens de evasões e recuperações, além das taxas.
Médias e desvios padrão referem-se às três sementes na mesma amostra e modelo;
não são intervalos de confiança nem medem variabilidade de treinamento.

Os parâmetros foram fixados antes da execução deste protocolo, sem seleção pelo
teste. Os resultados anteriores já eram conhecidos na revisão; não se alega
avaliação cega. Se parâmetros forem ajustados futuramente, é preciso separar
validação do treinamento e impedir que registros idênticos cruzem essa divisão.

**Seleção de atributos robustos.** Uma segunda defesa exclui, antes do
pré-processamento, os 13 atributos que podem variar nesta família de ataque.
Treina nas 29 entradas restantes, somente no treino limpo original. A exclusão
é definida pelo modelo de ameaça, sem escolha de atributos pelo desempenho de
teste. As previsões tornam-se invariantes às operações implementadas, ao custo
de perder informação útil. Essa garantia depende de o atacante alterar somente
os campos excluídos; não cobre manipulação dos atributos restantes.

## Resultados observados

| Modelo | Acurácia limpa | Recall limpo | F1 limpo | Falsos positivos | Recall sob ataque adaptativo a 30% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 87,10% | 98,55% | 89,37% | 26,93% | 97,47% |
| Controle por duplicação | 86,55% | 98,59% | 88,98% | 28,21% | 97,28% |
| Treinamento adversarial | 85,51% | 98,95% | 88,26% | 30,95% | 98,48% |
| Seleção de atributos | 86,17% | 97,84% | 88,63% | 28,12% | 98,30% |

O recall limpo da amostra de ataques do modelo original é 98,60%, diferente do
recall no teste completo. A busca a 30% causa queda média de 1,13 ponto percentual
nessa amostra. O treinamento adversarial reduz a evasão, mas aumenta os alarmes
falsos e reduz F1. O controle por duplicação não explica sozinho o benefício sob
ataque. Não há melhora geral de todas as métricas, nem evidência de proteção
contra qualquer família de ataque ou tráfego de uma rede real.

A seleção de atributos apresentou zero evasões e manteve recall de 98,30% na
amostra sob todas as intensidades. Ela já começa com recall limpo menor nessa
amostra, e tem F1 e falsos positivos piores que o original no teste completo.
Seu FPR é menor que o do treinamento adversarial. As duas defesas têm custos;
nenhuma melhorou simultaneamente todas as métricas do modelo original.

## Arquivos de saída e etapa 5

| Arquivo em `resultados/` | Conteúdo |
| --- | --- |
| `validacao_dados.json` | Integridade, distribuição, duplicações e proveniência |
| `reproducao_inicial.json` | Reprodução do experimento anterior |
| `modelo_random_forest.pkl`, `metricas.json`, `matriz_confusao.png` | Modelo e avaliação originais |
| `resultados_ataque.json`, `grafico_ataque.png` | Ruído, referência restrita e busca direcionada |
| `modelo_random_forest_controle.pkl`, `modelo_random_forest_defesa.pkl` | Controle e defesa |
| `treinamento_defesa.json` | Configuração, hashes e composição dos treinamentos aumentados |
| `modelo_random_forest_selecao.pkl`, `treinamento_selecao_robusta.json` | Defesa por seleção e atributos excluídos/retidos |
| `resultados_defesa.json` | Teste limpo, por categoria, sem coincidências, transferência e adaptação |
| `comparacao_metricas.csv`, `comparacao_ataques.csv` | Tabelas para análise e relatório |
| `comparacao_defesa.png` | Recall sob ataques transferidos e adaptativos |
| `metricas_limpo.png` | Métricas limpas e custo em alarmes falsos |
| `evasao_denominador_comum.png` | Evasão com denominador comum |
| `resumo_comparacao.json` | Síntese numérica para a etapa 5 |

O comparador recusa misturar modelos, datasets, amostras ou orçamentos diferentes.
Nos gráficos, as barras de erro mostram o desvio das sementes; o eixo de recall
é ampliado e identificado. A análise deve apresentar a queda causada pelo ataque,
a recuperação com defesa e o custo em falsos positivos. O relatório no modelo
do professor e os slides de até 15 minutos continuam pendentes.

## Referência do dataset

Moustafa, N.; Slay, J. UNSW-NB15: a comprehensive data set for network intrusion
detection systems (UNSW-NB15 network data set). MilCIS, 2015.
Consulte a página oficial para as demais referências solicitadas pelos autores.
