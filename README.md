# Aprendizado de Máquina Adversarial contra Detecção de Intrusão

Experimento com o **UNSW-NB15** para o tema 3 da disciplina Segurança em Sistemas Computacionais (UFJF). O projeto compara o desempenho de um Random Forest antes, durante e após o treinamento com dados sinteticamente perturbados.

## Requisitos

- Python 3.11 ou 3.12
- Dependências listadas em `requirements.txt`
- Dataset UNSW-NB15, divisão *Training and Testing Sets* da [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

## Configuração no Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Adicione os arquivos abaixo na pasta `data/` (não versionados):

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

## Executar na ordem

```powershell
python src/preparar_dados.py       # Etapa 1: inspecionar dataset
python src/treinar_modelo.py      # Etapa 2: treinar modelo original
python src/ataque_adversarial.py  # Etapa 3: avaliar a evasão
python src/defesa.py              # Etapa 4: treinamento adversarial
```

Se o arquivo `resultados/modelo_random_forest.pkl` já existir, **não é necessário repetir** o treinamento da etapa 2. O script da etapa 4 usa o modelo existente para clonar a arquitetura e comparar os resultados. Carregue modelos `joblib` apenas de fontes confiáveis.

## Como funciona a defesa (Etapa 4)

1. Carrega o modelo da etapa 2 e os conjuntos oficiais de treino/teste.
2. Calcula os intervalos interquartis (IQR) de seis variáveis **apenas com os dados de treino**: `dur`, `sbytes`, `dbytes`, `rate`, `sload`, `dload`.
3. Seleciona, com semente fixa, 20% dos registros maliciosos do **treinamento** e cria duas versões perturbadas (intensidades 10% e 30%) com a mesma regra do script `ataque_adversarial.py`. Inclui os registros originais e as cópias manipuladas na base ampliada, rotuladas como ataque.
4. Cria um novo Random Forest com a mesma configuração do original (`sklearn.base.clone`) e treina do zero na base ampliada. Não altera nem substitui o modelo anterior.
5. Avalia os dois classificadores no **mesmo conjunto de teste limpo** (acurácia, precisão, recall, F1 e matriz de confusão).
6. Avalia ambos nos mesmos 2.000 ataques de teste perturbados, usando sementes e intensidades iguais às da Etapa 3 (5%, 10%, 20% e 30%). Salva um gráfico comparativo do recall.

A **comparação principal** é o recall no mesmo conjunto de ataques perturbados: quanto maior, menos ataques passaram despercebidos. A taxa de evasão também é fornecida, mas considera apenas os ataques originalmente detectados por cada modelo; seus denominadores podem ser diferentes.

## Arquivos de saída

| Arquivo em `resultados/` | Conteúdo |
| --- | --- |
| `modelo_random_forest.pkl` | Modelo original (etapa 2) |
| `metricas.json`, `matriz_confusao.png` | Métricas e gráfico originais |
| `resultados_ataque.json`, `grafico_ataque.png` | Análise da evasão (etapa 3) |
| `modelo_random_forest_defesa.pkl` | Novo modelo com treinamento adversarial (etapa 4) |
| `resultados_defesa.json` | Resultados e parâmetros da comparação (etapa 4) |
| `comparacao_defesa.png` | Comparação do recall sob perturbação (etapa 4) |

Os arquivos JSON e PNG podem ser versionados; os `.pkl` e os arquivos CSV ficam ignorados no `.gitignore`.

## Limitações importantes

### Revisão do ataque (protocolo 2)

O ataque agora mantém o ruído IQR histórico e inclui uma referência aleatória
restrita e uma busca direcionada à classe normal, com 16 candidatos e retenção
do original. Executa três sementes (42, 43 e 44) sobre os mesmos 2.000 registros.
O orçamento de 5–30% da busca é relativo à duração e aos bytes originais;
o orçamento do ruído histórico continua relativo ao IQR e não é equivalente.
A busca permite apenas atraso e padding na origem, preserva bytes inteiros,
atualiza taxas e atributos temporais relacionados e congela fluxos de duração
zero. TTLs, categorias, contagens de pacotes e bytes de destino são imutáveis.
Isso impõe consistência parcial; ainda não demonstra viabilidade em pacotes.

Os resultados anteriores foram reproduzidos e preservados em
`resultados/reproducao_inicial.json`. Testes: `python -m unittest discover -s tests -v`.

- O ruído histórico é uma referência aleatória; a busca restrita otimiza a probabilidade prevista, sem demonstrar exploração de uma rede real.
- A perturbação pode produzir combinações fisicamente inconsistentes entre duração, bytes e taxas; não prova a viabilidade de gerar tais fluxos de rede.
- Treinamento com exemplos perturbados pode melhorar, não alterar ou **piorar** o recall e as falsas detecções; os resultados devem ser reportados como observados, não pressupostos.
- O conjunto de teste nunca é usado para escolher parâmetros da defesa. Os parâmetros fixos da etapa 4 devem ser tratados como pré-definidos, evitando ajustar o modelo repetidamente pelo desempenho do teste.

## Organização

```text
adversarial_ids/
├── data/                           # CSVs do dataset (baixados separadamente)
├── src/
│   ├── preparar_dados.py
│   ├── treinar_modelo.py
│   ├── ataque_adversarial.py
│   └── defesa.py
├── resultados/                     # Modelos, métricas e gráficos
├── requirements.txt
├── .gitignore
└── README.md
```
