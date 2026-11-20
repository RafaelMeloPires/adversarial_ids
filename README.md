# Aprendizado de Máquina Adversarial contra Detecção de Intrusão

Projeto acadêmico com o dataset UNSW-NB15. Etapa 1: leitura e inspeção inicial dos dados.

## Requisitos
- Python 3.11 ou 3.12
- VS Code (opcional)

## Como executar no Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/preparar_dados.py
```

Se a ativação estiver bloqueada, execute ` .\.venv\Scripts\python.exe -m pip install -r requirements.txt ` e ` .\.venv\Scripts\python.exe src/preparar_dados.py `.

## Dataset

Baixe `UNSW_NB15_training-set.csv` e `UNSW_NB15_testing-set.csv` da [página oficial UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) e coloque ambos em `data/`. Os arquivos CSV não estão incluídos neste ZIP.

## Estrutura
- `src/preparar_dados.py`: carrega e inspeciona os CSVs, separando características e rótulos.
- `src/treinar_modelo.py`: reservado para a próxima etapa.
- `src/ataque_adversarial.py`: reservado para a etapa de evasão.
- `src/defesa.py`: reservado para a etapa de defesa.
- `resultados/`: gráficos e métricas futuros.

**Observação:** a codificação de variáveis categóricas e o pré-processamento para treino serão implementados na próxima etapa. Os scripts das próximas etapas são apenas marcadores por enquanto.
