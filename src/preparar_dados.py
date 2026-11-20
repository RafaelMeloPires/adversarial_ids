from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

arquivos = {
    'treino': DATA_DIR / 'UNSW_NB15_training-set.csv',
    'teste': DATA_DIR / 'UNSW_NB15_testing-set.csv',
}

faltantes = [str(caminho) for caminho in arquivos.values() if not caminho.exists()]
if faltantes:
    raise SystemExit('Arquivos do dataset não encontrados:\n- ' + '\n- '.join(faltantes) + '\nConsulte data/README.md para baixar os CSVs.')

treino = pd.read_csv(arquivos['treino'])
teste = pd.read_csv(arquivos['teste'])

print('=== DATASET UNSW-NB15 ===')
print(f'Registros de treinamento: {len(treino)}')
print(f'Registros de teste: {len(teste)}')
print('\nPrimeiros registros:')
print(treino.head())

for nome, dados in [('treinamento', treino), ('teste', teste)]:
    print(f'\nDistribuição das classes no {nome}:')
    print(dados['label'].value_counts())
    print(f'Valores ausentes no {nome}: {dados.isnull().sum().sum()}')

colunas_remover = ['id', 'attack_cat', 'label']
X_treino = treino.drop(columns=colunas_remover)
y_treino = treino['label']
X_teste = teste.drop(columns=colunas_remover)
y_teste = teste['label']

print(f'\nQuantidade de características: {X_treino.shape[1]}')
print(f'Formato de X_treino: {X_treino.shape}; X_teste: {X_teste.shape}')
print('Preparação inicial concluída!')
