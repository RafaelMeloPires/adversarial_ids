# Dataset UNSW-NB15

Baixe os arquivos na página oficial: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Download reproduzível (espelho público fixado por commit, com SHA-256):

```powershell
python src/baixar_dados.py
python src/preparar_dados.py
```

O arquivo `origem.json` registra URLs, revisão, tamanhos e hashes. O espelho
é uma cópia de terceiros, não um servidor da UNSW. Os hashes fixam a versão
utilizada; não representam uma assinatura de autenticidade da universidade.
O script preserva arquivos existentes com hash divergente e interrompe a execução.

Arquivos utilizados:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Os CSVs não estão incluídos no ZIP devido ao tamanho e devem permanecer fora do Git.
