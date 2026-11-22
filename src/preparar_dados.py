"""Inspeciona os CSVs sem excluir silenciosamente registros."""

import pandas as pd

from dados import COLUNAS_REMOVER, RESULTADOS_DIR, carregar_dados, proveniencia, salvar_json


def main():
    treino, teste = carregar_dados()
    resumo = {"proveniencia": proveniencia(), "conjuntos": {}}
    colunas = treino.drop(columns=["id"]).columns.tolist()
    hashes_treino = pd.util.hash_pandas_object(treino[colunas], index=False)
    hashes_teste = pd.util.hash_pandas_object(teste[colunas], index=False)
    for nome, dados in [("treino", treino), ("teste", teste)]:
        resumo["conjuntos"][nome] = {
            "registros": len(dados), "caracteristicas": len(dados.columns) - len(COLUNAS_REMOVER),
            "classes": {str(k): int(v) for k, v in dados.label.value_counts().items()},
            "categorias_ataque": {str(k): int(v) for k, v in dados.attack_cat.value_counts().items()},
            "duplicados_sem_id": int(dados.duplicated(subset=colunas).sum()),
        }
        print(nome, resumo["conjuntos"][nome], flush=True)
    resumo["registros_teste_identicos_ao_treino_sem_id"] = int(hashes_teste.isin(hashes_treino).sum())
    resumo["politica_duplicados"] = "Divisões oficiais preservadas; coincidências reportadas, sem remoção automática."
    salvar_json(RESULTADOS_DIR / "validacao_dados.json", resumo)
    print("Validação concluída; detalhes em resultados/validacao_dados.json")


if __name__ == "__main__":
    main()
