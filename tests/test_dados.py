import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dados import COLUNAS, CATEGORICAS, separar, validar


def exemplo():
    dados = pd.DataFrame({c: [1, 2] for c in COLUNAS})
    for c in CATEGORICAS:
        dados[c] = ["tcp", "udp"]
    dados["attack_cat"] = ["Normal", "DoS"]
    dados["label"] = [0, 1]
    return dados


class DadosTest(unittest.TestCase):
    def test_esquema_valido_e_remocao_dos_rotulos(self):
        dados = exemplo()
        validar(dados)
        X, y = separar(dados)
        self.assertEqual(X.shape, (2, 42))
        self.assertTrue(set(["id", "label", "attack_cat"]).isdisjoint(X.columns))
        self.assertEqual(y.tolist(), [0, 1])

    def test_rejeita_corrupcao(self):
        for coluna, valor in [("dur", np.inf), ("dur", -1), ("dur", np.nan), ("label", 3), ("proto", ""), ("id", 2), ("attack_cat", "Normal")]:
            with self.subTest(coluna=coluna, valor=valor):
                dados = exemplo().astype({"dur": float})
                dados.loc[0, coluna] = valor
                if coluna == "attack_cat":
                    dados.loc[1, coluna] = valor
                with self.assertRaises(ValueError):
                    validar(dados)

    def test_rejeita_coluna_faltante(self):
        with self.assertRaises(ValueError):
            validar(exemplo().drop(columns="dur"))

    def test_rejeita_bytes_fracionarios(self):
        dados = exemplo().astype({"sbytes": float})
        dados.loc[0, "sbytes"] = 1.5
        with self.assertRaises(ValueError):
            validar(dados)


if __name__ == "__main__":
    unittest.main()
