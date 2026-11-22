"""Verificações reais opcionais; os testes unitários dispensam downloads."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dados import DATA_DIR, RESULTADOS_DIR, carregar_dados, separar, sha256
from baixar_dados import HASHES
from perturbacoes import MUTAVEIS, transformar


@unittest.skipUnless((DATA_DIR / 'UNSW_NB15_testing-set.csv').is_file(), 'CSV não baixado')
class IntegracaoDadosTest(unittest.TestCase):
    def test_hashes_e_restricoes_nos_fluxos_reais(self):
        for nome, esperado in HASHES.items():
            self.assertEqual(sha256(DATA_DIR / nome), esperado)
        _, teste = carregar_dados()
        X, _ = separar(teste)
        mod = transformar(X, .3, .3)
        imutaveis = [c for c in X if c not in MUTAVEIS]
        pd.testing.assert_frame_equal(X[imutaveis], mod[imutaveis])
        self.assertTrue((mod.sbytes >= X.sbytes).all())
        self.assertTrue((mod.sbytes <= np.floor(1.3 * X.sbytes)).all())
        self.assertTrue((mod.dur <= X.dur * 1.3 + 1e-12).all())
        self.assertTrue(np.isfinite(mod.select_dtypes(include='number')).all().all())
        np.testing.assert_allclose(mod.rate * mod.dur, X.rate * X.dur, rtol=1e-12)
        np.testing.assert_allclose(mod.dload * mod.dur, X.dload * X.dur, rtol=1e-12)


@unittest.skipUnless((RESULTADOS_DIR / 'resultados_ataque.json').is_file(), 'Ataque ainda não executado')
class IntegracaoAtaqueTest(unittest.TestCase):
    def test_resultados_reproduziveis_e_contagens(self):
        r = json.loads((RESULTADOS_DIR / 'resultados_ataque.json').read_text(encoding='utf-8'))
        if r.get('versao_protocolo') != 2:
            self.skipTest('Resultado do protocolo antigo')
        self.assertEqual(sha256(RESULTADOS_DIR / 'modelo_random_forest.pkl'), r['modelo_sha256'])
        self.assertEqual(len(r['experimentos']), 12)
        self.assertEqual(len(set(r['configuracao']['ids_amostra'])), r['configuracao']['amostras'])
        for e in r['experimentos']:
            m = e['busca_restrita']
            self.assertLessEqual(m['recall'], m['recall_antes'])
            self.assertEqual(m['recuperados'], 0)
            self.assertEqual(m['detectados_antes'] - m['evadiram'], m['detectados_depois'])
            self.assertEqual(e['consultas']['consultas_por_amostra'], r['configuracao']['tentativas'] + 1)


if __name__ == '__main__':
    unittest.main()
