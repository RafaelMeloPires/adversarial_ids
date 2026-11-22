import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perturbacoes import MUTAVEIS, TEMPOS, busca_evasao, transformar
from avaliacao import medir_classificacao, medir_evasao


def fluxos():
    dados = pd.DataFrame({
        "dur": [1., 0., 2.], "sbytes": [1000, 0, 4000], "dbytes": [500, 0, 600],
        "spkts": [10, 1, 2], "dpkts": [4, 0, 3], "smean": [100, 0, 2000],
        "rate": [13., 0., 2.], "sload": [7200., 0., 8000.], "dload": [3000., 0., 1000.],
        "proto": ["tcp", "udp", "tcp"], "sttl": [64, 254, 64],
    })
    for c in TEMPOS:
        dados[c] = [0.25, 0., .5]
    return dados


class DetectorTemporal:
    classes_ = np.array([0, 1])

    def predict_proba(self, dados):
        p = np.clip(.65 / np.maximum(dados.dur.to_numpy(), .01), 0, 1)
        return np.column_stack([1-p, p])


class PerturbacoesTest(unittest.TestCase):
    def test_orcamento_zero_e_identidade(self):
        dados = fluxos()
        pd.testing.assert_frame_equal(transformar(dados, 0, 0), dados)

    def test_orcamentos_dominios_e_relacoes(self):
        dados = fluxos()
        mod = transformar(dados, .3, .3)
        self.assertTrue((mod.dur >= dados.dur).all())
        self.assertTrue((mod.dur <= 1.3*dados.dur + 1e-12).all())
        self.assertTrue((mod.sbytes >= dados.sbytes).all())
        self.assertTrue((mod.sbytes <= np.floor(1.3*dados.sbytes)).all())
        self.assertTrue(pd.api.types.is_integer_dtype(mod.sbytes))
        self.assertEqual(mod.loc[2, 'sbytes'], dados.loc[2, 'sbytes'])
        pd.testing.assert_series_equal(mod.loc[1], dados.loc[1])
        imutaveis = [c for c in dados if c not in MUTAVEIS]
        pd.testing.assert_frame_equal(mod[imutaveis], dados[imutaveis])
        np.testing.assert_allclose(mod.rate*mod.dur, dados.rate*dados.dur)
        np.testing.assert_allclose(mod.dload*mod.dur, dados.dload*dados.dur)
        ativo = dados.sbytes > 0
        np.testing.assert_allclose((mod.sload*mod.dur/mod.sbytes)[ativo], (dados.sload*dados.dur/dados.sbytes)[ativo])
        np.testing.assert_allclose(mod.tcprtt, dados.tcprtt * np.where(dados.dur > 0, 1.3, 1))

    def test_busca_reprodutivel_retencao_original_e_orcamento(self):
        dados, modelo = fluxos(), DetectorTemporal()
        a, info = busca_evasao(modelo, dados, .3, semente=42, tentativas=8)
        b, _ = busca_evasao(modelo, dados, .3, semente=42, tentativas=8)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(info['consultas_por_amostra'], 9)
        self.assertTrue((modelo.predict_proba(a)[:, 1] <= modelo.predict_proba(dados)[:, 1]).all())
        self.assertGreater(info['amostras_com_probabilidade_reduzida'], 0)

    def test_rejeita_orcamento_invalido(self):
        for valor in [-.1, 1.1, np.nan]:
            with self.assertRaises(ValueError):
                transformar(fluxos(), valor, 0)


class MetricasTest(unittest.TestCase):
    def test_transicoes_pareadas_e_denominador_comum(self):
        m = medir_evasao([1, 1, 0, 1], [0, 1, 1, 0], elegiveis=[1, 1, 0, 0])
        self.assertEqual(m['evadiram'], 1)
        self.assertEqual(m['denominador_evasao'], 2)
        self.assertEqual(m['taxa_evasao'], .5)
        self.assertEqual(m['recuperados'], 1)
        self.assertEqual(m['queda_recall_pp'], 25.)

    def test_sem_deteccoes_e_indefinido(self):
        self.assertIsNone(medir_evasao([0, 0], [0, 0])['taxa_evasao'])

    def test_falsos_positivos(self):
        m = medir_classificacao([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(m['taxa_falsos_positivos'], .5)
        self.assertEqual(m['recall'], 1.)


if __name__ == '__main__':
    unittest.main()
