import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from defesa import montar_aumentacao


class AumentacaoTest(unittest.TestCase):
    def test_controle_e_defesa_preservam_rotulos_e_quantidades(self):
        X = pd.DataFrame({'dur': [1., 2., 3.]}, index=[9, 5, 100])
        y = pd.Series([1, 0, 1], index=X.index)
        selecionados = X.loc[y == 1]
        controle_X, controle_y = montar_aumentacao(X, y, selecionados, [selecionados.copy(), selecionados.copy()])
        alterados = selecionados * 1.3
        defesa_X, defesa_y = montar_aumentacao(X, y, selecionados, [alterados, alterados])
        pd.testing.assert_series_equal(controle_y, defesa_y)
        self.assertEqual(controle_y.tolist(), [1, 0, 1, 1, 1, 1, 1])
        self.assertEqual(len(controle_X), len(defesa_X))
        pd.testing.assert_frame_equal(defesa_X.iloc[:3].reset_index(drop=True), X.reset_index(drop=True))
        self.assertFalse(controle_X.equals(defesa_X))

    def test_rejeita_copia_desalinhada(self):
        X = pd.DataFrame({'dur': [1., 2.]}, index=[9, 5])
        with self.assertRaises(ValueError):
            montar_aumentacao(X, pd.Series([1, 0]), X, [X.reset_index(drop=True)])


if __name__ == '__main__':
    unittest.main()
