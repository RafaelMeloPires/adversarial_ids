import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from selecao_robusta import construir_modelo
from perturbacoes import MUTAVEIS, transformar
from test_perturbacoes import fluxos


class SelecaoRobustaTest(unittest.TestCase):
    def test_invariancia_e_serializacao(self):
        X = fluxos()
        original = Pipeline([
            ('preprocessamento', ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), ['proto'])], remainder='passthrough')),
            ('classificador', RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        modelo = construir_modelo(original, X.columns)
        modelo.fit(X, [1, 0, 1])
        retidas = modelo.named_steps['selecao_robusta'].get_feature_names_out()
        self.assertTrue(set(retidas).isdisjoint(MUTAVEIS))
        mod = transformar(X, .3, .3)
        np.testing.assert_array_equal(modelo.predict_proba(X), modelo.predict_proba(mod))
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / 'modelo.pkl'
            joblib.dump(modelo, caminho)
            carregado = joblib.load(caminho)
            np.testing.assert_array_equal(carregado.predict_proba(X), modelo.predict_proba(X))


if __name__ == '__main__':
    unittest.main()
