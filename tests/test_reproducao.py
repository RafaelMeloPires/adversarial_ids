import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from verificar_reproducao import comparar


class ReproducaoTest(unittest.TestCase):
    def preparar(self, pasta):
        met = {'acuracia': .8, 'precisao': .875, 'recall': .7, 'f1_score': 7/9,
            'matriz_confusao': [[9, 1], [3, 7]], 'taxa_falsos_positivos': .1,
            'tempo_treinamento_segundos': 1., 'configuracao': {'random_state': 42}}
        caminhos = [Path(pasta) / 'original', Path(pasta) / 'replica']
        for caminho in caminhos:
            caminho.mkdir()
            (caminho / 'metricas.json').write_text(json.dumps(met), encoding='utf-8')
            (caminho / 'modelo_random_forest.pkl').write_bytes(b'conteudo de teste, sem pickle')
        return caminhos, met

    def test_confere_identidade_sem_carregar_pickle(self):
        with tempfile.TemporaryDirectory() as pasta:
            caminhos, _ = self.preparar(pasta)
            r = comparar(*caminhos)
            self.assertTrue(r['serializacao_identica'])
            self.assertEqual(r['modelo_em_uso_sha256'], r['modelo_retreinado_sha256'])

    def test_rejeita_divergencia_de_metrica(self):
        with tempfile.TemporaryDirectory() as pasta:
            caminhos, met = self.preparar(pasta)
            met['recall'] = .6
            (caminhos[1] / 'metricas.json').write_text(json.dumps(met), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'recall'):
                comparar(*caminhos)
