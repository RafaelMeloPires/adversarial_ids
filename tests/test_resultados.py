import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dados import RESULTADOS_DIR, sha256


@unittest.skipUnless((RESULTADOS_DIR / 'resultados_defesa.json').is_file(), 'Defesa ainda não executada')
class ResultadosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = json.loads((RESULTADOS_DIR / 'resultados_defesa.json').read_text(encoding='utf-8'))
        if cls.r.get('versao_protocolo') != 2:
            raise unittest.SkipTest('Resultados do protocolo antigo')

    def test_matrizes_e_metricas_recalculadas(self):
        for nome, m in self.r['teste_limpo_completo'].items():
            with self.subTest(modelo=nome):
                (tn, fp), (fn, tp) = m['matriz_confusao']
                self.assertEqual(tn+fp+fn+tp, m['amostras'])
                self.assertAlmostEqual(m['acuracia'], (tn+tp)/m['amostras'])
                self.assertAlmostEqual(m['precisao'], tp/(tp+fp))
                self.assertAlmostEqual(m['recall'], tp/(tp+fn))
                self.assertAlmostEqual(m['f1_score'], 2*tp/(2*tp+fp+fn))
                self.assertAlmostEqual(m['taxa_falsos_positivos'], fp/(fp+tn))

    def test_transicoes_e_denominadores(self):
        comum = self.r['configuracao']['denominador_comum']
        for e in self.r['experimentos']:
            for tipo in ['transferencia', 'adaptativo']:
                m = e[tipo]
                self.assertEqual(m['detectados_depois'], m['detectados_antes'] - m['evadiram'] + m['recuperados'])
                self.assertEqual(m['denominador_comum'], comum)
                self.assertLessEqual(m['evadiram_comum'], m['evadiram'])
                self.assertAlmostEqual(m['taxa_evasao'], m['evadiram']/m['denominador_evasao'])
                self.assertAlmostEqual(m['taxa_evasao_comum'], m['evadiram_comum']/comum)
                if tipo == 'adaptativo':
                    self.assertEqual(m['recuperados'], 0)

    def test_mesmos_registros_modelo_e_resultados_da_etapa_3(self):
        ataque = json.loads((RESULTADOS_DIR / 'resultados_ataque.json').read_text(encoding='utf-8'))
        self.assertEqual(ataque['modelo_sha256'], self.r['configuracao']['modelo_original_sha256'])
        self.assertEqual(ataque['configuracao']['ids_amostra'], self.r['configuracao']['ids_amostra'])
        pontos = {(p['semente'],p['intensidade']): p['busca_restrita'] for p in ataque['experimentos']}
        for p in self.r['experimentos']:
            if p['modelo'] == 'original':
                met = pontos[(p['semente'], p['intensidade'])]
                for c in met:
                    self.assertEqual(met[c], p['adaptativo'][c])
                self.assertEqual(p['adaptativo'], p['transferencia'])

    def test_controle_e_defesa_tem_mesma_distribuicao(self):
        modelos = self.r['treinamento']['modelos']
        self.assertEqual(modelos['controle_duplicacao']['classes'], modelos['defesa']['classes'])
        self.assertEqual(modelos['controle_duplicacao']['registros'], modelos['defesa']['registros'])
        if not all((RESULTADOS_DIR / m['arquivo']).is_file() for m in modelos.values()):
            self.skipTest('Modelos locais ainda não gerados')
        for m in modelos.values():
            self.assertEqual(sha256(RESULTADOS_DIR / m['arquivo']), m['sha256'])

    def test_selecao_robusta_preserva_previsoes(self):
        if 'selecao_robusta' not in self.r['teste_limpo_completo']:
            self.skipTest('Seleção robusta ainda não avaliada')
        for e in self.r['experimentos']:
            if e['modelo'] == 'selecao_robusta':
                for tipo in ['transferencia', 'adaptativo']:
                    self.assertEqual(e[tipo]['evadiram'], 0)
                    self.assertEqual(e[tipo]['recuperados'], 0)
                    self.assertEqual(e[tipo]['queda_recall_pp'], 0.)


if __name__ == '__main__':
    unittest.main()
