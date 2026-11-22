import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from comparar_resultados import conferir_compatibilidade


class CompatibilidadeTest(unittest.TestCase):
    def test_recusa_resultados_de_experimentos_diferentes(self):
        config = {'ids_amostra': [4, 8], 'tentativas': 16, 'intensidades': [.1, .3], 'sementes_busca': [42, 43, 44]}
        ataque = {'versao_protocolo': 2, 'modelo_sha256': 'modelo-a', 'proveniencia': {'datasets': {'treino': 'hash-a'}}, 'configuracao': config}
        defesa = {'versao_protocolo': 2, 'proveniencia': copy.deepcopy(ataque['proveniencia']), 'configuracao': dict(config, modelo_original_sha256='modelo-a')}
        conferir_compatibilidade(ataque, defesa)
        for campo, valor in [('ids_amostra', [8,4]), ('tentativas', 32), ('intensidades', [.2]), ('sementes_busca', [42]), ('modelo_original_sha256', 'outro')]:
            with self.subTest(campo=campo):
                diferente = copy.deepcopy(defesa)
                diferente['configuracao'][campo] = valor
                with self.assertRaises(ValueError):
                    conferir_compatibilidade(ataque, diferente)
        diferente = copy.deepcopy(defesa)
        diferente['proveniencia']['datasets']['treino'] = 'outro'
        with self.assertRaises(ValueError):
            conferir_compatibilidade(ataque, diferente)


if __name__ == '__main__':
    unittest.main()
