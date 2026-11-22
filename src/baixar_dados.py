"""Baixa um espelho público fixado por commit e verifica SHA-256 antes de instalar."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REVISAO = "44bbd02b9e70f74dd915218d5eb702f125341ffc"
BASE_URL = f"https://raw.githubusercontent.com/Nir-J/ML-Projects/{REVISAO}/UNSW-Network_Packet_Classification"
HASHES = {
    "UNSW_NB15_training-set.csv": "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
    "UNSW_NB15_testing-set.csv": "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
}


def baixar(pasta=DATA_DIR):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    manifesto = {"origem_dataset": "https://research.unsw.edu.au/projects/unsw-nb15-dataset", "espelho_revisao": REVISAO, "arquivos": {}}
    for nome, esperado in HASHES.items():
        destino = pasta / nome
        url = f"{BASE_URL}/{nome}"
        if destino.exists():
            with destino.open("rb") as arquivo:
                atual = hashlib.file_digest(arquivo, "sha256").hexdigest()
            if atual != esperado:
                raise ValueError(f"{nome}: hash divergente; arquivo existente preservado")
        else:
            temporario = destino.with_suffix(".csv.part")
            try:
                with urlopen(url, timeout=180) as resposta, temporario.open("wb") as arquivo:
                    while bloco := resposta.read(1024 * 1024):
                        arquivo.write(bloco)
                with temporario.open("rb") as arquivo:
                    atual = hashlib.file_digest(arquivo, "sha256").hexdigest()
                if atual != esperado:
                    raise ValueError(f"{nome}: download com hash divergente")
                temporario.replace(destino)
            finally:
                temporario.unlink(missing_ok=True)
        manifesto["arquivos"][nome] = {"url": url, "sha256": esperado, "bytes": destino.stat().st_size}
        print(f"Verificado: {nome}", flush=True)
    (pasta / "origem.json").write_text(json.dumps(manifesto, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destino", type=Path, default=DATA_DIR)
    baixar(parser.parse_args().destino)
