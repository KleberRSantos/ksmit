"""Atualiza a base da apuração presidencial e gera as projeções do momento.

Uso:
    TSE_URL_RESULTADOS=<url do JSON de apuração> python main.py
    python main.py --url <url do JSON de apuração> --pesos data/pesos_referencia.csv
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from apuracao_tse import coleta, modelos

RAIZ = Path(__file__).resolve().parent
CAMINHO_BASE = RAIZ / "data" / "base_apuracao.csv"
CAMINHO_PESOS = RAIZ / "data" / "pesos_referencia.csv"
PASTA_SAIDAS = RAIZ / "saidas"
FUSO = ZoneInfo("America/Sao_Paulo")


def montar_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("TSE_URL_RESULTADOS"), help="URL do JSON de apuração")
    parser.add_argument("--pesos", type=Path, default=CAMINHO_PESOS, help="CSV com uf, cidade e peso")
    return parser.parse_args()


def main() -> None:
    argumentos = montar_argumentos()
    if not argumentos.url:
        raise SystemExit("Informe a URL com --url ou na variável TSE_URL_RESULTADOS.")

    horario = datetime.now(FUSO)
    payload = coleta.baixar_json(argumentos.url)
    snapshot = coleta.extrair_snapshot(payload, horario)
    base = coleta.atualizar_base(snapshot, CAMINHO_BASE)
    print(f"Base atualizada: {len(base)} linhas em {CAMINHO_BASE}")

    pesos = pd.read_csv(argumentos.pesos)
    estado = modelos.estado_atual(base, pesos)
    finais, intervalo = modelos.projetar(estado)
    tabela = modelos.montar_tabela(estado, finais, intervalo)

    percentual_brasil = snapshot["percentual_apurado_brasil"].iloc[0]
    municipios_apurados = int((estado.fracao > 0).sum())
    cobertura = 100 * (estado.peso * estado.fracao).sum()

    print(f"Horário da divulgação: {horario:%d/%m/%Y %H:%M:%S}")
    print(f"Percentual apurado no Brasil: {percentual_brasil:.2f}%")
    print(f"Municípios com apuração iniciada: {municipios_apurados} de {len(estado.peso)}")
    print(f"Cobertura ponderada pelo peso nacional: {cobertura:.2f}%")
    if intervalo is None:
        print("Modelo 3 sem intervalo: poucos municípios apurados para a regressão.")
    print()
    print(tabela.to_string(index=False))

    PASTA_SAIDAS.mkdir(exist_ok=True)
    caminho_saida = PASTA_SAIDAS / f"projecao_{horario:%Y%m%d_%H%M%S}.csv"
    tabela.assign(horario_divulgacao=horario.isoformat(timespec="seconds")).to_csv(caminho_saida, index=False)
    print(f"\nTabela salva em {caminho_saida}")


if __name__ == "__main__":
    main()
