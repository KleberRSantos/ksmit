from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from apuracao_tse import coleta

FUSO = ZoneInfo("America/Sao_Paulo")


def _payload():
    def municipio(nome, pct, votos_a, votos_b):
        return {
            "nm": nome,
            "pst": pct,
            "cand": [
                {"nm": "LUIZ A", "vap": votos_a},
                {"nm": "JOSÉ B", "vap": votos_b},
            ],
        }

    return {
        "pst": "12,50",
        "abr": [
            {"nm": "SP", "mu": [municipio("SAO PAULO", "20,00", 100, 80)]},
            {"nm": "MG", "mu": [municipio("BELO HORIZONTE", "0,00", 0, 0)]},
        ],
    }


def test_extrai_uma_linha_por_municipio_com_colunas_de_votos():
    horario = datetime(2026, 10, 25, 17, 0, tzinfo=FUSO)

    snapshot = coleta.extrair_snapshot(_payload(), horario)

    assert len(snapshot) == 2
    assert {"votos_luiz_a", "votos_jose_b"}.issubset(snapshot.columns)
    assert snapshot.loc[0, "percentual_apurado"] == 20.0
    assert snapshot.loc[0, "percentual_apurado_brasil"] == 12.5
    assert snapshot.loc[0, "horario_divulgacao"] == "2026-10-25T17:00:00-03:00"


def test_nome_de_coluna_remove_acentos_e_espacos():
    assert coleta.nome_coluna("JOSÉ DA SILVA") == "votos_jose_da_silva"


def test_atualizar_base_nao_duplica_mesmo_horario(tmp_path):
    horario = datetime(2026, 10, 25, 17, 0, tzinfo=FUSO)
    snapshot = coleta.extrair_snapshot(_payload(), horario)
    caminho = tmp_path / "base.csv"

    coleta.atualizar_base(snapshot, caminho)
    base = coleta.atualizar_base(snapshot, caminho)

    assert len(base) == 2
    assert len(pd.read_csv(caminho)) == 2
