"""Coleta a apuração presidencial do TSE e mantém a base histórica em CSV."""

import re
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

TEMPO_LIMITE_SEGUNDOS = 30

# Nomes das chaves do JSON publicado pelo TSE.
# Conferir contra um arquivo real antes da primeira execução de produção.
CHAVES = {
    "percentual_brasil": "pst",
    "ufs": "abr",
    "uf_nome": "nm",
    "municipios": "mu",
    "municipio_nome": "nm",
    "percentual_municipio": "pst",
    "candidatos": "cand",
    "candidato_nome": "nm",
    "candidato_votos": "vap",
}


def nome_coluna(nome_candidato: str) -> str:
    """Transforma o nome do candidato em nome de coluna, ex.: 'votos_lula_da_silva'."""
    sem_acento = unicodedata.normalize("NFKD", nome_candidato).encode("ascii", "ignore").decode()
    return "votos_" + re.sub(r"[^a-z0-9]+", "_", sem_acento.lower()).strip("_")


def _numero(valor) -> float:
    return float(str(valor).replace(",", "."))


def baixar_json(url: str) -> dict:
    resposta = requests.get(url, timeout=TEMPO_LIMITE_SEGUNDOS)
    resposta.raise_for_status()
    return resposta.json()


def extrair_snapshot(payload: dict, horario: datetime) -> pd.DataFrame:
    """Uma linha por município, com os votos de cada candidato e o percentual apurado."""
    pct_brasil = _numero(payload[CHAVES["percentual_brasil"]])
    horario_texto = horario.isoformat(timespec="seconds")

    linhas = []
    for uf in payload[CHAVES["ufs"]]:
        for municipio in uf[CHAVES["municipios"]]:
            linha = {
                "horario_divulgacao": horario_texto,
                "uf": uf[CHAVES["uf_nome"]],
                "cidade": municipio[CHAVES["municipio_nome"]],
                "percentual_apurado": _numero(municipio[CHAVES["percentual_municipio"]]),
                "percentual_apurado_brasil": pct_brasil,
            }
            for candidato in municipio[CHAVES["candidatos"]]:
                coluna = nome_coluna(candidato[CHAVES["candidato_nome"]])
                linha[coluna] = int(candidato[CHAVES["candidato_votos"]])
            linhas.append(linha)

    if not linhas:
        raise ValueError("JSON sem municípios. Verifique o mapeamento CHAVES.")
    return pd.DataFrame(linhas)


def atualizar_base(snapshot: pd.DataFrame, caminho: Path) -> pd.DataFrame:
    """Acrescenta o snapshot à base. Repetir o mesmo horário não duplica linhas."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if caminho.exists():
        base = pd.concat([pd.read_csv(caminho), snapshot], ignore_index=True)
    else:
        base = snapshot

    base = base.drop_duplicates(["horario_divulgacao", "uf", "cidade"], keep="last")
    base = base.sort_values(["horario_divulgacao", "uf", "cidade"]).reset_index(drop=True)
    base.to_csv(caminho, index=False)
    return base
