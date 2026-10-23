"""Modelos de projeção do resultado final a partir da apuração parcial.

Regra comum aos três modelos:
1. Cada município recebe um volume de votos válidos proporcional ao seu peso
   na votação nacional (peso de referência).
2. A parte já apurada usa o realizado. A parte não apurada usa a participação
   estimada por cada modelo.
3. Os modelos diferem só na estimativa da participação dos candidatos
   para o que falta apurar.
"""

import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Peso (em participação nacional) que a média nacional recebe no encolhimento da UF.
PESO_PRIORI_UF = 0.02
MINIMO_MUNICIPIOS_REGRESSAO = 30
N_SIMULACOES = 500
SEMENTE = 42


@dataclass(frozen=True)
class Estado:
    votos: np.ndarray        # (municípios, candidatos) votos apurados
    fracao: np.ndarray       # (municípios,) fração das seções apuradas, entre 0 e 1
    peso: np.ndarray         # (municípios,) participação no total de votos válidos, soma 1
    uf: np.ndarray           # (municípios,)
    candidatos: list         # nomes das colunas de votos, na ordem de `votos`


def colunas_votos(base: pd.DataFrame) -> list:
    return [coluna for coluna in base.columns if coluna.startswith("votos_")]


def _sem_acento_maiuscula(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return sem_acento.strip().upper()


def _chave_municipio(tabela: pd.DataFrame) -> pd.Series:
    return tabela["uf"].str.upper() + "|" + tabela["cidade"].map(_sem_acento_maiuscula)


def estado_atual(base: pd.DataFrame, pesos: pd.DataFrame) -> Estado:
    """Usa o último snapshot de cada município e junta o peso de referência."""
    colunas = colunas_votos(base)
    ultimo = base.sort_values("horario_divulgacao").drop_duplicates(["uf", "cidade"], keep="last")
    ultimo = ultimo.assign(chave=_chave_municipio(ultimo))

    referencia = pesos.assign(chave=_chave_municipio(pesos))
    junto = ultimo.merge(referencia[["chave", "peso"]], on="chave", how="left", validate="one_to_one")

    sem_peso = junto["peso"].isna()
    if sem_peso.any():
        exemplos = junto.loc[sem_peso, ["uf", "cidade"]].head(5).values.tolist()
        raise ValueError(f"{sem_peso.sum()} municípios sem peso de referência. Exemplos: {exemplos}")

    peso = junto["peso"].to_numpy(float)
    peso = peso / peso.sum()

    return Estado(
        votos=junto[colunas].fillna(0).to_numpy(float),
        fracao=np.clip(junto["percentual_apurado"].to_numpy(float) / 100, 0, 1),
        peso=peso,
        uf=junto["uf"].to_numpy(),
        candidatos=colunas,
    )


def _participacao_observada(estado: Estado):
    total = estado.votos.sum(axis=1)
    part = np.divide(
        estado.votos,
        total[:, None],
        out=np.zeros_like(estado.votos),
        where=total[:, None] > 0,
    )
    return total, part


def _normalizar(matriz: np.ndarray) -> np.ndarray:
    soma = matriz.sum(axis=1, keepdims=True)
    k = matriz.shape[1]
    return np.where(soma > 0, matriz / np.where(soma > 0, soma, 1), 1 / k)


def _misturar(estado: Estado, share_modelo: np.ndarray, part: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Para município parcialmente apurado, combina o realizado com a estimativa do modelo."""
    f = estado.fracao[:, None]
    mistura = f * part + (1 - f) * share_modelo
    return np.where(total[:, None] > 0, mistura, share_modelo)


def _exigir_apuracao(estado: Estado) -> None:
    if (estado.peso * estado.fracao).sum() <= 0:
        raise ValueError("Nenhum município apurado. Não há base para projetar.")


def _finais(estado: Estado, share_modelo: np.ndarray) -> np.ndarray:
    """Votos finais projetados por candidato (inclui os já apurados)."""
    total, part = _participacao_observada(estado)
    cobertura = (estado.peso * estado.fracao).sum()

    votos_validos_brasil = total.sum() / cobertura
    falta = (1 - estado.fracao) * estado.peso * votos_validos_brasil
    mistura = _misturar(estado, share_modelo, part, total)
    return estado.votos.sum(axis=0) + (falta[:, None] * mistura).sum(axis=0)


def modelo_1(estado: Estado) -> np.ndarray:
    """Razão ponderada nacional: o que falta se parece com o total já apurado."""
    total, part = _participacao_observada(estado)
    w = estado.peso * estado.fracao * (total > 0)
    part_nacional = (w[:, None] * part).sum(axis=0) / w.sum()
    return np.tile(part_nacional, (len(estado.peso), 1))


def modelo_2(estado: Estado) -> np.ndarray:
    """Estratificado por UF, com encolhimento para a média nacional (média bayesiana simples)."""
    total, part = _participacao_observada(estado)
    nacional = modelo_1(estado)[0]
    w = estado.peso * estado.fracao * (total > 0)

    ufs, grupo = np.unique(estado.uf, return_inverse=True)
    n_uf = np.bincount(grupo, weights=w, minlength=len(ufs))
    soma_uf = np.zeros((len(ufs), estado.votos.shape[1]))
    np.add.at(soma_uf, grupo, w[:, None] * part)

    share_uf = (soma_uf + PESO_PRIORI_UF * nacional) / (n_uf[:, None] + PESO_PRIORI_UF)
    return share_uf[grupo]


def _X(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x])


def _ajustar(x: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mínimos quadrados ponderados de y em x, com intercepto. Devolve coeficientes (2, candidatos)."""
    raiz = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(_X(x) * raiz, y * raiz, rcond=None)
    return beta


def _prever(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _normalizar(np.clip(_X(x) @ beta, 0, None))


def modelo_3(estado: Estado):
    """Regressão da participação no log do peso do município. Vale para o que ainda não foi apurado.

    Devolve a estimativa central e um intervalo de simulação. Com poucos municípios apurados,
    cai para o modelo 1 e devolve `None` no intervalo.
    """
    total, part = _participacao_observada(estado)
    ok = np.flatnonzero((total > 0) & (estado.fracao > 0))
    if ok.size < MINIMO_MUNICIPIOS_REGRESSAO:
        return modelo_1(estado), None

    x = np.log(estado.peso)
    w = estado.peso * estado.fracao
    beta = _ajustar(x[ok], w[ok], part[ok])
    share_central = _prever(beta, x)
    intervalo = _simular_modelo_3(estado, x, w, part, ok)
    return share_central, intervalo


def _simular_modelo_3(estado: Estado, x, w, part, ok) -> np.ndarray:
    """Monte Carlo: reamostra os municípios apurados, refaz o ajuste e soma o ruído dos resíduos."""
    rng = np.random.default_rng(SEMENTE)
    n_candidatos = estado.votos.shape[1]
    finais = np.empty((N_SIMULACOES, n_candidatos))

    for b in range(N_SIMULACOES):
        amostra = rng.choice(ok, size=ok.size, replace=True)
        beta = _ajustar(x[amostra], w[amostra], part[amostra])
        residuos = part[amostra] - _X(x[amostra]) @ beta
        ruido = residuos[rng.integers(0, ok.size, size=len(x))]
        share = _normalizar(np.clip(_X(x) @ beta + ruido, 0, None))
        finais[b] = _finais(estado, share)
    return finais


def projetar(estado: Estado):
    """Roda os três modelos. Devolve votos finais por modelo e o intervalo do modelo 3."""
    _exigir_apuracao(estado)
    share_3, intervalo = modelo_3(estado)
    finais = {
        "modelo_1": _finais(estado, modelo_1(estado)),
        "modelo_2": _finais(estado, modelo_2(estado)),
        "modelo_3": _finais(estado, share_3),
    }
    return finais, intervalo


def montar_tabela(estado: Estado, finais: dict, intervalo, quantidade_principais: int = 2) -> pd.DataFrame:
    """Uma linha por candidato principal, com apuração TSE e as três projeções."""
    realizado = estado.votos.sum(axis=0)
    principais = np.argsort(realizado)[::-1][:quantidade_principais]

    if intervalo is not None:
        pct_simulado = 100 * intervalo / intervalo.sum(axis=1, keepdims=True)

    linhas = []
    for i in principais:
        linha = {
            "candidato": estado.candidatos[i].removeprefix("votos_").replace("_", " ").upper(),
            "apuracao_tse_votos": int(realizado[i]),
            "apuracao_tse_pct": 100 * realizado[i] / realizado.sum(),
        }
        for nome, votos in finais.items():
            linha[f"{nome}_votos"] = votos[i]
            linha[f"{nome}_pct"] = 100 * votos[i] / votos.sum()

        if intervalo is not None:
            linha["modelo_3_p10_pct"], linha["modelo_3_p90_pct"] = np.percentile(pct_simulado[:, i], [10, 90])
        else:
            linha["modelo_3_p10_pct"], linha["modelo_3_p90_pct"] = np.nan, np.nan
        linhas.append(linha)

    return pd.DataFrame(linhas).round(2)
