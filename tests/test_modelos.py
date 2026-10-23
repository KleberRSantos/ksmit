import numpy as np
import pandas as pd
import pytest

from apuracao_tse import modelos


def _cenario(fracao_maxima: float, semente: int = 0, n_municipios: int = 60):
    """Base sintética: municípios em 4 UFs, com preferências ligadas ao tamanho."""
    rng = np.random.default_rng(semente)
    ufs = ["SP", "MG", "BA", "PA"]
    linhas, pesos = [], []
    horario = "2026-10-25T17:00:00-03:00"

    tamanho = rng.lognormal(mean=8, sigma=1.2, size=n_municipios)
    peso_bruto = tamanho / tamanho.sum()

    for i in range(n_municipios):
        uf = ufs[i % 4]
        cidade = f"CIDADE {i:03d}"
        pesos.append({"uf": uf, "cidade": cidade, "peso": peso_bruto[i]})

        fracao = rng.uniform(0, fracao_maxima)
        share_a = np.clip(0.45 + 0.04 * np.log(peso_bruto[i] * n_municipios) + rng.normal(0, 0.03), 0.05, 0.95)
        total = peso_bruto[i] * 1_000_000 * fracao
        votos_a = int(round(total * share_a))
        votos_b = int(round(total * (1 - share_a) * 0.9))
        votos_c = int(round(total * (1 - share_a) * 0.1))
        linhas.append({
            "horario_divulgacao": horario,
            "uf": uf,
            "cidade": cidade,
            "percentual_apurado": 100 * fracao,
            "percentual_apurado_brasil": 0.0,
            "votos_a": votos_a,
            "votos_b": votos_b,
            "votos_c": votos_c,
        })
    return pd.DataFrame(linhas), pd.DataFrame(pesos)


def test_apuracao_completa_reproduz_o_realizado():
    base, pesos = _cenario(fracao_maxima=1.0)
    base["percentual_apurado"] = 100.0
    estado = modelos.estado_atual(base, pesos)

    finais, _ = modelos.projetar(estado)
    realizado = estado.votos.sum(axis=0)

    for nome, votos in finais.items():
        np.testing.assert_allclose(votos, realizado, rtol=1e-9, err_msg=nome)


def test_participacoes_somam_um():
    base, pesos = _cenario(fracao_maxima=0.5)
    estado = modelos.estado_atual(base, pesos)

    for share in (modelos.modelo_1(estado), modelos.modelo_2(estado)):
        np.testing.assert_allclose(share.sum(axis=1), 1.0)

    share_3, _ = modelos.modelo_3(estado)
    np.testing.assert_allclose(share_3.sum(axis=1), 1.0)


def test_modelo_1_e_media_ponderada_pelo_apurado():
    base, pesos = _cenario(fracao_maxima=0.5)
    estado = modelos.estado_atual(base, pesos)

    total, part = modelos._participacao_observada(estado)
    w = estado.peso * estado.fracao * (total > 0)
    esperado = (w[:, None] * part).sum(axis=0) / w.sum()

    np.testing.assert_allclose(modelos.modelo_1(estado)[0], esperado)


def test_sem_apuracao_levanta_erro():
    base, pesos = _cenario(fracao_maxima=0.5)
    base["percentual_apurado"] = 0.0
    estado = modelos.estado_atual(base, pesos)

    with pytest.raises(ValueError, match="Nenhum município apurado"):
        modelos.projetar(estado)


def test_municipio_sem_peso_levanta_erro():
    base, pesos = _cenario(fracao_maxima=0.5)
    pesos = pesos.iloc[1:]

    with pytest.raises(ValueError, match="sem peso de referência"):
        modelos.estado_atual(base, pesos)


def test_tabela_tem_apuracao_e_tres_modelos():
    base, pesos = _cenario(fracao_maxima=0.6)
    estado = modelos.estado_atual(base, pesos)
    finais, intervalo = modelos.projetar(estado)

    tabela = modelos.montar_tabela(estado, finais, intervalo)

    assert len(tabela) == 2
    colunas = {"candidato", "apuracao_tse_pct", "modelo_1_pct", "modelo_2_pct", "modelo_3_pct"}
    assert colunas.issubset(tabela.columns)
    assert (tabela["modelo_3_p10_pct"] <= tabela["modelo_3_pct"] + 1e-6).all()
