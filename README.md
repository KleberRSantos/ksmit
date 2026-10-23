# Apuração presidencial TSE e projeções

Projeto que baixa a apuração presidencial do TSE, mantém uma base histórica
e gera três projeções do resultado final dos dois candidatos principais.

## Estrutura

| Caminho | Função |
|---|---|
| `apuracao_tse/coleta.py` | Baixa o JSON, converte em uma linha por cidade e atualiza a base |
| `apuracao_tse/modelos.py` | Três modelos de projeção e montagem da tabela final |
| `main.py` | Executa um ciclo completo: atualiza a base, projeta e salva a tabela |
| `data/base_apuracao.csv` | Base histórica, gerada na primeira execução |
| `data/pesos_referencia.csv` | Entrada obrigatória: peso de cada cidade na votação nacional |
| `saidas/` | Tabela de cada execução (ignorada pelo git) |
| `tests/` | Testes com dados sintéticos |

## Formato da base

| Coluna | Conteúdo |
|---|---|
| `horario_divulgacao` | Horário em que o arquivo foi lido (fuso de Brasília) |
| `uf`, `cidade` | Localidade |
| `percentual_apurado` | Percentual de seções apuradas na cidade |
| `percentual_apurado_brasil` | Percentual de seções apuradas no Brasil naquele horário |
| `votos_<candidato>` | Votos de cada candidato na cidade (uma coluna por candidato) |

Cada execução acrescenta um bloco de linhas. Repetir o mesmo horário não duplica dados.

## Arquivo de pesos

`data/pesos_referencia.csv` tem as colunas `uf`, `cidade` e `peso`. O peso é a
participação da cidade no total de votos válidos do Brasil, na eleição de referência.
Os pesos são normalizados para somar 1. Cidades sem peso interrompem a execução.

## Como rodar

```bash
pip install -r requirements.txt
export TSE_URL_RESULTADOS="<URL do JSON de apuração>"
python main.py --pesos data/pesos_referencia.csv
python -m pytest
```

## Modelos

Regra comum: a parte já apurada usa o realizado. Para a parte não apurada,
cada cidade recebe votos proporcionais ao seu peso nacional, divididos segundo
a participação estimada por cada modelo.

| Modelo | Estimativa da participação não apurada | Saída extra |
|---|---|---|
| 1. Razão ponderada nacional | Participação geral do que já foi apurado, ponderada por peso e fração apurada | Nenhuma |
| 2. Estratificado por UF | Participação da UF, encolhida para a média nacional (constante `PESO_PRIORI_UF`) | Nenhuma |
| 3. Regressão com simulação | Regressão da participação no log do peso da cidade, com 500 reamostragens e ruído dos resíduos | Intervalo de 10% a 90% |

O modelo 3 cai para o modelo 1 quando há menos de 30 cidades com apuração iniciada.

## Limitações conhecidas

1. O mapeamento das chaves do JSON (`CHAVES` em `coleta.py`) não foi validado contra um arquivo real do TSE. Confira na primeira execução.
2. Os pesos vêm de uma eleição anterior. Se o perfil de voto mudou, as projeções herdam esse viés.
3. Só o modelo 3 tem intervalo. Os modelos 1 e 2 são estimativas pontuais.
4. Votos do exterior não estão tratados. Se o arquivo de pesos não cobrir essas localidades, a execução para com erro.
