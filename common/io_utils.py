#!/usr/bin/env python3
"""
common/io_utils.py
===================
Funcoes de entrada/saida compartilhadas entre a abordagem direta e a indireta:

  - load_csv(...)   : reads a PDD CSV (depth x field/SSD), detecting
                       separador e tipo de parametro automaticamente.
  - prompt(...)      : prompt de terminal com valor padrao.
  - save_df(...)     : salva DataFrame em CSV + .xlsx irmao.

IMPORTANTE sobre `save_df` (mudou):
  - Ate pouco tempo atras, `save_df` usava por padrao a convencao pt-BR
    (';' como separador de coluna, ',' como decimal). Essa convencao SO e
    lida corretamente se o locale regional de quem abre o arquivo tambem
    for pt-BR -- caso contrario, a ',' decimal e mal interpretada como
    separador de milhar (ex.: um numero como 118,142857142857 sendo lido
    como o inteiro 118142857142857, depois exibido com separadores de
    milhar: "11,814,285,714,285,700"). Isso ja causou corrupcao real de
    dados exportados.
  - Agora `save_df` usa por padrao a convencao internacional (',' separador
    de coluna, '.' decimal), que qualquer ferramenta interpreta do mesmo
    jeito independente do locale. Alem disso, `save_df` grava tambem um
    .xlsx irmao (mesmo nome, extensao .xlsx) com os mesmos dados: no .xlsx
    os numeros sao gravados como valor numerico real (formato binario/
    OOXML), nao como texto que precisa ser reinterpretado -- e por isso a
    forma mais segura de garantir que o valor abra certo em qualquer
    Excel/maquina.

Convencao adotada para os CSVs de ENTRADA (load_csv), que permanece
inalterada (dados de laboratorio/equipamento seguem sendo pt-BR):
  - separador de colunas   : detectado automaticamente (';' ou ',')
  - separador decimal      : ','
"""

import os
import re
import numpy as np
import pandas as pd

SEP_DEFAULT = ','
DECIMAL_DEFAULT = '.'


# ------------------------------------------------------------------
#  Parameter detection (field, SSD, energy...) from names
#  das colunas de um CSV
# ------------------------------------------------------------------

def _extract_param(label):
    label = str(label).strip()
    m = re.fullmatch(r'(\d+)[xX](\d+)', label)
    if m:
        return int(m.group(1))
    nums = re.findall(r'\d+(?:[.,]\d+)?', label)
    if nums:
        return float(nums[0].replace(',', '.'))
    raise ValueError(f"Could not extract numeric parameter from '{label}'")


def _detect_param_type(columns):
    sample = ' '.join(str(c) for c in columns).lower()
    if re.search(r'\d+x\d+', sample):
        return "Field size", "cm"
    if 'ssd' in sample:
        return "SSD", "cm"
    if 'mv' in sample or 'mev' in sample:
        return "Energy", "MV"
    return "Parameter", ""


def load_csv(filepath):
    """
    Le um CSV de PDD (1a coluna = profundidade, demais colunas = curvas para
    each field/SSD/energy value). Detects separator (',' or ';') and decimal
    (assume decimal ',' no arquivo de origem, convencao brasileira).

    Returns
    -------
    datasets    : dict {label: (xm, ym)}
    params      : dict {label: valor numerico do parametro}
    param_type  : str  ("Field size", "SSD", "Energy" or "Parameter")
    param_unit  : str  ("cm", "MV" ou "")
    """
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        first = f.readline()
    sep = ';' if first.count(';') > first.count(',') else ','
    df = pd.read_csv(filepath, sep=sep, encoding='utf-8-sig', decimal=',')
    if df.shape[1] == 1:
        raise ValueError("CSV with only 1 column. Check the separator.")
    xm = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values

    param_type, param_unit = _detect_param_type(df.columns[1:])

    datasets, params = {}, {}
    for col in df.columns[1:]:
        label = str(col).strip()
        if not label:
            break
        try:
            val = _extract_param(label)
        except ValueError:
            continue
        ym_raw = pd.to_numeric(df[col], errors='coerce').values
        mask = ~np.isnan(ym_raw)
        ym = ym_raw[mask]
        datasets[label] = (xm[:len(ym)], ym)
        params[label] = val

    return datasets, params, param_type, param_unit


# ------------------------------------------------------------------
#  Prompt de terminal
# ------------------------------------------------------------------

def prompt(text, default=None):
    hint = f" [{default}]" if default is not None else ""
    raw = input(f"{text}{hint}: ").strip()
    return raw if raw else (str(default) if default is not None else None)


def prompt_float(text, default=None):
    return float(prompt(text, default))


def prompt_int(text, default=None):
    return int(prompt(text, default))


def prompt_yesno(text, default="n"):
    return prompt(text, default).strip().lower().startswith('s')


# ------------------------------------------------------------------
#  CSV saving (convencao BR)
# ------------------------------------------------------------------

def save_df(df, path, sep=SEP_DEFAULT, decimal=DECIMAL_DEFAULT, index=False):
    """Salva `df` em CSV (padrao internacional: ',' separador, '.' decimal,
    por default) e, ao lado, um .xlsx irmao com os mesmos dados.

    O .xlsx e a garantia real contra corrupcao de numero por locale: nele
    os valores sao gravados em formato numerico binario (OOXML), entao o
    Excel exibe cada numero formatado conforme o locale de quem esta
    olhando (usando ',' ou '.' conforme a config da maquina), mas o VALOR
    em si nunca e mal interpretado -- ao contrario do CSV, que e sempre
    texto e portanto depende de alguem (planilha, script) interpretar o
    separador decimal correto na hora de ler.

    Se `path` nao terminar em '.csv', o .xlsx irmao e gravado como
    `path + '.xlsx'`; normalmente `path` ja vem com '.csv' de quem chama.
    """
    df.to_csv(path, sep=sep, decimal=decimal, index=index)
    print(f"  OK CSV saved: {path}")

    caminho_xlsx = path[:-4] + ".xlsx" if path.lower().endswith(".csv") else path + ".xlsx"
    try:
        df.to_excel(caminho_xlsx, index=index)
        print(f"  OK XLSX saved: {caminho_xlsx}")
    except Exception as e:
        print(f"  AVISO: nao foi possivel gerar .xlsx irmao de {path}: {e}")
