#!/usr/bin/env python3
"""
direct/surfaces.py
===================
Abordagem DIRETA — passo 2: a partir de um CSV base de PDD (profundidade x
field), measured at a single SSD1, generates variations for other SSDs via factor of
Mayneord (common/mayneord.py), ajusta o modelo PDP (a, b) para cada par
(field, SSD) and builds A(field, SSD) and B(field, SSD) surfaces.
"""

import numpy as np
import pandas as pd

from .fitting import fit_pdp, pdp_model
from common.mayneord import apply_mayneord


def detect_dmax_xy(xm: np.ndarray, ym: np.ndarray) -> float:
    """Detecta d0 (profundidade de dose maxima) para uma curva (xm, ym)."""
    exact = np.where(np.isclose(ym, 100.0, atol=1e-6))[0]
    if exact.size:
        return float(xm[exact[0]])
    return float(xm[np.argmax(ym)])


def gerar_superficie_ab(datasets: dict, params: dict, param_type: str,
                         ssd1: float, ssds, fixed_d0: float = None,
                         verbose: bool = True):
    """
    For each field in the base CSV and each SSD in `ssds`, applies the correction of
    Mayneord (a partir da curva medida em ssd1) e ajusta o modelo PDP (a, b).

    Parameters
    ----------
    datasets   : dict {label: (xm, ym)}          (de common.io_utils.load_csv)
    params     : dict {label: valor do campo}     (de common.io_utils.load_csv)
    param_type : nome do parametro de campo (ex.: "Field size")
    ssd1       : reference SSD of base CSV (cm)
    ssds       : list/array of output SSDs (cm) -- may include ssd1 itself
    fixed_d0   : se informado, usa esse d0 fixo para todos os campos;
                 senao, detecta automaticamente o d0 de cada campo
    verbose    : prints progress field by field / SSD by SSD

    Returns
    -------
    df_tidy : DataFrame longo, uma linha por (campo, ssd)
    """
    rows = []
    total = len(datasets) * len(ssds)
    contador = 0

    for campo, (xm, ym) in datasets.items():
        d0 = fixed_d0 if fixed_d0 is not None else detect_dmax_xy(xm, ym)

        for ssd2 in ssds:
            contador += 1
            if abs(ssd2 - ssd1) < 1e-9:
                pdd_corrigida = ym
            else:
                pdd_corrigida = apply_mayneord(ym, xm, ssd1, ssd2, d0)

            a, b, ym_scale = fit_pdp(xm, pdd_corrigida)
            ajustada = pdp_model(xm, a, b) * ym_scale
            residual = float(np.sum((ajustada - pdd_corrigida) ** 2))

            rows.append({
                'campo': campo,
                param_type: params[campo],
                'ssd': ssd2,
                'd0': d0,
                'a': a,
                'b': b,
                'ym_scale': ym_scale,
                'residual_SS': residual,
            })

            if verbose:
                print(f"  [{contador:>4}/{total}] field={campo:<8} "
                      f"SSD={ssd2:>6.1f}  ->  a={a:.6f}  b={b:.6f}")

    df_tidy = pd.DataFrame(rows).sort_values([param_type, 'ssd']).reset_index(drop=True)
    return df_tidy


def montar_superficies(df_tidy: pd.DataFrame, param_type: str):
    """
    Converte a tabela tidy em duas matrizes (pivot tables):
      superficie_a: linhas = valor do parametro de campo, colunas = ssd
      superficie_b: idem, para o parametro b
    """
    superficie_a = df_tidy.pivot(index=param_type, columns='ssd', values='a')
    superficie_b = df_tidy.pivot(index=param_type, columns='ssd', values='b')
    return superficie_a, superficie_b
