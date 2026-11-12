#!/usr/bin/env python3
"""
indirect/kerma_batch.py
========================
INDIRECT approach — step 1: for each field in the base CSV and each SSD of a
grade (gerada via Mayneord, common/mayneord.py, exatamente como na abordagem
direta), roda a rede de Hopfield (ou OLS, mais rapido) para extrair o kerma
numerico F(y) daquela curva.

The result is a collection of kermas F(y; field, SSD) to be analyzed in
indirect/kerma_variation.py.
"""

import numpy as np
import pandas as pd

from direct.fitting import fit_pdp
from direct.surfaces import detect_dmax_xy
from common.mayneord import apply_mayneord
from .hopfield import extrair_kerma_hopfield, extrair_kerma_fourier


def gerar_kermas_batch(datasets: dict, params: dict, param_type: str,
                        ssd1: float, ssds, fixed_d0: float = None,
                        N: int = 64, t_end: float = 100.0, y_max: float = 40.0,
                        metodo: str = 'hopfield', margem: float = 2.0,
                        regularizar_fourier: bool = True, lam_fourier: float = 1e-3,
                        verbose: bool = True):
    """
    For each (field, SSD), generates the PDD curve via Mayneord (from the curve
    base medida em ssd1), ajusta (a, b) [necessarios para montar o kernel] e
    extrai o kerma numerico F(y) via Hopfield ou via deconvolucao analitica
    de Fourier.

    Parameters
    ----------
    metodo : 'hopfield' (rede de Hopfield, mais fiel a ruido/nao-linearidades)
        ou 'fourier' (deconvolucao direta no dominio de Fourier, assumindo
        kernel espacialmente invariante -- ver indirect.hopfield.extrair_kerma_fourier).
    margem : folga de amplitude do Hopfield acima de ym_scale (so usada se
        method='hopfield'; ver docstring de indirect.hopfield.run_hopfield).
        Sem essa folga a rede fica presa ao teto do chute fechado (a,b) e a
        PDD reconstruida tende a sair sistematicamente abaixo do esperado
        depois do buildup.
    regularizar_fourier : se True (recomendado para dados experimentais),
        aplica regularizacao Wiener/Tikhonov na deconvolucao de Fourier, que
        e muito sensivel a ruido sem essa regularizacao. So usada se
        method='fourier'.
    lam_fourier : intensidade da regularizacao de Fourier (so usada se
        method='fourier' e regularizar_fourier=True).

    Returns
    -------
    df_tidy   : DataFrame com uma linha por (campo, ssd), incluindo a,b,d0
    kermas    : dict {(campo, ssd): {'y': array, 'F': array, 'pdd_rec': array, 'ok': bool}}
    """
    assert metodo in ('hopfield', 'fourier')

    rows = []
    kermas = {}
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

            if metodo == 'hopfield':
                y, F, pdd_rec, ok = extrair_kerma_hopfield(
                    xm, pdd_corrigida, a, b, ym_scale, N=N, t_end=t_end,
                    y_max=y_max, margem=margem)
            else:
                y, F, pdd_rec, ok = extrair_kerma_fourier(
                    xm, pdd_corrigida, b, ym_scale, N=N, y_max=y_max,
                    regularizar=regularizar_fourier, lam=lam_fourier)

            err = float(np.linalg.norm(pdd_rec - pdd_corrigida) ** 2)

            rows.append({
                'campo': campo, param_type: params[campo], 'ssd': ssd2,
                'd0': d0, 'a': a, 'b': b, 'ym_scale': ym_scale,
                'reconstruction_error_sq': err, 'ok': ok,
            })
            kermas[(campo, ssd2)] = {'y': y, 'F': F, 'pdd_rec': pdd_rec, 'ok': ok,
                                     # target curve (measured / Mayneord): needed by the V2 PDD-space fit
                                     'x_target': np.asarray(xm, float), 'pdd_target': np.asarray(pdd_corrigida, float)}

            if verbose:
                print(f"  [{contador:>4}/{total}] field={campo:<8} SSD={ssd2:>6.1f}  "
                      f"->  a={a:.6f}  b={b:.6f}  error={err:.4e}  method={metodo}")

    df_tidy = pd.DataFrame(rows).sort_values([param_type, 'ssd']).reset_index(drop=True)
    return df_tidy, kermas
