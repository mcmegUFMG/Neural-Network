#!/usr/bin/env python3
"""
common/mayneord.py
===================
Mayneord factor: converts a PDD curve measured at a reference SSD
(SSD1) to a new source-surface distance (SSD2), without needing
novas medidas/simulacoes.

       ( SSD2 + d0    SSD1 + d  ) 2
  F =  ( ---------  *  --------  )
       ( SSD2 + d     SSD1 + d0 )

Used in both direct approach (to build the (field, SSD) grid used in
A/B surfaces) and indirect approach (to build the (field, SSD) grid used
na analise dos kermas F(y)).
"""

import numpy as np


def mayneord_factor(ssd1: float, ssd2: float, d, d0: float):
    """Scaling factor F(d) that converts PDD(d, SSD1) -> PDD(d, SSD2)."""
    return ((ssd2 + d0) / (ssd2 + d) * (ssd1 + d) / (ssd1 + d0)) ** 2


def apply_mayneord(pdd_ref: np.ndarray, depths: np.ndarray,
                    ssd1: float, ssd2: float, d0: float) -> np.ndarray:
    """
    Aplica a correcao de Mayneord a uma curva de PDD inteira.

    Parameters
    ----------
    pdd_ref : PDD (%) at SSD1, shape (N,)
    depths  : profundidades (cm), shape (N,)
    ssd1    : reference SSD (measured/simulated)
    ssd2    : target SSD
    d0      : profundidade de dose maxima (dmax)

    Returns
    -------
    pdd_new : PDD (%) corrected to SSD2, shape (N,)
    """
    pdd_new = np.empty_like(pdd_ref, dtype=float)
    at_dmax = np.isclose(depths, d0, atol=1e-9)
    pdd_new[at_dmax] = 100.0
    if np.any(~at_dmax):
        d = depths[~at_dmax]
        F = mayneord_factor(ssd1, ssd2, d, d0)
        pdd_new[~at_dmax] = pdd_ref[~at_dmax] * F
    return pdd_new


def detect_dmax(depths: np.ndarray, pdd: np.ndarray) -> float:
    """
    Detecta d0 (profundidade de dose maxima) numa curva (depths, pdd):
    usa o ponto onde PDD == 100 se existir, senao o argmax.
    """
    exact = np.where(np.isclose(pdd, 100.0, atol=1e-6))[0]
    if exact.size:
        return float(depths[exact[0]])
    return float(depths[np.argmax(pdd)])
