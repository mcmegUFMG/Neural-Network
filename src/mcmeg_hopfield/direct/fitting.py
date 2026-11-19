#!/usr/bin/env python3
"""
direct/fitting.py
==================
Abordagem DIRETA — passo 1: ajusta cada curva de PDD experimental/simulada
ao modelo fechado PDP(x; a, b), obtendo os dois parametros A e B.

PDD(x) = kernel(b)  convoluido  kerma(a),   kerma(y) = exp(-a*y)

O modelo fechado abaixo e a forma analitica dessa convolucao (ver
common/kernel.py para a versao numerica/discretizada usada na reconstrucao).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import erfc


def pdp_model(x, a, b):
    """Closed-form PDD(x) = kernel(b) * kerma(a) (analytical convolution)."""
    rb = np.sqrt(b)
    tv = rb * (-x + a / (2.0 * b))
    return (np.exp(a**2 / (4.0 * b) - a * x) / rb) * erfc(tv) * np.sqrt(np.pi) / 2.0


def fit_pdp(xm, ym):
    """
    Fits (a, b) minimizing quadratic error between pdp_model(x; a, b) and the
    normalized curve ym/max(ym). Tries several initial guesses (Nelder-Mead)
    and keeps the best result.

    Returns
    -------
    a, b, ym_scale
    """
    ym_scale = float(np.max(ym)) or 1.0
    ym_n = ym / ym_scale

    def obj(p):
        a, b = p
        if b <= 0 or a < 0:
            return 1e10
        try:
            pred = pdp_model(xm, a, b)
            return np.sum((pred - ym_n) ** 2) if np.all(np.isfinite(pred)) else 1e10
        except Exception:
            return 1e10

    best_res, best_val = None, np.inf
    guesses = ([0.05, 2.60], [0.03, 2.00], [0.07, 3.00],
              [0.10, 1.50], [0.02, 3.50], [0.01, 5.00])
    for x0 in guesses:
        try:
            r = minimize(obj, x0, method='Nelder-Mead',
                         options={'xatol': 1e-10, 'fatol': 1e-10,
                                  'maxiter': 200000, 'adaptive': True})
            if r.fun < best_val and r.x[1] > 0 and r.x[0] >= 0:
                best_val, best_res = r.fun, r
        except Exception:
            pass

    if best_res is None:
        return 0.05, 2.60, ym_scale
    return best_res.x[0], best_res.x[1], ym_scale


def fit_all_curves(datasets: dict) -> dict:
    """
    Ajusta (a, b) para cada curva de um dict {label: (xm, ym)}.

    Returns
    -------
    dict {label: {'xm', 'ym', 'a', 'b', 'ym_scale'}}
    """
    resultados = {}
    for nome, (xm, ym) in datasets.items():
        a, b, ym_scale = fit_pdp(xm, ym)
        resultados[nome] = dict(xm=xm, ym=ym, a=a, b=b, ym_scale=ym_scale)
    return resultados
