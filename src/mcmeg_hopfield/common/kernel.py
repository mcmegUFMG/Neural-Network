#!/usr/bin/env python3
"""
common/kernel.py
================
Nucleo de convolucao compartilhado:

    PDD(x)  ~=  integral  K(x, y) * f(y) dy
    K(x, y) = exp(-b * (x - y)^2)

Onde:
  - b  parametriza o kernel (largura do espalhamento), vem do ajuste PDP (a,b)
  - f(y) e a "kerma": exp(-a*y) na abordagem direta (forma fechada fixa),
    ou uma funcao geral F(y) obtida numericamente na abordagem indireta
    (Hopfield / minimos quadrados).

Este modulo so constroi a matriz do kernel discretizado; a forma de f(y)
(fechada vs. numerica) e decidida por quem chama.
"""

import numpy as np


def build_kernel_matrix(xm: np.ndarray, b: float, N: int = 64, y_max: float = 40.0):
    """
    Discretizes the Gaussian kernel K(x, y) = exp(-b*(x-y)^2) on a uniform
    grid of N cell-centred nodes y_j = (j + 0.5)*h in [0, y_max], using the
    composite MIDPOINT rule: every node has weight h.

    NOTE (fix): this used to halve the weight of the first and last nodes
    (a trapezoid-rule convention). That convention is only valid when the
    nodes sit ON the interval endpoints (y_0 = 0, y_{N-1} = y_max); with
    cell-centred nodes it drops half of the first and last cells, which
    under-estimated the PDD near the surface (x < ~1.5 cm) by up to ~25
    percentage points and made the Hopfield network double F(y) in the
    first cell to compensate. The y grid is unchanged, so every caller
    (Hopfield, Fourier, reconstructions) keeps working as before.

    Returns
    -------
    y : y grid, shape (N,)
    K : kernel matrix, shape (len(xm), N)
    """
    h = y_max / N
    y = np.array([(j + 0.5) * h for j in range(N)])
    K = np.zeros((len(xm), N))
    for j in range(N):
        K[:, j] = np.exp(-b * (xm - y[j]) ** 2) * h
    return y, K


def build_kernel(xm: np.ndarray, a: float, b: float, N: int = 64, y_max: float = 40.0):
    """
    Versao "direta": alem do kernel K, ja devolve o kerma de forma fechada
    f(y) = exp(-a*y) e o produto K @ f (usado como chute inicial do Hopfield
    na abordagem indireta, e como reconstrucao final na abordagem direta).

    Returns
    -------
    y, K, f, Kf
    """
    y, K = build_kernel_matrix(xm, b, N=N, y_max=y_max)
    f = np.exp(-a * y)
    return y, K, f, K @ f


def compute_pdd_convolucao(a: float, b: float, x: np.ndarray, d0: float,
                            N: int = 64, y_max: float = 40.0) -> np.ndarray:
    """
    Reconstroi a curva de PDD (normalizada a 100 em d0) pela via direta:
    kernel fechado exp(-b(x-y)^2) convoluido com kerma fechado exp(-a*y).
    """
    _, _, _, kf_x = build_kernel(x, a, b, N=N, y_max=y_max)
    _, _, _, kf_d0 = build_kernel(np.array([d0]), a, b, N=N, y_max=y_max)
    norm = kf_d0[0] if kf_d0[0] != 0 else 1.0
    return 100.0 * kf_x / norm
