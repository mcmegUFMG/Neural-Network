#!/usr/bin/env python3
"""
indirect/hopfield.py
=====================
Abordagem INDIRETA — nucleo numerico: dado (a, b) de um ajuste PDP (usado so
para montar o kernel e o chute inicial), roda uma rede de Hopfield que
relaxa o kerma f(y) de exp(-a*y) (forma fechada) para uma funcao numerica
F(y) que reconstroi melhor a curva de PDD medida via K @ F.

Hopfield energy:  E(V) = -1/2 V.T Tij V - Ii.T V
com Tij = -(K.T K)  e  Ii = K.T @ ym_n
com regularizacao de Tikhonov (lam):

    E(V) = 1/2 ||K V - ym||^2 + 1/2 lam ||V||^2
         => Tij = -(K.T K + lam I),   dU/dt = K.T (ym - K V) - lam V

O termo lam*I soma lam na diagonal de Tij: cada neuronio passa a se auto-inibir
(um "vazamento" que puxa V para zero), o que suprime os modos de alta
frequencia que o kernel gaussiano nao enxerga e que o ruido amplificaria.
lam=0 reproduz o comportamento anterior (sem regularizacao explicita; ficam so
a regularizacao implicita do t_end e o limite da tanh).

Tambem inclui a alternativa por minimos quadrados direto (OLS), mais rapida,
usada como fallback/comparacao.
"""

import numpy as np
from scipy.integrate import solve_ivp

from common.kernel import build_kernel, build_kernel_matrix


def run_hopfield(K, ym, f_init, ym_scale, t_end=100.0, margem=2.0, lam=1e-3):
    """
    Relaxa o kerma inicial f_init (chute fechado exp(-a*y)) para o kerma
    numerico F(y) = V que melhor reconstroi ym via K @ V, integrando a
    dinamica continua de Hopfield ate t_end.

    `margem` (>=1.0): fator de folga de amplitude acima de `ym_scale`. Sem
    essa folga, a parametrizacao V = tanh(U)*ym_scale trava |V| < ym_scale
    para SEMPRE, nao importa quanto tempo se integre -- e como ym_scale vem
    do mesmo ajuste fechado (a,b) usado so como chute inicial, isso impede a
    rede de encontrar um kerma com pico MAIOR do que o chute fechado ja
    assumia (fisicamente plausivel: kerma perto da superficie costuma
    exceder o pico da propria PDD medida, ja que a convolucao com o kernel
    tende a diluir amplitude). O sintoma tipico dessa saturacao e a PDD
    reconstruida ficando sistematicamente abaixo do esperado depois do
    buildup. `margem=2.0` da o dobro de amplitude disponivel; `margem=1.0`
    reproduz o comportamento antigo (sem folga).

    `lam` (>=0): intensidade da regularizacao de Tikhonov (ver docstring do
    modulo). Maior = kerma mais suave/estavel, menos fiel aos dados; menor =
    ajuste mais proximo, mais sujeito a oscilacao por ruido. `lam=0` desliga
    a regularizacao explicita. A escala de `lam` e relativa a K.T K, entao o
    valor ideal depende de b, do passo h e da normalizacao -- vale conferir
    com `oscillation_metrics` ou com uma curva L (residuo x norma de V).

    Returns
    -------
    V (kerma numerico, escala original), ok (bool, sucesso da integracao)
    """
    escala = margem * ym_scale
    ym_n = ym / ym_scale
    f_norm = f_init / escala
    Ii = K.T @ ym_n
    Tij = -(K.T @ K + lam * np.eye(K.shape[1]))   # Tikhonov: +lam na diagonal
    U0 = np.arctanh(np.clip(f_norm, -0.9999, 0.9999))

    def ode(t, U):
        # dU/dt = -dE/dV_n, com V_n = margem*tanh(U) (ver docstring do
        # modulo); o fator `margem` entra multiplicando o termo de energia
        # interna (Tij@tanh(U)) para manter a EDO consistente com essa
        # reparametrizacao -- sem ele, a folga de amplitude desalinharia o
        # ponto de equilibrio da dinamica.
        return margem * (Tij @ np.tanh(U)) + Ii

    sol = solve_ivp(ode, [0, t_end], U0, method='RK45',
                     rtol=1e-6, atol=1e-8, dense_output=False)
    V = margem * np.tanh(sol.y[:, -1]) * ym_scale
    return V, sol.success


def reconstruct_pdd_conv_ols(xm, ym, b, ym_scale, N=64, y_max=40.0):
    """
    Alternativa rapida ao Hopfield: extrai o kerma numerico F(y) por minimos
    quadrados ordinarios (OLS), resolvendo K @ F = ym diretamente.

    Mantida no modulo por compatibilidade/uso interno (ex.: comparacoes),
    mas nao e mais oferecida como opcao de metodo na UI (substituida pela
    extracao analitica via Fourier, ver `extrair_kerma_fourier`).

    Returns
    -------
    y, F (kerma numerico), pdd_reconstruida (K @ F)
    """
    y, K = build_kernel_matrix(xm, b, N=N, y_max=y_max)
    ym_n = ym / ym_scale
    F_n, _, _, _ = np.linalg.lstsq(K, ym_n, rcond=None)
    return y, F_n * ym_scale, (K @ F_n) * ym_scale


def extrair_kerma_fourier(xm, ym, b, ym_scale, N=64, y_max=40.0,
                          regularizar=True, lam=1e-3):
    """
    Extrai o kerma f(y) diretamente por DECONVOLUCAO NO DOMINIO DE FOURIER,
    em vez de relaxar uma rede de Hopfield ou resolver um problema de
    minimos quadrados.

    Como PDD(x) = integral K(x,y) f(y) dy com K(x,y) = exp(-b*(x-y)^2), o
    kernel e espacialmente invariante (depende so de x-y para um `b` fixo)
    -- ou seja, e uma convolucao classica PDD = g * f com g(t) = exp(-b*t^2).
    Isso permite recuperar f diretamente por divisao no dominio de Fourier:

        F(f) = F(PDD) / F(g)      (sem regularizacao)
        F(f) = F(PDD) * conj(F(g)) / (|F(g)|^2 + lam)     (Wiener/Tikhonov)

    Passos:
      1. a curva PDD medida (grade `xm`, possivelmente irregular) e
         reamostrada para a MESMA grade uniforme de y usada em
         `build_kernel_matrix` (0..y_max, N pontos), para que a operacao
         seja de fato uma convolucao discreta sobre uma grade comum;
      2. o kernel 1D g(t)=exp(-b t^2) e amostrado com zero-padding (2N
         pontos) para uma deconvolucao circular sem "wrap-around";
      3. divide-se no dominio de frequencia (com ou sem regularizacao) e
         volta-se ao dominio espacial via FFT inversa;
      4. o kerma resultante e forcado a ser nao-negativo (fisicamente o
         kerma nao pode ser negativo).

    Sem regularizacao, a divisao por |F(g)| pequeno em altas frequencias
    amplifica fortemente qualquer ruido da curva de PDD medida -- por isso
    a opcao `regularizar=True` (Wiener/Tikhonov, controlada por `lam`) e a
    recomendada para dados experimentais. `regularizar=False` fica
    disponivel para comparacao/depuracao (ex.: dados simulados sem ruido).

    LIMITACAO CONHECIDA: por usar uma discretizacao levemente diferente da
    used in Direct/Hopfield method (trapezoid with full weight on all
    pontos, em vez do peso reduzido nos extremos), o ponto mais proximo da
    superficie (y menor) tende a ser o mais sensivel a erro/artefato de
    borda desse metodo -- vale conferir esse ponto contra o Hopfield antes
    de confiar nele isoladamente.

    Parameters
    ----------
    lam : intensidade da regularizacao (quanto maior, mais suave/estavel e
        menos fiel aos detalhes de alta frequencia). So usada se
        `regularizar=True`.

    Returns
    -------
    y, F (kerma numerico, escala original), pdd_reconstruida (no grid
    original `xm`, calculada com a MESMA matriz de kernel do resto do
    pipeline, para comparacao consistente com os outros metodos), ok (bool)
    """
    h = y_max / N
    y = np.array([(j + 0.5) * h for j in range(N)])

    ym_n = ym / ym_scale
    pdd_grid = np.interp(y, xm, ym_n, left=0.0, right=ym_n[-1])

    # kernel 1D com zero-padding (2N pontos) para deconvolucao circular
    M = 2 * N
    t = (np.arange(M) - N) * h
    g = np.exp(-b * t ** 2)
    g = np.fft.ifftshift(g)  # centraliza o pico do kernel em t=0

    pdd_padded = np.zeros(M)
    pdd_padded[:N] = pdd_grid

    # a convolucao discreta (soma sobre a grade) equivale a (g*f) MULTIPLICADO
    # por h; por isso, antes de deconvoluir, removemos esse fator h de pdd
    # (senao a divisao em frequencia fica numa escala inconsistente com G)
    G = np.fft.rfft(g)
    P = np.fft.rfft(pdd_padded / h)

    if regularizar:
        F_hat = P * np.conj(G) / (np.abs(G) ** 2 + lam)
    else:
        G_seguro = np.where(np.abs(G) < 1e-10, 1e-10, G)
        F_hat = P / G_seguro

    f_full = np.fft.irfft(F_hat, n=M)
    F = np.clip(f_full[:N], 0.0, None) * ym_scale

    ok = bool(np.all(np.isfinite(F)))
    if not ok:
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    # reconstroi no grid original xm com a mesma matriz de kernel usada no
    # resto do pipeline, para uma comparacao de erro consistente
    _, K_xm = build_kernel_matrix(xm, b, N=N, y_max=y_max)
    pdd_rec = (K_xm @ (F / ym_scale)) * ym_scale

    return y, F, pdd_rec, ok


def oscillation_metrics(V):
    """Metricas de oscilacao usadas para escolher N (neuronios) e t_end."""
    d1 = np.diff(V)
    d2 = np.diff(d1)
    tv = np.sum(np.abs(d1))
    rms_d2 = np.sqrt(np.mean(d2 ** 2))
    sign_changes = int(np.sum(np.diff(np.sign(d1)) != 0))
    return tv, rms_d2, sign_changes


def extrair_kerma_hopfield(xm, ym, a, b, ym_scale, N=64, t_end=100.0, y_max=40.0, margem=2.0,
                           lam=1e-3):
    """
    Pipeline completo para uma unica curva: monta o kernel a partir de (a, b),
    usa exp(-a*y) como chute inicial e roda o Hopfield para obter F(y).

    `margem`: ver docstring de `run_hopfield` -- folga de amplitude acima de
    `ym_scale` para a rede nao ficar presa ao teto do chute fechado.
    `lam`: regularizacao de Tikhonov (ver `run_hopfield`).

    Returns
    -------
    y, F, pdd_reconstruida, ok
    """
    y, K, f_init, _ = build_kernel(xm, a, b, N=N, y_max=y_max)
    V, ok = run_hopfield(K, ym, f_init, ym_scale, t_end=t_end, margem=margem, lam=lam)
    pdd_rec = (K @ (V / ym_scale)) * ym_scale
    return y, V, pdd_rec, ok