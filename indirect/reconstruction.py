#!/usr/bin/env python3
"""
indirect/reconstruction.py
============================
INDIRECT approach — step 3: for a new, arbitrary (field, SSD),

  1. predicts kerma F(y) = A(field,SSD) * exp(-(y/tau(field,SSD))**beta(field,SSD))
     (exponencial estirada) usando os ajustes de indirect/kerma_variation.py;
  2. obtem b (e d0) para montar o kernel exp(-b*(x-y)^2). Sempre que a
     surface B(field,SSD) from the DIRECT method (direct/reconstruction.py,
     `carregar_ou_construir_interpoladores`) estiver disponivel, ela e
     REAPROVEITADA aqui (RectBivariateSpline, ajustada numa grade bem mais
     finer in SSDs than the sparse kerma extraction grid) -- and the same
     parametro fisico b do kernel de convolucao nos dois metodos, entao usar
     the Direct surface gives a much more reliable b than re-interpolating
     from scratch over the sparse indirect table. If the Direct grid
     nao estiver disponivel, cai para uma interpolacao (griddata) simples
     sobre a propria tabela de kermas, como fallback.
  3. reconstroi PDD(x) = convolucao( kernel(b), F(y) ), normalizada a 100 em d0.
"""

import numpy as np
from scipy.interpolate import griddata

from common.kernel import build_kernel_matrix


def _eval_rgi(interp, campo, ssd):
    """Avalia um RegularGridInterpolator (usado para interp_d0) num unico
    ponto (campo, ssd). Duplicado aqui (em vez de importar de
    direct.reconstruction) para nao depender de uma funcao privada de outro
    modulo."""
    return float(interp([[campo, ssd]])[0])


def _interp_grade(df_coef, coluna, campo_novo, ssd_novo, param_type='campo'):
    """Fallback grosseiro: interpola `coluna` via griddata sobre a propria
    sparse kerma table (used only if the Direct method grid is not
    estiver disponivel)."""
    pontos = df_coef[[param_type, 'ssd']].values.astype(float)
    valores = df_coef[coluna].values.astype(float)
    val = griddata(pontos, valores, [(campo_novo, ssd_novo)], method='linear')
    if np.isnan(val[0]):
        val = griddata(pontos, valores, [(campo_novo, ssd_novo)], method='nearest')
    return float(val[0])


def _obter_b_d0(df_coef, campo, ssd, param_type, grade_direto=None):
    """
    Decide de onde vem `b` e `d0` para a reconstrucao:
      - se `grade_direto` (grade de superficies do metodo Direto) foi
        passado, usa `interp_b`/`interp_d0` de la (preferido: mais suave e
        mais confiavel, especialmente fora da grade esparsa do kerma);
      - senao, cai no fallback griddata sobre a tabela de kermas.

    Returns
    -------
    b, d0, origem (str, para deixar claro na UI/relatorio de onde veio)
    """
    if grade_direto is not None:
        b = float(grade_direto['interp_b'].ev(campo, ssd))
        d0 = _eval_rgi(grade_direto['interp_d0'], campo, ssd)
        if b <= 0:
            raise ValueError(
                f"b interpolated from the Direct method surface is invalid ({b:.4f}). "
                "Field/SSD likely outside the grid built in 'Direto > "
                "Build A and B surfaces'."
            )
        return b, d0, 'superficie B do metodo Direto'

    b = _interp_grade(df_coef, 'b_pdp', campo, ssd, param_type=param_type)
    d0 = _interp_grade(df_coef, 'd0', campo, ssd, param_type=param_type)
    return b, d0, 'griddata sobre a tabela esparsa de kermas (fallback)'


def prever_kerma(fit_A: dict, fit_tau: dict, fit_beta: dict,
                  campo: float, ssd: float, y: np.ndarray):
    """
    Predicts kerma F(y) = A * exp( -(y/tau)^beta ) for a (field, SSD)
    novo, a partir dos 3 ajustes generalizados (A, tau, beta).

    Os coeficientes previstos sao limitados a uma faixa fisica
    (A >= 0, tau > 0, beta > 0) para que extrapolacoes do modelo de
    superficie nunca gerem kerma negativo nem divisao por zero / potencias
    invalidas.
    """
    A = max(float(fit_A['prever'](campo, ssd)), 0.0)
    tau = max(float(fit_tau['prever'](campo, ssd)), 1e-6)
    beta = float(np.clip(fit_beta['prever'](campo, ssd), 0.05, 10.0))
    F_pred = A * np.exp(-((np.asarray(y, dtype=float) / tau) ** beta))
    return F_pred, A, tau, beta


def reconstruir_pdd_indireta(df_coef, fit_A: dict, fit_tau: dict, fit_beta: dict,
                              campo: float, ssd: float,
                              n_pts: int = 300, x_max: float = 30.0,
                              N: int = 64, y_max: float = 40.0,
                              param_type: str = None, grade_direto: dict = None,
                              modelo: str = 'stretched', modelo_v2=None):
    """
    Reconstroi a curva de PDD completa para (campo, ssd) pela via indireta:
    kerma exponencial estirada previsto (A, tau, beta) convoluido com o
    kernel (b coming, preferably, from the already fitted B(field,SSD) surface
    in the Direct method).

    `campo` deve ser o valor NUMERICO do parametro de campo (o mesmo usado
    em fit_A/fit_tau/fit_beta, ex.: 5, 10, 20 para "Field size").
    `param_type`, se None, e inferido de df_coef (coluna que nao e
    campo/ssd/d0/a_pdp/b_pdp/A_kerma/tau_kerma/beta_kerma/r2_kerma_fit/
    y_fim_valido).

    `grade_direto`: dict retornado por
    `direct.reconstruction.carregar_ou_construir_interpoladores` (ou
    equivalente, com chaves `interp_b`/`interp_d0`). Se fornecido, b e d0
    vem de la (recomendado). Se None, usa o fallback griddata sobre a
    tabela esparsa de kermas -- mais fraco, principalmente fora da grade
    onde os kermas foram extraidos.

    `modelo`: 'stretched' (legacy, default) or 'v2' (F5 family fitted in PDD space, see
    indirect/kerma_model_v2.py; needs `modelo_v2`, a ModeloKermaV2). The kernel, b and d0
    are the same in both; `N`/`y_max` only apply to the legacy path (V2 uses its own graded grid).

    Returns
    -------
    x, pdd, info dict (com b, d0, A/tau/beta previstos, e `b_origem`
    indicando de onde veio o b usado)
    """
    if modelo == 'v2':      # new F(y) model (same kernel, b and d0 source); legacy path below is untouched
        from .kerma_model_v2 import reconstruir_pdd_indireta_v2
        if modelo_v2 is None:
            raise ValueError("modelo='v2' requires modelo_v2 (see kerma_model_v2.ModeloKermaV2)")
        return reconstruir_pdd_indireta_v2(modelo_v2, campo, ssd, n_pts=n_pts, x_max=x_max,
                                           df_coef=df_coef, param_type=param_type,
                                           grade_direto=grade_direto)
    if param_type is None:
        cols_fixas = {'campo', 'ssd', 'd0', 'a_pdp', 'b_pdp',
                      'A_kerma', 'tau_kerma', 'beta_kerma',
                      'r2_kerma_fit', 'y_fim_valido'}
        param_type = next(c for c in df_coef.columns if c not in cols_fixas)

    b_interp, d0_interp, b_origem = _obter_b_d0(df_coef, campo, ssd, param_type, grade_direto)

    x = np.linspace(0.0, x_max, n_pts)
    y, K = build_kernel_matrix(x, b_interp, N=N, y_max=y_max)
    F_pred, A, tau, beta = prever_kerma(fit_A, fit_tau, fit_beta, campo, ssd, y)

    pdd_raw = K @ F_pred

    _, K_d0 = build_kernel_matrix(np.array([d0_interp]), b_interp, N=N, y_max=y_max)
    norm = float((K_d0 @ F_pred)[0])
    if norm == 0:
        raise ValueError("Null normalization when reconstructing indirect PDD; check the fits.")
    pdd = 100.0 * pdd_raw / norm

    return x, pdd, {
        'campo': campo, 'ssd': ssd, 'b': b_interp, 'd0': d0_interp,
        'A_previsto': A, 'tau_previsto': tau, 'beta_previsto': beta,
        'y_grid': y, 'F_previsto': F_pred,
        'b_origem': b_origem,
    }


# ------------------------------------------------------------------
#  RECONSTRUCAO INDIRETA SIMPLES
# ------------------------------------------------------------------
#
# More limited mode: instead of generalizing A(field,SSD), tau(field,SSD) and
# beta(field,SSD) via a continuous model, directly uses the numeric kerma
# F(y) que ja foi extraido pela rede de Hopfield/OLS (etapa "Extrair kermas").
# Only works for the (field, SSD) combinations already in the calculated grid
# — nao generaliza para pontos novos. Serve como referencia/sanidade para
# comparar com a reconstrucao avancada (que usa o modelo generalizado).

def encontrar_chave_kerma(kermas: dict, campo, ssd, tol: float = 1e-6):
    """
    Procura em `kermas` a chave (campo, ssd) correspondente.

    Tenta primeiro igualdade exata (funciona para qualquer tipo — string,
    int, float — e é o caso comum quando `campo`/`ssd` vem direto de
    `kermas.keys()`, como no seletor da reconstrução simples). Se não achar
    e os valores forem numericos, tenta de novo com tolerancia `tol`.
    """
    for (c, s) in kermas.keys():
        if c == campo and s == ssd:
            return (c, s)

    try:
        campo_f = float(campo)
        ssd_f = float(ssd)
    except (TypeError, ValueError):
        return None

    for (c, s) in kermas.keys():
        try:
            if abs(float(c) - campo_f) < tol and abs(float(s) - ssd_f) < tol:
                return (c, s)
        except (TypeError, ValueError):
            continue
    return None


def reconstruir_pdd_indireta_simples(df_coef, kermas: dict, campo: float, ssd: float,
                                      n_pts: int = 300, x_max: float = 30.0,
                                      param_type: str = None):
    """
    Reconstroi PDD(x) usando diretamente o F(y) numerico ja extraido para o
    par (campo, ssd) EXATO (sem passar pelo modelo generalizado de
    A/tau/beta). b e d0 sao os valores ja calculados (nao interpolados) dessa
    mesma linha da tabela de coeficientes.

    Raises ValueError if (field, ssd) is not among the already
    extraidos em `kermas` — nesse caso, use a reconstrucao avancada
    (`reconstruir_pdd_indireta`), que generaliza para qualquer ponto.
    """
    if param_type is None:
        cols_fixas = {'campo', 'ssd', 'd0', 'a_pdp', 'b_pdp',
                      'A_kerma', 'tau_kerma', 'beta_kerma',
                      'r2_kerma_fit', 'y_fim_valido'}
        param_type = next(c for c in df_coef.columns if c not in cols_fixas)

    chave = encontrar_chave_kerma(kermas, campo, ssd)
    if chave is None:
        raise ValueError(
            f"Nao ha kerma numerico extraido para field={campo}, ssd={ssd}. "
            "O modo SIMPLES so reconstroi para combinacoes (campo, SSD) ja "
            "calculated in the 'Extract kermas' step. For any (field, SSD), "
            "use a reconstrucao avancada (modelo generalizado)."
        )

    linha = df_coef[(df_coef['campo'] == chave[0]) & (df_coef['ssd'] == chave[1])]
    if linha.empty:
        raise ValueError("(field, ssd) pair found in kermas but not in coefficients table.")

    b = float(linha['b_pdp'].values[0])
    d0 = float(linha['d0'].values[0])

    k = kermas[chave]
    y_kerma = np.asarray(k['y'], dtype=float)
    F_kerma = np.asarray(k['F'], dtype=float)
    y_max_kerma = float(y_kerma.max())

    x = np.linspace(0.0, x_max, n_pts)
    y_kernel, K = build_kernel_matrix(x, b, N=len(y_kerma), y_max=y_max_kerma)

    # a grade de y do kernel deve reproduzir a grade usada na extracao do
    # kerma (mesmo N e y_max); se por algum motivo nao bater exatamente,
    # interpola F para a grade do kernel em vez de falhar.
    if not np.allclose(y_kernel, y_kerma, rtol=1e-3, atol=1e-6):
        F_kerma = np.interp(y_kernel, y_kerma, F_kerma)

    pdd_raw = K @ F_kerma

    _, K_d0 = build_kernel_matrix(np.array([d0]), b, N=len(y_kerma), y_max=y_max_kerma)
    norm = float((K_d0 @ F_kerma)[0])
    if norm == 0:
        raise ValueError("Null normalization when reconstructing simple indirect PDD; check the extracted kerma.")
    pdd = 100.0 * pdd_raw / norm

    return x, pdd, {
        'campo': chave[0], 'ssd': chave[1], 'b': b, 'd0': d0,
        'y_grid': y_kernel, 'F_usado': F_kerma, 'modo': 'simples',
    }
