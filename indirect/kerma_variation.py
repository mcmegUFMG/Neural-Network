#!/usr/bin/env python3
"""
[NOTE 2026-11 / plan S08: the trapezoid-weight statement below is OUTDATED. common/kernel.py
now uses the midpoint rule, so the first cells are PHYSICAL (surface dose / electron
contamination) and carry the field-size dependence of the PDD. The legacy 'stretched' model
discards them (that is why it cannot reproduce the measured surface dose); the 'v2' model
(kerma_model_v2.py) keeps them. The text below describes the legacy model only.]


indirect/kerma_variation.py
=============================
Abordagem INDIRETA — passo 2: analisa como os kermas numericos F(y; campo,
SSD) (obtained in indirect/kerma_batch.py) vary.

MODELO DO KERMA — EXPONENCIAL ESTIRADA (stretched exponential)
----------------------------------------------------------------
O kerma numerico F(y) extraido pela rede (Hopfield/OLS) tem, fisicamente,
um decaimento suave tipo bulk ao longo de toda a profundidade. Perto da
superficie (primeira celula da grade) aparece um pico numerico, e no fim do
dominio aparece um corte abrupto -- ambos ARTEFATOS de borda (peso de
trapezio no extremo inicial e dominio de treino da rede no final), nao
fisica por curva.

O modelo antigo (biexponential + gaussian bubble + sigmoid cutoff, 5
parametros por curva, 3 constantes globais calibradas) tentava reproduzir
literalmente esses artefatos. Isso (a) gastava parametros sem ganho real de
reconstrucao, (b) fazia cada coeficiente variar de forma ruidosa com
(field, SSD) and (c) was worse: in ADVANCED reconstruction the RMSE error in
conjunto 10x10 caia de ~8.5% para ~2.2% ao trocar pelo modelo abaixo.

O modelo atual resume cada F(y) por uma unica exponencial estirada:

    F(y) = A * exp( -(y / tau) ** beta )

com apenas 3 parametros por curva:
  - A    : amplitude (valor extrapolado em y=0);
  - tau  : escala de profundidade (em cm);
  - beta : indice de estiramento (beta=1 => exponencial pura; beta>1 =>
           decaimento mais rapido que o exponencial; beta<1 => mais lento).

A exponencial estirada e monotonica e suave, capta a curvatura do
decaimento em escala log (que o biexponencial puro nao capta sem degenerar)
e nao precisa de termos artificiais. As regioes de artefato (primeira
celula e corte final) sao simplesmente EXCLUIDAS da janela de ajuste -- ver
`_janela_ajuste_kerma`.

GENERALIZATION MODEL (field, SSD) -> (A, tau, beta)
------------------------------------------------------
Os 3 coeficientes por curva sao descritos pela mesma familia de funcao
saturante bivariada:

    coefficient(field, SSD) = a - b*exp(-field/c) - d*exp(-SSD/e)

Os sinais de b e d nao sao restritos: dependendo do coeficiente a funcao
aparece como decaimento saturante (b,d>0) ou crescimento saturante (b<0).

IMPORTANTE -- bounds de c e e: c e um comprimento de decaimento em
unidades de "campo" (seus campos variam 3-40), e e e um comprimento de
decay in "SSD" units (your SSDs vary 100-354). Without bounds,
c e/ou e podem fugir para 1e5-1e6 e tornar a exponencial praticamente
linear, quebrando a identificabilidade de (a, d, e). Por isso
`_ajustar_decaimento_vs_campo_ssd` usa bounds compativeis com o dominio
real field/SSD (see `SURFACE_BOUNDS_C`/`SURFACE_BOUNDS_E` below).
Widen them only if the real field/SSD domain is larger than described.

FILTRAGEM DE OUTLIERS ANTES DO AJUSTE GENERALIZADO
----------------------------------------------------
Um kerma individual mal ajustado (Hopfield/OLS que nao convergiu direito,
ou o modelo de curva degenerado num ponto isolado) tem `r2_kerma_fit` bem
abaixo do resto da tabela e, se entrar no ajuste generalizado, distorce a
superficie inteira via minimos quadrados -- mesmo sendo 1 unico ponto
entre varias dezenas. Por isso, `ajustar_modelo_kerma` descarta, antes de
chamar o curve_fit, os pontos com `r2_kerma_fit` abaixo de um limiar
(default 0.85; ver `_filtrar_pontos_validos`). Se a filtragem deixar menos
de 5 pontos (base pequena ou muito ruidosa), ela e ignorada e a tabela
original e usada sem filtro, para o pipeline nao quebrar.

Se novos dados mostrarem outro comportamento, revise `_modelo_kerma` e a
janela de ajuste (`_janela_ajuste_kerma`).
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ------------------------------------------------------------------
#  Formulas (para exibicao na interface)
# ------------------------------------------------------------------

FORMULA_KERMA = "F(y) = A · exp( −(y/τ)^β )"
FORMULA_CONSTANTES = (
    "3-parameter model per curve (A, τ, β). The edges (1st cell and "
    "final cutoff) are excluded from the fit."
)
FORMULA_A = "A(campo, SSD) = a − b·exp(−campo/c) − d·exp(−SSD/e)   (amplitude)"
FORMULA_TAU = "τ(campo, SSD) = a − b·exp(−campo/c) − d·exp(−SSD/e)   (escala de profundidade)"
FORMULA_BETA = "β(campo, SSD) = a − b·exp(−campo/c) − d·exp(−SSD/e)   (índice de estiramento)"


# ------------------------------------------------------------------
#  Numeric kerma F(y) fit: stretched exponential
# ------------------------------------------------------------------

def _modelo_kerma(y, A, tau, beta):
    """F(y) = A * exp( -(y/tau)^beta )."""
    return A * np.exp(-((np.asarray(y, dtype=float) / tau) ** beta))


# Alias de compatibilidade (nome descritivo).
curve_model = _modelo_kerma


# Bounds dos 3 parametros por curva: A, tau, beta.
#   - A    : amplitude positiva (o kerma e positivo);
#   - tau  : escala de profundidade em cm, no dominio fisico;
#   - beta : estiramento, mantido num intervalo amplo mas finito (beta=1 e
#            a exponencial pura; fugir muito disso indica ajuste degenerado).
CURVE_BOUNDS = ([0.0, 0.1, 0.05], [1e4, 1e4, 5.0])

# Limiar de salto relativo para reconhecer o corte abrupto de borda no fim
# do dominio. Um salto relativo (F[j]-F[j+1])/F[j] acima disso e tratado
# como artefato de fim (pontos a partir dele sao descartados). Se nenhum
# salto atingir esse limiar, todo o dominio apos a primeira celula e usado.
LIMIAR_SALTO_BORDA = 0.35

# Numero minimo de pontos na janela de ajuste (garante que o fit seja
# identificavel mesmo numa grade curta ou com corte de borda agressivo).
MIN_PONTOS_AJUSTE = 8


def _janela_ajuste_kerma(y: np.ndarray, F: np.ndarray):
    """
    Determina a janela [i_ini, i_fim] (inclusiva) usada no ajuste.

    Regras:
      - descarta a PRIMEIRA celula (pico numerico de borda em y~0);
      - detecta o CORTE abrupto do fim do dominio como o maior salto
        relativo negativo em F; se esse salto superar `LIMIAR_SALTO_BORDA`,
        mantem os pontos ate ANTES do ponto que cai (exclui o degrau e o
        que vem depois); se nao houver salto significativo, usa todo o
        dominio restante;
      - garante pelo menos `MIN_PONTOS_AJUSTE` pontos na janela.

    Returns
    -------
    (i_ini, i_fim) com i_fim >= i_ini.
    """
    y = np.asarray(y, dtype=float)
    F = np.asarray(F, dtype=float)
    n = F.size
    if n <= 3:
        return 0, max(n - 1, 0)

    i_ini = 1
    # saltos relativos entre pontos consecutivos, do inicio do dominio
    base = np.maximum(np.abs(F[:-1]), 1e-12)
    rel = -(np.diff(F)) / base          # positivo quando F cai
    j = int(np.argmax(rel))

    if rel[j] >= LIMIAR_SALTO_BORDA and j >= i_ini:
        # mantem [i_ini .. j-1]: exclui o ponto que cai (j+1 e posteriores
        # ja estao fora) e o ponto imediatamente antes do salto (j).
        i_fim = j - 1
    else:
        i_fim = n - 1

    if i_fim < i_ini + MIN_PONTOS_AJUSTE - 1:
        i_fim = min(n - 1, i_ini + MIN_PONTOS_AJUSTE - 1)
    i_fim = max(i_fim, i_ini + 1)
    return i_ini, i_fim


def ajustar_curva_kerma(y: np.ndarray, F: np.ndarray) -> dict:
    """
    Ajusta F(y) pela exponencial estirada `_modelo_kerma`, por minimos
    quadrados nao lineares, direto sobre F(y) (nao em log), na janela
    definida por `_janela_ajuste_kerma`.

    Returns
    -------
    dict com A, tau, beta, r2 (sobre a janela de ajuste) e y_fim_valido
    (profundidade final considerada confiavel).
    """
    y = np.asarray(y, dtype=float)
    F = np.asarray(F, dtype=float)

    i_ini, i_fim = _janela_ajuste_kerma(y, F)
    y_fit, F_fit = y[i_ini:i_fim + 1], F[i_ini:i_fim + 1]
    y_fim_valido = float(y_fit[-1]) if y_fit.size else float(y.max())

    if y_fit.size < 4:
        return dict(A=np.nan, tau=np.nan, beta=np.nan, r2=np.nan,
                    y_fim_valido=y_fim_valido)

    # --- chute inicial a partir de um ajuste log-linear (beta=1) ---
    positivos = F_fit > 0
    if positivos.sum() >= 2:
        Amat = np.vstack([y_fit[positivos], np.ones(int(positivos.sum()))]).T
        (slope, intercept), *_ = np.linalg.lstsq(
            Amat, np.log(F_fit[positivos]), rcond=None)
        tau0 = -1.0 / slope if slope < -1e-6 else 20.0
        A0 = float(np.exp(intercept))
    else:
        tau0, A0 = 20.0, float(np.nanmax(F_fit)) if F_fit.size else 1.0
    p0 = [float(np.clip(A0, CURVE_BOUNDS[0][0], CURVE_BOUNDS[1][0])),
          float(np.clip(tau0, CURVE_BOUNDS[0][1], CURVE_BOUNDS[1][1])),
          1.0]

    try:
        popt, _ = curve_fit(_modelo_kerma, y_fit, F_fit, p0=p0,
                            bounds=CURVE_BOUNDS, maxfev=50000)
        A, tau, beta = (float(v) for v in popt)
    except (RuntimeError, ValueError):
        # fallback: exponencial pura (beta=1) com os valores do chute
        A, tau, beta = float(p0[0]), float(p0[1]), 1.0

    pred = _modelo_kerma(y_fit, A, tau, beta)
    ss_res = float(np.sum((F_fit - pred) ** 2))
    ss_tot = float(np.sum((F_fit - F_fit.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return dict(A=A, tau=tau, beta=beta, r2=float(r2),
                y_fim_valido=y_fim_valido)


def tabelar_coeficientes(df_tidy: pd.DataFrame, kermas: dict,
                          param_type: str, modelo: str = 'stretched') -> pd.DataFrame:
    """
    modelo='v2' -> returns the per-curve V2 table (F5 parameters fitted in PDD space, see
    indirect/kerma_model_v2.tabelar_v2) instead of the stretched-exponential table.
    For each row of df_tidy (one per field/SSD), fits the kerma by
    exponencial estirada e devolve uma tabela consolidada com as 3
    componentes (A_kerma, tau_kerma, beta_kerma).
    """
    if modelo == 'v2':
        from .kerma_model_v2 import tabelar_v2
        return tabelar_v2(df_tidy, kermas, param_type)
    rows = []
    for _, row in df_tidy.iterrows():
        chave = (row['campo'], row['ssd'])
        kerma = kermas[chave]
        ajuste = ajustar_curva_kerma(kerma['y'], kerma['F'])
        rows.append({
            'campo': row['campo'], param_type: row[param_type], 'ssd': row['ssd'],
            'a_pdp': row['a'], 'b_pdp': row['b'], 'd0': row['d0'],
            'A_kerma': ajuste['A'], 'tau_kerma': ajuste['tau'],
            'beta_kerma': ajuste['beta'],
            'r2_kerma_fit': ajuste['r2'], 'y_fim_valido': ajuste['y_fim_valido'],
        })
    return pd.DataFrame(rows).sort_values([param_type, 'ssd']).reset_index(drop=True)


# ------------------------------------------------------------------
#  MODELOS DE GENERALIZACAO: os 3 coeficientes (A, tau, beta) vs
#  (field, SSD), all with the same bivariate saturating family.
# ------------------------------------------------------------------

def _modelo_decaimento(X, a, b, c, d, e):
    """
    Familia unica de funcao saturante bivariada, usada para os 3
    coeficientes (A, tau, beta). Os sinais de b e d nao sao restritos --
    o curve_fit ajusta livremente, e dependendo dos dados a funcao aparece
    como decaimento saturante (b,d>0) ou crescimento saturante (b<0).
    """
    campo, ssd = X
    return a - b * np.exp(-campo / c) - d * np.exp(-ssd / e)


# Bounds de c/e da superficie: c e um comprimento de decaimento em
# unidades de "campo" (dominio real 3-40), e e um comprimento de
# decay in "SSD" units (real domain 100-354). Without ceiling,
# c/e podem fugir para 1e5+ e tornar (a, d, e) nao identificaveis. Alargue
# only if the real field/SSD domain is larger than described above.
SURFACE_BOUNDS_C = (0.05, 500.0)
SURFACE_BOUNDS_E = (1.0, 2000.0)


def _empacotar_fit(coluna, modelo, formula, popt, func, campo, ssd, z, aviso):
    a, b, c, d, e = [float(v) for v in popt]

    def prever(campo_novo, ssd_novo):
        campo_novo = np.atleast_1d(np.asarray(campo_novo, dtype=float))
        ssd_novo = np.atleast_1d(np.asarray(ssd_novo, dtype=float))
        pred = func((campo_novo, ssd_novo), *popt)
        return pred if pred.size > 1 else float(pred[0])

    if campo.size:
        pred_treino = func((campo, ssd), *popt)
        resid = pred_treino - z
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((z - z.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        rmse, r2 = np.nan, np.nan

    formula_numerica = (f"{coluna}: a={a:.5g}, b={b:.5g}, c={c:.5g}, "
                         f"d={d:.5g}, e={e:.5g}")

    return {
        'coluna': coluna, 'modelo': modelo, 'formula': formula,
        'formula_numerica': formula_numerica,
        'params': {'a': a, 'b': b, 'c': c, 'd': d, 'e': e},
        'coef': popt, 'prever': prever,
        'rmse_treino': rmse, 'r2_treino': r2, 'aviso': aviso,
    }


def _ajustar_decaimento_vs_campo_ssd(df_coef, coluna, param_type, formula, nome_curto):
    campo = df_coef[param_type].values.astype(float)
    ssd = df_coef['ssd'].values.astype(float)
    z = df_coef[coluna].values.astype(float)
    mask = np.isfinite(z) & np.isfinite(campo) & np.isfinite(ssd)
    campo, ssd, z = campo[mask], ssd[mask], z[mask]

    amplitude = max(z.max() - z.min(), 1e-6) if z.size else 1.0
    c0 = float(np.clip(campo.mean(), *SURFACE_BOUNDS_C)) if campo.size else float(np.mean(SURFACE_BOUNDS_C))
    e0 = float(np.clip(ssd.mean(), *SURFACE_BOUNDS_E)) if ssd.size else float(np.mean(SURFACE_BOUNDS_E))
    p0 = [
        z.max() if z.size else 1.0,
        amplitude * 0.7,
        max(c0, SURFACE_BOUNDS_C[0]),
        amplitude * 0.3,
        max(e0, SURFACE_BOUNDS_E[0]),
    ]
    # bounds: c and e restricted to real field/SSD domain (see
    # SURFACE_BOUNDS_C/E acima) -- evita c/e fugirem para valores absurdos
    # e tornarem (a, d, e) nao identificaveis.
    bounds = (
        [-np.inf, -np.inf, SURFACE_BOUNDS_C[0], -np.inf, SURFACE_BOUNDS_E[0]],
        [np.inf, np.inf, SURFACE_BOUNDS_C[1], np.inf, SURFACE_BOUNDS_E[1]],
    )

    aviso = (f"Bivariate saturating exponential model ({nome_curto}), fitted "
             "by non-linear least squares over field and SSD, with c/e "
             "restricted to real field/SSD domain.")
    try:
        popt, _ = curve_fit(_modelo_decaimento, (campo, ssd), z, p0=p0,
                             bounds=bounds, maxfev=30000)
    except (RuntimeError, ValueError) as exc:
        popt = np.array([float(z.mean()) if z.size else 0.0, 0.0,
                          SURFACE_BOUNDS_C[0], 0.0, SURFACE_BOUNDS_E[0]])
        aviso = (f"AVISO: ajuste nao-linear de {nome_curto} nao convergiu ({exc}); "
                 "usando valor constante (media) como fallback.")

    return _empacotar_fit(coluna, 'exponencial_saturante_bivariado', formula,
                           popt, _modelo_decaimento, campo, ssd, z, aviso)


def _filtrar_pontos_validos(df_coef: pd.DataFrame, r2_min: float = 0.85) -> pd.DataFrame:
    """
    Remove, antes de ajustar os modelos generalizados, os pontos (campo,
    SSD) whose individual curve fit (`r2_kerma_fit`, calculated in
    `ajustar_curva_kerma`) ficou muito abaixo do resto da tabela --
    tipicamente um kerma onde o Hopfield/OLS nao convergiu direito, ou o
    ajuste ficou degenerado. Um unico ponto assim distorce a superficie
    inteira via minimos quadrados, mesmo entre varias dezenas de pontos
    bons.

    Se a filtragem deixar menos de 5 pontos (base pequena ou muito
    ruidosa) ou a coluna `r2_kerma_fit` nao existir, devolve `df_coef`
    original sem filtrar, para o pipeline nao quebrar.
    """
    if 'r2_kerma_fit' not in df_coef.columns:
        return df_coef

    valido = df_coef['r2_kerma_fit'] >= r2_min
    if valido.sum() < 5:
        return df_coef

    n_descartados = int((~valido).sum())
    if n_descartados:
        print(f"  [ajuste generalizado] descartando {n_descartados} ponto(s) "
              f"com r2_kerma_fit < {r2_min} antes de ajustar os modelos "
              "(provavel kerma mal ajustado/outlier).")
    return df_coef[valido]


def ajustar_modelo_kerma(df_coef: pd.DataFrame, param_type: str, r2_min: float = 0.85,
                         modelo: str = 'stretched'):
    """
    modelo='v2' -> `df_coef` must be the table returned by tabelar_coeficientes(..., modelo='v2');
    returns a single kerma_model_v2.ModeloKermaV2 (tensor generalisation) instead of the
    (fit_A, fit_tau, fit_beta) tuple of the legacy model.

    Ajusta os 3 modelos de generalizacao de uma vez, todos com a mesma
    familia de funcao saturante bivariada:

        A_kerma, tau_kerma, beta_kerma (field, SSD)
            = a - b*exp(-campo/c) - d*exp(-SSD/e)

    with c/e restricted to real field/SSD domain (see
    `SURFACE_BOUNDS_C`/`SURFACE_BOUNDS_E`). Antes de ajustar, descarta
    pontos com `r2_kerma_fit` muito baixo (ver `_filtrar_pontos_validos`)
    para nao deixar um kerma mal ajustado distorcer a superficie inteira.

    Returns
    -------
    fit_A, fit_tau, fit_beta (dicts com 'coluna', 'prever',
    'rmse_treino', 'r2_treino', 'aviso', ...)
    """
    if modelo == 'v2':
        from .kerma_model_v2 import ModeloKermaV2
        return ModeloKermaV2(df_coef, param_type)
    df_valido = _filtrar_pontos_validos(df_coef, r2_min=r2_min)

    fit_A = _ajustar_decaimento_vs_campo_ssd(
        df_valido, 'A_kerma', param_type, FORMULA_A, 'A (amplitude)')
    fit_tau = _ajustar_decaimento_vs_campo_ssd(
        df_valido, 'tau_kerma', param_type, FORMULA_TAU, 'tau (escala de profundidade)')
    fit_beta = _ajustar_decaimento_vs_campo_ssd(
        df_valido, 'beta_kerma', param_type, FORMULA_BETA, 'beta (estiramento)')
    return fit_A, fit_tau, fit_beta


# Aliases de compatibilidade com nomes antigos, caso algo mais no projeto
# ainda importe.
ajustar_modelo_biexp = ajustar_modelo_kerma
ajustar_modelo_alpha_C = ajustar_modelo_kerma
ajustar_placeholders = ajustar_modelo_kerma
