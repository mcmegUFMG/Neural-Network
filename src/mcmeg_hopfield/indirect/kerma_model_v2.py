#!/usr/bin/env python3
"""
indirect/kerma_model_v2.py
==========================
"V2" kerma model for the indirect approach (ported from experiments/pdd_space.py,
experiments/models/v2_pddfamily.py and v2_ssd.py; plan.md steps S03-S06).

What stays the SAME as the legacy ('stretched') path
----------------------------------------------------
* The convolution kernel is unchanged:  K(x, y) = exp(-b (x - y)^2)  (midpoint rule,
  every cell weighted by its width), and  PDD(x) = 100 * (K @ F)(x) / (K @ F)(d0).
* b and d0 come from exactly the same place as in the legacy reconstruction
  (`reconstruction._obter_b_d0`: Direct-method B / d0 surfaces `grade_direto`,
  fallback = table interpolation).  The Direct B surface equals the per-curve
  b of the closed-form (a, b) fit at every fitted node (checked: 2e-15), so training
  and prediction use the same kernel.
* Mayneord pseudo-curves, field/SSD grid and the Hopfield kerma extraction are untouched.

What changes
------------
1. F(y) family ("F5"): bulk stretched exponential  exp(-(y/tau)^beta)  PLUS three free
   amplitudes a1, a2, a3 on the surface zones [0,.3), [.3,.8), [.8,1.5) cm.  The amplitude A
   is dropped (it cancels in the d0 normalisation).  The legacy fit threw the surface cells
   away; that is where the field-size dependence of the measured PDD lives.
2. The parameters are fitted IN PDD SPACE (K @ F normalised at d0 vs the target curve), not
   by fitting F(y) in F-space -> independent of Hopfield convergence.
3. The y grid does NOT need 64 neurons: a geometrically graded grid (cell width 0.1 cm at the
   surface growing x1.08 up to 0.5 cm, y_max = 40 cm -> 91 cells) is used by default;
   any grid can be passed (`edges=`).  The kernel formula is identical, only the quadrature
   nodes/weights differ.
4. Generalisation over (field, SSD): "tensor" form = PCHIP in ln(field) at every SSD node,
   then PCHIP over SSD (linear extrapolation outside).  The SSD dependence is not separable
   (additive / multiplicative forms were clearly worse in S06).

Public API
----------
    tabelar_v2(df_tidy, kermas, param_type)        -> DataFrame (per-curve F5 parameters)
    ModeloKermaV2(df_v2, param_type)                -> generalisation object (.params, .b_tabela)
    reconstruir_pdd_indireta_v2(modelo, campo, ssd, ...)  -> x, pdd, info   (same signature
        family as reconstruction.reconstruir_pdd_indireta)
"""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

# ------------------------------------------------------------------ constants
GRADE_V2 = dict(ymax=40.0, h1=0.1, h2=0.5, r=1.08)      # 91 cells
ZONAS = (0.0, 0.3, 0.8, 1.5)                            # surface zones [cm]
PNAMES = ('tau', 'beta', 'a1', 'a2', 'a3')
LO = np.array([1.0, 0.3, 0.0, 0.0, 0.0])
HI = np.array([200.0, 4.0, 8.0, 8.0, 8.0])
STARTS = [[20, 1.1, 1.0, 1.0, 1.0], [20, 1.1, 1.5, 1.2, 1.0], [20, 1.1, 0.6, 0.8, 1.0]]

FORMULA_KERMA_V2 = "F(y) = exp(−(y/τ)^β)  ,  y ≥ 1.5 cm;   F(y) = a1, a2, a3  on [0,.3), [.3,.8), [.8,1.5) cm"
FORMULA_GENERALIZACAO_V2 = "p(campo, SSD): PCHIP em ln(campo) a cada SSD, depois PCHIP em SSD (p = τ, β, a1, a2, a3, b)"


# ------------------------------------------------------------------ grid / kernel
def grade_graduada(ymax=40.0, h1=0.1, h2=0.5, r=1.08):
    """Cell edges: width h1 at the surface, growing geometrically (ratio r) up to h2, then constant."""
    w, e = h1, [0.0]
    while e[-1] < ymax - 1e-9:
        e.append(min(e[-1] + w, ymax))
        w = min(w * r, h2)
    return np.array(e)


def kernel_celulas(x, b, edges):
    """K[i, j] = exp(-b (x_i - y_j)^2) * w_j  (midpoint rule, true cell widths).
    Same kernel as common.kernel.build_kernel_matrix; identical to it for a uniform grid."""
    edges = np.asarray(edges, float)
    y = 0.5 * (edges[:-1] + edges[1:])
    w = np.diff(edges)
    x = np.atleast_1d(np.asarray(x, float))
    return y, np.exp(-b * (x[:, None] - y[None, :]) ** 2) * w[None, :]


def kerma_f5(y, tau, beta, a1, a2, a3):
    F = np.exp(-(np.asarray(y, float) / tau) ** beta)
    for a, lo, hi in zip((a1, a2, a3), ZONAS[:-1], ZONAS[1:]):
        F = np.where((y >= lo) & (y < hi), a, F)
    return F


def _pdd(F, K, Kd0):
    return 100.0 * (K @ F) / float((Kd0 @ F)[0])


# ------------------------------------------------------------------ per-curve fit (PDD space)
def ajustar_curva_v2(x, pdd, b, d0, edges=None, starts=None, max_nfev=3000):
    """Least-squares fit of the F5 parameters to ONE PDD curve, in PDD space (b fixed).
    Returns dict(params[5], rmse, pred, bound_hits)."""
    edges = grade_graduada(**GRADE_V2) if edges is None else np.asarray(edges, float)
    x = np.asarray(x, float); pdd = np.asarray(pdd, float)
    y, K = kernel_celulas(x, b, edges)
    _, Kd = kernel_celulas([d0], b, edges)

    def model(p):
        return _pdd(kerma_f5(y, *p), K, Kd)

    best = None
    for s0 in (starts or STARTS):
        p0 = np.clip(s0, LO + 1e-9, HI - 1e-9)
        try:
            r = least_squares(lambda p: model(p) - pdd, p0, bounds=(LO, HI), x_scale='jac', max_nfev=max_nfev)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("V2 fit failed for every start")
    p = best.x
    span = HI - LO
    hits = [n for n, v, l, h, s in zip(PNAMES, p, LO, HI, span) if min(v - l, h - v) < 1e-3 * s]
    pred = model(p)
    return dict(params=p, rmse=float(np.sqrt(np.mean((pred - pdd) ** 2))), pred=pred, bound_hits=hits)


def tabelar_v2(df_tidy, kermas, param_type, edges=None, verbose=False):
    """Per-curve V2 fits for every (field, SSD) row of df_tidy.

    Needs kermas[(campo, ssd)]['pdd_target'] and ['x_target'] (stored by
    kerma_batch.gerar_kermas_batch since V2); raises a clear error for old kermas.
    SSD is swept upwards for each field and the previous solution is added as an extra start
    (continuation), as in the experiments.
    """
    edges = grade_graduada(**GRADE_V2) if edges is None else np.asarray(edges, float)
    rows = []
    for campo, g in df_tidy.groupby('campo', sort=False):
        g = g.sort_values('ssd')
        prev = None
        for _, r in g.iterrows():
            k = kermas[(r['campo'], r['ssd'])]
            if 'pdd_target' not in k or 'x_target' not in k:
                raise ValueError("kermas lack 'pdd_target'/'x_target' (generated by an older version). "
                                 "Re-run 'Extract kermas' to use the V2 model.")
            st = list(STARTS) + ([prev] if prev is not None else [])
            fit = ajustar_curva_v2(k['x_target'], k['pdd_target'], float(r['b']), float(r['d0']), edges, st)
            prev = fit['params']
            rows.append({'campo': r['campo'], param_type: r[param_type], 'ssd': float(r['ssd']),
                         'b': float(r['b']), 'd0': float(r['d0']),
                         **{n: float(v) for n, v in zip(PNAMES, fit['params'])},
                         'rmse_v2': fit['rmse'], 'bound_hits_v2': len(fit['bound_hits'])})
            if verbose:
                print(f"  [V2] field={r['campo']:<8} SSD={r['ssd']:>6.1f}  rmse={fit['rmse']:.3f}  hits={fit['bound_hits']}")
    return pd.DataFrame(rows).sort_values([param_type, 'ssd']).reset_index(drop=True)


# ------------------------------------------------------------------ generalisation
def _pchip_lin(x, y):
    """PCHIP with linear extrapolation (end slope); a single node -> constant."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) == 1:
        return lambda t: np.full_like(np.asarray(t, float), y[0], dtype=float)
    p = PchipInterpolator(x, y); d = p.derivative()
    lo, hi = x[0], x[-1]

    def f(t):
        t = np.asarray(t, float)
        out = p(np.clip(t, lo, hi))
        return np.where(t < lo, y[0] + d(lo) * (t - lo), np.where(t > hi, y[-1] + d(hi) * (t - hi), out))
    return f


class ModeloKermaV2:
    """Tensor generalisation of the F5 parameters (and of the Direct b table) over (field, SSD)."""

    def __init__(self, df_v2, param_type, edges=None):
        self.param_type = param_type
        self.edges = grade_graduada(**GRADE_V2) if edges is None else np.asarray(edges, float)
        self.campos = np.array(sorted(df_v2[param_type].unique()), float)
        self.ssds = np.array(sorted(df_v2['ssd'].unique()), float)
        nf, ns = len(self.campos), len(self.ssds)
        if len(df_v2) != nf * ns:
            raise ValueError(f"V2 tensor model needs a rectangular (field x SSD) grid: {nf} x {ns} = {nf * ns} "
                             f"curves expected, {len(df_v2)} found.")
        self.P = np.zeros((nf, ns, len(PNAMES))); self.B = np.zeros((nf, ns))
        for i, f in enumerate(self.campos):
            for j, s in enumerate(self.ssds):
                r = df_v2[(df_v2[param_type] == f) & np.isclose(df_v2['ssd'], s)].iloc[0]
                self.P[i, j] = [r[n] for n in PNAMES]; self.B[i, j] = r['b']
        lf = np.log(self.campos)
        self._sp = [[_pchip_lin(lf, self.P[:, j, k]) for k in range(len(PNAMES))] for j in range(ns)]
        self._sb = [_pchip_lin(lf, self.B[:, j]) for j in range(ns)]
        self.rmse_treino = float(df_v2['rmse_v2'].mean()) if 'rmse_v2' in df_v2 else float('nan')
        self.d0 = float(df_v2['d0'].median())      # used only when no Direct surface is available

    def params(self, campo, ssd):
        """(tau, beta, a1, a2, a3) at (campo, ssd)."""
        lf = np.log(float(campo))
        out = [float(_pchip_lin(self.ssds, np.array([self._sp[j][k](lf) for j in range(len(self.ssds))]))(float(ssd)))
               for k in range(len(PNAMES))]
        return np.clip(out, LO, HI)

    def b_tabela(self, campo, ssd):
        """b interpolated from the Direct-fit b table (only used when no Direct surface is given)."""
        lf = np.log(float(campo))
        v = float(_pchip_lin(self.ssds, np.array([self._sb[j](lf) for j in range(len(self.ssds))]))(float(ssd)))
        return float(np.clip(v, 1.0, 4.5))


def reconstruir_pdd_indireta_v2(modelo: ModeloKermaV2, campo: float, ssd: float,
                                 n_pts: int = 300, x_max: float = 30.0,
                                 df_coef=None, param_type: str = None, grade_direto: dict = None):
    """PDD(campo, ssd) from the V2 model.  Same output convention as reconstruir_pdd_indireta:
    returns x, pdd, info.  b and d0 come from the same source as the legacy path (Direct surfaces
    when `grade_direto` is given), so the kernel is the legacy kernel."""
    if grade_direto is not None:
        from .reconstruction import _obter_b_d0       # same source as the legacy path (Direct B / d0 surfaces)
        b, d0, b_origem = _obter_b_d0(df_coef, campo, ssd, param_type or modelo.param_type, grade_direto)
    else:
        b, d0, b_origem = modelo.b_tabela(campo, ssd), modelo.d0, 'tabela de b do metodo Direto (V2, sem superficie)'
    x = np.linspace(0.0, x_max, n_pts)
    p = modelo.params(campo, ssd)
    y, K = kernel_celulas(x, b, modelo.edges)
    _, Kd = kernel_celulas([d0], b, modelo.edges)
    F = np.maximum(kerma_f5(y, *p), 1e-4)
    norm = float((Kd @ F)[0])
    if norm == 0:
        raise ValueError("Null normalization when reconstructing V2 PDD; check the fits.")
    pdd = 100.0 * (K @ F) / norm
    return x, pdd, {
        'campo': campo, 'ssd': ssd, 'b': b, 'd0': d0, 'b_origem': b_origem, 'modelo': 'v2',
        'A_previsto': 1.0,                      # relative amplitude (A cancels in the d0 normalisation)
        'tau_previsto': float(p[0]), 'beta_previsto': float(p[1]),
        'a1_previsto': float(p[2]), 'a2_previsto': float(p[3]), 'a3_previsto': float(p[4]),
        'y_grid': y, 'F_previsto': F, 'n_celulas': len(y),
    }
