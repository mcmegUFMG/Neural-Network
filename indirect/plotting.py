#!/usr/bin/env python3
"""
indirect/plotting.py
======================
Funcoes de plot da abordagem INDIRETA: kermas F(y) extraidos pela rede de
Hopfield/OLS, variacao dos coeficientes (A, tau, beta) da exponencial
stretched with field and SSD, and the final reconstructed PDD curve from the
kerma previsto.

Todos os textos que aparecem NOS graficos (labels, titulos, legendas) sao em
ingles -- inclusive quando `param_type` (identificador interno, ex.: 'campo'
ou 'ssd') e usado para rotular um eixo, ele passa por `_rotulo_param` antes
de ir para o plot. Cada funcao aceita opcionalmente `fixed_info`: um dict
com os parametros do feixe que NAO variam no arquivo de entrada
(typically energy, and field OR SSD -- whichever is not the axis varying in the
data) and that the user entered in the interface, so they are visible in the
plot.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


_PARAM_LABELS = {
    'campo': 'Field size',
    'ssd': 'SSD (cm)',
    'energia': 'Energy (MeV)',
}


def _rotulo_param(param_type):
    """Translates the internal `param_type` identifier (e.g.: 'field', 'ssd')
    to an English label suitable for appearing on a plot axis/title.
    plot. Any unmapped value is returned capitalized, never
    in raw Portuguese."""
    return _PARAM_LABELS.get(str(param_type).lower(), str(param_type).replace('_', ' ').capitalize())


def _fmt_valor(v):
    """Formats `v` as compact number when possible; otherwise
    returns str(v) (field can be a text label, e.g.: '10x10')."""
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def _anotar_parametros_fixos(fig, fixed_info):
    """Adds, in the figure footer, the fixed beam parameters
    (not varying in the input file) and entered by the user in the
    interface -- energy, and field or SSD, as applicable. Does nothing if
    `fixed_info` is None/empty or if none of the values are present."""
    if not fixed_info:
        return
    partes = []
    if fixed_info.get('energia') is not None:
        partes.append(f"Energy = {_fmt_valor(fixed_info['energia'])} MeV")
    if fixed_info.get('campo') is not None:
        partes.append(f"Field = {_fmt_valor(fixed_info['campo'])}")
    if fixed_info.get('ssd') is not None:
        partes.append(f"SSD = {_fmt_valor(fixed_info['ssd'])} cm")
    if partes:
        fig.text(0.99, 0.01, "  |  ".join(partes), ha='right', va='bottom',
                  fontsize=8, style='italic', color='dimgray')


def plotar_kermas(kermas: dict, out_path: str, titulo="Extracted kermas F(y)",
                   fixed_info=None):
    """Plots all F(y) from a dict {(field, ssd): {...}} in a single figure."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    chaves = sorted(kermas.keys())
    cores = plt.cm.viridis(np.linspace(0, 1, len(chaves)))
    for (campo, ssd), cor in zip(chaves, cores):
        k = kermas[(campo, ssd)]
        ax.plot(k['y'], k['F'], color=cor, lw=1.4, alpha=0.85,
                label=f"field={campo}, SSD={ssd:.1f}")
    ax.set_xlabel('y (cm)')
    ax.set_ylabel('F(y)  (numerical kerma)')
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    if len(chaves) <= 12:
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    _anotar_parametros_fixos(fig, fixed_info)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  OK Figure saved: {out_path}")


def plotar_coeficiente_vs_campo_ssd(df_coef, coluna, out_path, param_type='campo',
                                     fixed_info=None):
    """
    Simple 3D scatter of `coluna` (e.g.: beta_kerma) as a function of (field, ssd),
    for visual inspection of variation before/after applying the placeholder.
    """
    rotulo_param = _rotulo_param(param_type)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df_coef[param_type], df_coef['ssd'], df_coef[coluna],
                     c=df_coef[coluna], cmap='viridis', s=40, edgecolor='k', linewidth=0.3)
    ax.set_xlabel(rotulo_param)
    ax.set_ylabel('SSD (cm)')
    ax.set_zlabel(coluna)
    ax.set_title(f"Variation of {coluna} with ({rotulo_param}, SSD)")
    fig.colorbar(sc, shrink=0.6, aspect=12, pad=0.08)
    _anotar_parametros_fixos(fig, fixed_info)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  OK Figure saved: {out_path}")


def plotar_ajuste_modelo(df_coef, fit_dict, out_path, param_type='campo',
                          fixed_info=None):
    """
    Compares the computed points of `fit_dict['coluna']` with the surface of the
    fitted generalized model (bivariate saturating exponential, applied to
    A, tau or beta), on a regular (field, ssd) grid.
    """
    rotulo_param = _rotulo_param(param_type)
    coluna = fit_dict['coluna']
    campo_vals = np.linspace(df_coef[param_type].min(), df_coef[param_type].max(), 30)
    ssd_vals = np.linspace(df_coef['ssd'].min(), df_coef['ssd'].max(), 30)
    Campo, Ssd = np.meshgrid(campo_vals, ssd_vals)
    Z = fit_dict['prever'](Campo.ravel(), Ssd.ravel()).reshape(Campo.shape)

    r2_txt = f", R²={fit_dict['r2_treino']:.4g}" if fit_dict.get('r2_treino') is not None else ""

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Campo, Ssd, Z, cmap='viridis', alpha=0.55, edgecolor='none')
    ax.scatter(df_coef[param_type], df_coef['ssd'], df_coef[coluna],
               color='red', s=25, label='Computed values (per-curve stretched-exponential fit)')
    ax.set_xlabel(rotulo_param)
    ax.set_ylabel('SSD (cm)')
    ax.set_zlabel(coluna)
    ax.set_title(f"Model for {coluna}  (training RMSE={fit_dict['rmse_treino']:.4g}{r2_txt})")
    ax.legend(fontsize=8, loc='upper left')
    _anotar_parametros_fixos(fig, fixed_info)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  OK Figure saved: {out_path}")
    print(f"  {fit_dict['aviso']}")


# Alias de compatibilidade com o nome antigo.
plotar_ajuste_placeholder = plotar_ajuste_modelo


def plotar_pdd_reconstruida(x, pdd, info, out_path, campo=None, ssd=None,
                             fixed_info=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    label_curva = "Inverse PDD"
    if campo is not None and ssd is not None:
        label_curva += f"  |  field={_fmt_valor(campo)}  SSD={_fmt_valor(ssd)} cm"
    ax.plot(x, pdd, 'b-', lw=2.5, label=label_curva)
    ax.axvline(info['d0'], color='green', lw=1, ls=':', alpha=0.7,
               label=f"d0 = {info['d0']:.1f} cm (interpolated)")
    ax.set_xlabel('Depth (cm)')
    ax.set_ylabel('PDD (%)')
    titulo = (
        f"Inverse reconstruction  |  A={info.get('A_previsto', float('nan')):.4f}  "
        f"τ={info.get('tau_previsto', float('nan')):.4f}  "
        f"β={info.get('beta_previsto', float('nan')):.4f}  "
        f"b={info['b']:.4f}"
    )
    if campo is not None and ssd is not None:
        titulo = f"Field = {_fmt_valor(campo)}  |  SSD = {_fmt_valor(ssd)} cm\n" + titulo
    ax.set_title(titulo, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _anotar_parametros_fixos(fig, fixed_info)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  OK Curve saved: {out_path}")
