#!/usr/bin/env python3
"""
direct/plotting.py
===================
Funcoes de plot da abordagem DIRETA: superficies 3D de A e B, superficie 3D
de PDD, familia de curvas 2D e o grafico de uma curva unica reconstruida.

Todos os textos que aparecem NOS graficos (labels, titulos, legendas) sao em
ingles. Cada funcao aceita opcionalmente `fixed_info`: um dict com os
parametros do feixe que NAO variam no arquivo de entrada (tipicamente
energy, and field OR SSD -- whichever is not the axis varying in the data) e que
o usuario informou na interface, para que fiquem visiveis no plot.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from .fitting import pdp_model


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


def plotar_superficie_ab(matriz, titulo, zlabel, out_path, elev=30, azim=-60,
                          fixed_info=None):
    """Plots a 3D surface (field x SSD x value of A or B)."""
    campos = matriz.index.values.astype(float)
    ssds = matriz.columns.values.astype(float)
    X, Y = np.meshgrid(ssds, campos)
    Z = matriz.values

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k',
                            linewidth=0.2, antialiased=True)
    ax.set_xlabel('SSD (cm)')
    ax.set_ylabel('Field')
    ax.set_zlabel(zlabel)
    ax.set_title(titulo)
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.08)
    _anotar_parametros_fixos(fig, fixed_info)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  OK Figure saved: {out_path}")


def plotar_superficie_pdd(x_prof, Y_variado, Z_pdd, modo, valor_fixo, out_path,
                           elev=30, azim=-60, fixed_info=None):
    ylabel = 'Field' if modo == 'A' else 'SSD (cm)'
    titulo = (f"PDD(depth, field)  |  SSD = {valor_fixo:.1f} cm"
              if modo == 'A' else
              f"PDD(depth, SSD)  |  Field = {valor_fixo:.2f}")

    Xm, Ym = np.meshgrid(x_prof, Y_variado, indexing='ij')
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xm, Ym, Z_pdd, cmap='viridis',
                            edgecolor='k', linewidth=0.15, antialiased=True, alpha=0.92)
    z_floor = Z_pdd.min() - 5
    ax.contourf(Xm, Ym, Z_pdd, zdir='z', offset=z_floor, cmap='viridis', alpha=0.25, levels=15)
    ax.set_xlabel('Depth (cm)', fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.set_zlabel('PDD (%)', fontsize=11, labelpad=8)
    ax.set_title(titulo, fontsize=13, pad=14)
    ax.set_zlim(z_floor, 105)
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.08).set_label('PDD (%)', fontsize=10)
    _anotar_parametros_fixos(fig, fixed_info)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  OK 3D Surface saved: {os.path.abspath(out_path)}")


def plotar_familia_curvas(x_prof, Y_variado, Z_pdd, modo, valor_fixo, out_path,
                           fixed_info=None):
    label_var = 'Field' if modo == 'A' else 'SSD (cm)'
    titulo = (f"PDD family  |  SSD = {valor_fixo:.1f} cm"
              if modo == 'A' else
              f"PDD family  |  Field = {valor_fixo:.2f}")
    colors = cm.viridis(np.linspace(0, 1, len(Y_variado)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, (val, col) in enumerate(zip(Y_variado, colors)):
        ax.plot(x_prof, Z_pdd[:, j], color=col, lw=1.6, label=f"{label_var} = {val:.2f}")
    ax.set_xlabel('Depth (cm)', fontsize=12)
    ax.set_ylabel('PDD (%)', fontsize=12)
    ax.set_title(titulo, fontsize=13)
    ax.grid(True, alpha=0.3)
    if len(Y_variado) <= 12:
        ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
    else:
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                    norm=plt.Normalize(Y_variado.min(), Y_variado.max()))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=label_var, pad=0.01)
    _anotar_parametros_fixos(fig, fixed_info)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  OK Curve family saved: {os.path.abspath(out_path)}")


def plotar_curva(x_curva, pdd_conv, info, ref, datasets_orig, ssd1, param_type, out_path,
                  fixed_info=None):
    """Grafico de uma unica curva PDD reconstruida, com o modelo analitico e
    (se disponivel) o dado experimental mais proximo como referencia."""
    fig, ax = plt.subplots(figsize=(9, 5))

    pdd_analitica = pdp_model(x_curva, info['a'], info['b'])
    norm_analitica = float(np.ravel(pdp_model(np.array([info['d0']]), info['a'], info['b']))[0])
    if norm_analitica != 0:
        pdd_analitica = 100.0 * pdd_analitica / norm_analitica

    ax.plot(x_curva, pdd_analitica, 'r--', lw=1.5, alpha=0.8,
            label=f"Analytical PDP (a={info['a']:.4f}, b={info['b']:.4f})")
    ax.plot(x_curva, pdd_conv, 'b-', lw=2.5,
            label=f"PDD K*f  |  field={info['campo']}  SSD={info['ssd']} cm")

    if ref is not None and datasets_orig:
        chave_ref = ref.get('campo')
        if chave_ref and chave_ref in datasets_orig:
            from common.mayneord import apply_mayneord
            xm_ref, ym_ref = datasets_orig[chave_ref]
            if abs(ref['ssd'] - ssd1) > 0.1:
                ym_ref = apply_mayneord(ym_ref, xm_ref, ssd1, ref['ssd'], ref.get('d0', info['d0']))
            ax.scatter(xm_ref, ym_ref, s=30, color='gray', zorder=5,
                       label=f"Measured data  |  {param_type}={ref.get(param_type, '')}  "
                             f"SSD={ref['ssd']:.1f} cm")

    ax.axvline(info['d0'], color='green', lw=1, ls=':', alpha=0.7,
               label=f"d0 = {info['d0']:.1f} cm")
    ax.set_xlabel('Depth (cm)', fontsize=12)
    ax.set_ylabel('PDD (%)', fontsize=12)
    ax.set_title(f"Interpolated PDD  |  Field = {info['campo']}  |  SSD = {info['ssd']} cm",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _anotar_parametros_fixos(fig, fixed_info)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  OK Curve saved: {os.path.abspath(out_path)}")
