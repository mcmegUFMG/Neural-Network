#!/usr/bin/env python3
"""
direct/comparison.py
=====================
Compara as curvas de PDD geradas pela reconstrucao direta (direct/reconstruction.py)
com dados de referencia (medidos/simulados) para cada campo, lidos de arquivos
.xlsx (uma ou varias abas por arquivo), buscados recursivamente numa pasta base.

Cada aba deve conter colunas "Depth (cm)", "PDD", ... e uma coluna de cabecalho
type "SSD = 354 cm".
"""

import os
import re
import glob
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .reconstruction import curva_pdd_interpolada

CAMPO_REGEX = re.compile(r'(\d+(?:[.,]\d+)?)\s*[xX]\s*(\d+(?:[.,]\d+)?)')


def fix_depth_value(v):
    """Corrige profundidades tipo 'X,5' que o Excel converteu em data."""
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)):
        return float(v.month) + float(v.day) / 10.0
    if isinstance(v, str):
        return float(v.strip().replace(',', '.'))
    return float(v)


def parse_campo(texto):
    if texto is None:
        return None
    m = CAMPO_REGEX.search(str(texto))
    if not m:
        return None
    lado1 = float(m.group(1).replace(',', '.'))
    lado2 = float(m.group(2).replace(',', '.'))
    return (lado1 + lado2) / 2.0


def resolver_campo(sheet_name, file_path):
    for fonte in (sheet_name, os.path.basename(os.path.dirname(file_path)),
                  os.path.basename(file_path)):
        campo = parse_campo(fonte)
        if campo is not None:
            return campo
    raise ValueError(f"Could not identify the field (aba='{sheet_name}', arquivo='{file_path}')")


def parse_ssd_from_header(columns):
    for col in columns:
        m = re.search(r'SSD\s*=\s*(\d+(?:[.,]\d+)?)', str(col))
        if m:
            return float(m.group(1).replace(',', '.'))
    raise ValueError(f"Nao foi possivel localizar 'SSD = ... cm' nas colunas: {list(columns)}")


def carregar_referencias(base_dir, filename):
    padrao = os.path.join(base_dir, '**', filename)
    arquivos = sorted(glob.glob(padrao, recursive=True))
    if not arquivos:
        raise SystemExit(f"Nenhum arquivo '{filename}' encontrado em subpastas de '{os.path.abspath(base_dir)}'.")

    referencias = []
    for path in arquivos:
        try:
            sheets = pd.read_excel(path, sheet_name=None)
        except Exception as e:
            print(f"  AVISO: erro ao abrir '{path}': {e}")
            continue

        for sheet_name, df in sheets.items():
            try:
                campo = resolver_campo(sheet_name, path)
                ssd = parse_ssd_from_header(df.columns)
                col_prof = next(c for c in df.columns if 'depth' in str(c).lower())
                col_pdd = next(c for c in df.columns if str(c).strip().lower() == 'pdd')

                sub = df[[col_prof, col_pdd]].dropna()
                sub.columns = ['profundidade_cm', 'pdd_ref']
                sub['profundidade_cm'] = sub['profundidade_cm'].apply(fix_depth_value)
                sub['pdd_ref'] = sub['pdd_ref'].astype(float)
                sub = sub.sort_values('profundidade_cm').reset_index(drop=True)

                referencias.append({
                    'arquivo': f"{os.path.relpath(path, base_dir)}::{sheet_name}",
                    'campo': campo, 'ssd': ssd,
                    'x_ref': sub['profundidade_cm'].values.astype(float),
                    'pdd_ref': sub['pdd_ref'].values.astype(float),
                })
            except Exception as e:
                print(f"  AVISO: erro ao processar aba '{sheet_name}' de '{path}': {e}")
                continue
    return referencias


def comparar_campo(grade, ref, n_pts=300, x_max=30.0):
    x_mod, pdd_mod, info = curva_pdd_interpolada(grade, ref['campo'], ref['ssd'], n_pts=n_pts, x_max=x_max)
    pdd_mod_em_x_ref = np.interp(ref['x_ref'], x_mod, pdd_mod)
    residuo = ref['pdd_ref'] - pdd_mod_em_x_ref
    return {
        **ref, 'x_mod': x_mod, 'pdd_mod': pdd_mod,
        'pdd_mod_interp_em_x_ref': pdd_mod_em_x_ref, 'residuo': residuo,
        'rmse': float(np.sqrt(np.mean(residuo ** 2))),
        'mae': float(np.mean(np.abs(residuo))),
        'max_err': float(np.max(np.abs(residuo))),
        'info': info,
    }


def plotar_comparacao_individual(resultado, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(resultado['x_ref'], resultado['pdd_ref'], 'ko', ms=4,
             label=f"Reference (field={resultado['campo']:.1f}, SSD={resultado['ssd']:.1f} cm)")
    ax1.plot(resultado['x_mod'], resultado['pdd_mod'], 'b-', lw=2, label="Reconstrucao direta (K*f)")
    ax1.set_ylabel('PDD (%)')
    ax1.set_title(f"Comparacao PDD -- {resultado['arquivo']}  "
                  f"(RMSE={resultado['rmse']:.2f}, MAE={resultado['mae']:.2f})")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0, color='gray', lw=1)
    ax2.plot(resultado['x_ref'], resultado['residuo'], 'r.-', lw=1, ms=4)
    ax2.set_xlabel('Depth (cm)')
    ax2.set_ylabel('Residual\n(Ref - Modelo)')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    stem = resultado['arquivo'].replace(os.sep, '_').replace('::', '_')
    out_path = os.path.join(out_dir, f"comparacao_{stem}.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  OK plot saved: {out_path}")


def plotar_resumo_todos_campos(resultados, out_dir):
    resultados_ordenados = sorted(resultados, key=lambda r: r['campo'])
    colors = plt.cm.viridis(np.linspace(0, 1, len(resultados_ordenados)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for r, col in zip(resultados_ordenados, colors):
        ax.plot(r['x_ref'], r['pdd_ref'], 'o', color=col, ms=3, alpha=0.6)
        ax.plot(r['x_mod'], r['pdd_mod'], '-', color=col, lw=2,
                label=f"field={r['campo']:.1f}  (RMSE={r['rmse']:.2f})")
    ax.set_xlabel('Depth (cm)')
    ax.set_ylabel('PDD (%)')
    ax.set_title('Reconstrucao direta vs. Referencia -- todos os campos')
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "comparacao_resumo_todos_campos.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  OK summary saved: {out_path}")


def plotar_erro_vs_campo(resultados, out_dir):
    resultados_ordenados = sorted(resultados, key=lambda r: r['campo'])
    campos = [r['campo'] for r in resultados_ordenados]
    rmses = [r['rmse'] for r in resultados_ordenados]
    maes = [r['mae'] for r in resultados_ordenados]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(campos, rmses, 'o-', label='RMSE')
    ax.plot(campos, maes, 's--', label='MAE')
    ax.set_xlabel('Field (equivalent side, cm)')
    ax.set_ylabel('Error (PDD percentage points)')
    ax.set_title('Model error vs. field size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "erro_vs_campo.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  OK error plot saved: {out_path}")
