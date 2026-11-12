#!/usr/bin/env python3
"""
direct/reconstruction.py
=========================
DIRECT approach — step 3: uses A(field,SSD) and B(field,SSD) surfaces
(direct/surfaces.py) to interpolate (a, b) at any (field, SSD) and
reconstroi a curva de PDD final via convolucao kernel(b) * kerma(a).

Tambem monta superficies 3D de PDD(profundidade, campo) ou
PDD(depth, SSD) varying one of the two parameters.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

from common.kernel import compute_pdd_convolucao
from .surfaces import gerar_superficie_ab, montar_superficies

CACHE_FILES = {
    'sup_a': 'superficie_A.csv',
    'sup_b': 'superficie_B.csv',
    'tidy': 'ab_tidy.csv',
}


def cache_existe(out_dir: str) -> bool:
    return all(os.path.isfile(os.path.join(out_dir, f)) for f in CACHE_FILES.values())


def _eval_rgi(interp, campo, ssd):
    return float(interp([[campo, ssd]])[0])


def _montar_interpoladores(superficie_a, superficie_b, df_tidy, param_type):
    campos_eixo = superficie_a.index.values.astype(float)
    ssds_eixo = superficie_a.columns.values.astype(float)

    kx = min(3, len(campos_eixo) - 1)
    ky = min(3, len(ssds_eixo) - 1)
    interp_a = RectBivariateSpline(campos_eixo, ssds_eixo, superficie_a.values, kx=kx, ky=ky)
    interp_b = RectBivariateSpline(campos_eixo, ssds_eixo, superficie_b.values, kx=kx, ky=ky)

    d0_pivot = df_tidy.pivot(index=param_type, columns='ssd', values='d0')
    d0_pivot = d0_pivot.sort_index().sort_index(axis=1)
    interp_d0 = RegularGridInterpolator(
        (d0_pivot.index.values.astype(float), d0_pivot.columns.values.astype(float)),
        d0_pivot.values, method='linear', bounds_error=False, fill_value=None,
    )

    return {
        'interp_a': interp_a, 'interp_b': interp_b, 'interp_d0': interp_d0,
        'superficie_a': superficie_a, 'superficie_b': superficie_b,
        'df_tidy': df_tidy, 'param_type': param_type,
        'campos_grade': campos_eixo, 'ssds_grade': ssds_eixo,
    }


def interpoladores_do_cache(out_dir: str) -> dict:
    print("\n  [CACHE] Superficies A/B encontradas — carregando sem reprocessar...")

    def ler_pivot(nome):
        path = os.path.join(out_dir, nome)
        df = pd.read_csv(path, sep=';', decimal=',', index_col=0)
        df.index = df.index.astype(str).str.replace(',', '.', regex=False).astype(float)
        df.columns = df.columns.astype(str).str.replace(',', '.', regex=False).astype(float)
        return df.sort_index().sort_index(axis=1)

    superficie_a = ler_pivot(CACHE_FILES['sup_a'])
    superficie_b = ler_pivot(CACHE_FILES['sup_b'])

    tidy_path = os.path.join(out_dir, CACHE_FILES['tidy'])
    df_tidy = pd.read_csv(tidy_path, sep=';', decimal=',')
    for col in ['a', 'b', 'd0', 'ssd', 'ym_scale', 'residual_SS']:
        if col in df_tidy.columns:
            df_tidy[col] = pd.to_numeric(
                df_tidy[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')

    cols_fixas = {'campo', 'ssd', 'd0', 'a', 'b', 'ym_scale', 'residual_SS'}
    param_type = next(c for c in df_tidy.columns if c not in cols_fixas)

    return _montar_interpoladores(superficie_a, superficie_b, df_tidy, param_type)


def construir_interpoladores(datasets, params, param_type, ssd1: float,
                              ssds_grade: np.ndarray, out_dir: str,
                              fixed_d0: float = None, verbose: bool = True) -> dict:
    if verbose:
        print("\n  Building A and B surfaces on (field, SSD) grid...")

    df_tidy = gerar_superficie_ab(datasets, params, param_type, ssd1, ssds_grade,
                                   fixed_d0=fixed_d0, verbose=verbose)
    superficie_a, superficie_b = montar_superficies(df_tidy, param_type)
    superficie_a = superficie_a.sort_index().sort_index(axis=1)
    superficie_b = superficie_b.sort_index().sort_index(axis=1)

    os.makedirs(out_dir, exist_ok=True)
    superficie_a.to_csv(os.path.join(out_dir, CACHE_FILES['sup_a']), sep=';', decimal=',')
    superficie_b.to_csv(os.path.join(out_dir, CACHE_FILES['sup_b']), sep=';', decimal=',')
    df_tidy.to_csv(os.path.join(out_dir, CACHE_FILES['tidy']), sep=';', decimal=',', index=False)
    print(f"  [CACHE] Superficies salvas em: {os.path.abspath(out_dir)}")

    return _montar_interpoladores(superficie_a, superficie_b, df_tidy, param_type)


def carregar_ou_construir_interpoladores(out_dir: str, datasets=None, params=None,
                                          param_type=None, ssd1: float = None,
                                          ssds_grade: np.ndarray = None,
                                          fixed_d0: float = None,
                                          forcar_rebuild: bool = False,
                                          verbose: bool = True) -> dict:
    if not forcar_rebuild and cache_existe(out_dir):
        return interpoladores_do_cache(out_dir)

    if datasets is None or ssd1 is None or ssds_grade is None:
        raise ValueError(
            f"Cache not found in '{out_dir}'. Provide datasets, ssd1 and "
            "ssds_grade para construir as superficies."
        )
    return construir_interpoladores(datasets, params, param_type, ssd1, ssds_grade,
                                     out_dir, fixed_d0, verbose)


def curva_pdd_interpolada(grade: dict, campo_desejado: float, ssd_desejada: float,
                           n_pts: int = 300, x_max: float = 30.0):
    """
    Interpolates (a, b, d0) on the `grade` surfaces for the (field, SSD)
    desejado e reconstroi a curva de PDD final via convolucao.
    """
    a_interp = float(grade['interp_a'].ev(campo_desejado, ssd_desejada))
    b_interp = float(grade['interp_b'].ev(campo_desejado, ssd_desejada))
    d0_interp = _eval_rgi(grade['interp_d0'], campo_desejado, ssd_desejada)

    if a_interp <= 0:
        raise ValueError(f"invalid interpolated a ({a_interp:.4f}). Field/SSD outside grid?")
    if b_interp <= 0:
        raise ValueError(f"invalid interpolated b ({b_interp:.4f}). Field/SSD outside grid?")

    x_curva = np.linspace(0.0, x_max, n_pts)
    pdd_conv = compute_pdd_convolucao(a_interp, b_interp, x_curva, d0_interp)

    return x_curva, pdd_conv, {
        'a': a_interp, 'b': b_interp, 'd0': d0_interp,
        'campo': campo_desejado, 'ssd': ssd_desejada,
        'campos_grade': grade['campos_grade'], 'ssds_grade': grade['ssds_grade'],
    }


def gerar_superficie_pdd(grade: dict, modo: str, valor_fixo: float,
                          valores_variados: np.ndarray, n_pts: int = 300,
                          x_max: float = 30.0):
    """
    mode == 'A': fixed SSD, varying field.
    mode == 'B': fixed field, varying SSD.
    """
    x_prof = np.linspace(0.0, x_max, n_pts)
    Z_pdd = np.zeros((n_pts, len(valores_variados)))

    for j, val in enumerate(valores_variados):
        campo_q, ssd_q = (val, valor_fixo) if modo == 'A' else (valor_fixo, val)
        a = float(grade['interp_a'].ev(campo_q, ssd_q))
        b = float(grade['interp_b'].ev(campo_q, ssd_q))
        d0 = _eval_rgi(grade['interp_d0'], campo_q, ssd_q)
        Z_pdd[:, j] = compute_pdd_convolucao(a, b, x_prof, d0)

    return x_prof, valores_variados, Z_pdd


def referencia_mais_proxima(grade: dict, campo: float, ssd: float) -> dict:
    """Finds, in the tidy table, the (field, SSD) row closest to the requested one."""
    df = grade['df_tidy'].copy()
    pt = grade['param_type']
    df['dist'] = np.sqrt(
        ((df[pt].astype(float) - campo) / max(campo, 1e-9)) ** 2 +
        ((df['ssd'] - ssd) / max(ssd, 1e-9)) ** 2
    )
    return df.loc[df['dist'].idxmin()].to_dict()
