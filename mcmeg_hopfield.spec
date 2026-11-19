# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec do MCMEG - Hopfield Network (interface PyQt5).

Gera um pacote autossuficiente (nao precisa de Python nem das bibliotecas
instaladas) a partir de run_mcmeg_hopfield.py.

Uso:
    pyinstaller --noconfirm mcmeg_hopfield.spec

Variaveis de ambiente:
    MCMEG_ONEFILE=1   gera um unico executavel (mais lento para abrir, mais
                    simples de distribuir). Padrao: pasta (onedir).
    MCMEG_CONSOLE=1   mantem o console (util para depurar). Padrao: sem console.

O resultado fica em dist/MCMEG_Hopfield (ou dist/MCMEG_Hopfield[.exe] no onefile).
"""

import os

proj = os.path.abspath(SPECPATH)
onefile = os.environ.get("MCMEG_ONEFILE", "0") == "1"
console = os.environ.get("MCMEG_CONSOLE", "0") == "1"

# nenhum arquivo de dados e embutido no pacote
datas = []

hiddenimports = [
    "matplotlib.backends.backend_agg",
]

a = Analysis(
    ["run_mcmeg_hopfield.py"],
    pathex=[os.path.join(proj, "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # nao usados pela interface; reduz tamanho
        "tkinter", "IPython", "jupyter", "notebook", "pytest",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

if onefile:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="MCMEG_Hopfield",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=console,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="MCMEG_Hopfield",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=console,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="MCMEG_Hopfield",
    )
