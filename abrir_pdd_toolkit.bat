@echo off
REM ============================================================
REM  PDD Toolkit - Atalho para abrir a interface (PyQt5)
REM ============================================================
REM Coloque este .bat na mesma pasta do main_pyqt.py (raiz do projeto).

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

python main_pyqt.py

if errorlevel 1 (
    echo.
    echo Ocorreu um erro ao rodar o programa.
    echo Verifique se as dependencias estao instaladas:
    echo   py -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
)
