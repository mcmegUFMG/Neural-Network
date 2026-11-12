#!/usr/bin/env bash
# ============================================================
#  [LINUX] PDD Toolkit - Atalho para abrir a interface (PyQt5)
#  Equivalente Linux do abrir_pdd_toolkit.bat
# ============================================================
# Coloque este arquivo na mesma pasta do main_pyqt.py (raiz do projeto).
# Rode primeiro o setup_venv.sh uma unica vez (cria o ambiente);
# depois, use sempre este script para abrir o programa:
#
#     bash abrir_pdd_toolkit.sh

cd "$(dirname "$0")"

VENV_DIR=".venv"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo
    echo "Ambiente virtual nao encontrado em '$VENV_DIR'."
    echo "Rode primeiro:  bash setup_venv.sh"
    echo
    read -rp "Pressione Enter para sair..."
    exit 1
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

python main_pyqt.py
STATUS=$?

deactivate

if [ $STATUS -ne 0 ]; then
    echo
    echo "Ocorreu um erro ao rodar o programa. Verifique se as dependencias"
    echo "estao instaladas corretamente (rode: bash setup_venv.sh)."
    echo
    read -rp "Pressione Enter para sair..."
fi
