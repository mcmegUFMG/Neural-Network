#!/usr/bin/env bash
# ============================================================
#  [LINUX] PDD Toolkit - cria uma venv local e instala as dependencias
# ============================================================
# Coloque este arquivo na mesma pasta do main_pyqt.py (raiz do projeto)
# e rode uma unica vez:
#
#     bash setup_venv.sh
#
# Ele NAO modifica main_pyqt.py nem nenhum outro arquivo do programa --
# so cria uma pasta .venv/ ao lado dele com um Python isolado.

set -e  # para o script no primeiro erro

# entra na pasta onde este script esta (raiz do projeto)
cd "$(dirname "$0")"

VENV_DIR=".venv"

echo "============================================================"
echo " [LINUX] PDD Toolkit - configuracao do ambiente"
echo "============================================================"
echo "Pasta do projeto: $(pwd)"
echo

# ---- 1. Verifica se o Python 3 esta instalado ----
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERRO: python3 nao encontrado."
    echo "Instale com:  sudo apt update && sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

echo "Python encontrado: $(python3 --version)"

# ---- 2. Verifica se o modulo venv esta disponivel ----
if ! python3 -c "import venv" >/dev/null 2>&1; then
    echo
    echo "ERRO: o modulo 'venv' do Python nao esta instalado."
    echo "No Ubuntu, instale com:"
    echo "    sudo apt update && sudo apt install python3-venv"
    exit 1
fi

# ---- 3. Cria a venv (se ainda nao existir) ----
if [ -d "$VENV_DIR" ]; then
    echo "Ambiente virtual '$VENV_DIR' ja existe -- pulando criacao."
else
    echo "Criando ambiente virtual em '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
fi

# ---- 4. Ativa a venv e instala as dependencias ----
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo
echo "Atualizando pip..."
pip install --upgrade pip

echo
if [ -f "requirements.txt" ]; then
    echo "Instalando dependencias a partir de requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt nao encontrado -- instalando o conjunto padrao:"
    echo "  PyQt5, numpy, pandas, scipy, matplotlib, openpyxl"
    pip install PyQt5 numpy pandas scipy matplotlib openpyxl
fi

deactivate

echo
echo "============================================================"
echo " Pronto! Ambiente configurado em: $(pwd)/$VENV_DIR"
echo " Para abrir o programa, use: bash abrir_pdd_toolkit.sh"
echo "============================================================"
