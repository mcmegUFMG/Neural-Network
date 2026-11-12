#!/usr/bin/env bash
# ============================================================
#  PDD Toolkit - instalador para Linux x86_64
# ============================================================
# Baixa o executavel pronto (nao precisa de Python) do ultimo
# release do GitHub, instala em ~/.local/opt/pdd-toolkit e cria
# o comando `pdd-toolkit` em ~/.local/bin.
#
# Uso:
#   curl -fsSL https://raw.githubusercontent.com/lsbeloti/pdd-toolkit/main/install.sh | bash
#
# Variaveis opcionais:
#   PDD_REPO=usuario/repo   outro repositorio
#   PDD_HOME=/caminho       outro diretorio de instalacao
# ============================================================
set -euo pipefail

REPO="${PDD_REPO:-lsbeloti/pdd-toolkit}"
ASSET="pdd-toolkit-linux-x86_64.tar.gz"
APP="${PDD_HOME:-$HOME/.local/opt/pdd-toolkit}"
BIN="$HOME/.local/bin"
URL="https://github.com/$REPO/releases/latest/download/$ASSET"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64|Linux-amd64) ;;
  *) echo "ERRO: este instalador cobre apenas Linux x86_64 (detectado: $(uname -s) $(uname -m))." >&2; exit 1 ;;
esac

if ! command -v curl >/dev/null 2>&1; then
  echo "ERRO: 'curl' nao encontrado. Instale com: sudo apt install curl" >&2
  exit 1
fi

echo "==> Baixando PDD Toolkit de $REPO"
mkdir -p "$APP" "$BIN"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if ! curl -fL --progress-bar "$URL" -o "$tmp/pdd.tar.gz"; then
  echo >&2
  echo "ERRO: download falhou: $URL" >&2
  echo "Verifique se o repositorio ja tem um release publicado (aba 'Releases')." >&2
  exit 1
fi

# remove instalacao anterior (evita arquivos obsoletos de versoes antigas)
rm -rf "$APP/PDD_Toolkit"

echo "==> Extraindo em $APP"
tar -xzf "$tmp/pdd.tar.gz" -C "$APP"
chmod +x "$APP/PDD_Toolkit/PDD_Toolkit"

echo "==> Criando o comando 'pdd-toolkit'"
cat > "$BIN/pdd-toolkit" <<EOF
#!/usr/bin/env bash
exec "$APP/PDD_Toolkit/PDD_Toolkit" "\$@"
EOF
chmod +x "$BIN/pdd-toolkit"

# atalho no menu de aplicativos (opcional, sem quebrar se falhar)
if command -v update-desktop-database >/dev/null 2>&1; then
  apps="$HOME/.local/share/applications"
  mkdir -p "$apps"
  cat > "$apps/pdd-toolkit.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=PDD Toolkit
Comment=Curvas de PDD: ajuste, Mayneord, reconstrucao e comparacao
Exec=$BIN/pdd-toolkit
Terminal=false
Categories=Science;Education;
EOF
  update-desktop-database "$apps" >/dev/null 2>&1 || true
fi

# teste rapido: nao e fatal (build sem console nao imprime nada)
if "$BIN/pdd-toolkit" --selftest >/dev/null 2>&1; then
  echo "==> Autoteste: OK"
else
  echo "==> Aviso: autoteste nao retornou OK. O programa pode falhar ao abrir"
  echo "    (bibliotecas Qt/OpenGL ausentes?). Tente:"
  echo "    sudo apt install libgl1 libxcb-cursor0"
fi

echo
echo "============================================================"
echo " Pronto! Abra com:  pdd-toolkit"
echo " Instalado em:      $APP"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *)
    echo
    echo " ATENCAO: $BIN nao esta no seu PATH. Adicione ao ~/.bashrc:"
    echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
    ;;
esac
echo "============================================================"
