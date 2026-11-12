#!/usr/bin/env python3
"""
main_pyqt.py
============
PDD Toolkit -- PyQt5 interface.

Replaces the web interface (Streamlit, main.py) keeping ALL
functionalities:

  * Direct tab  : base CSV load, (A,B) fit, Mayneord, A/B surfaces,
                  point reconstruction, 3D surface.
  * Inverse tab : kermas (Hopfield/Fourier), coefficients, model
                  generalized (A, tau, beta), advanced and simple
                  reconstruction.
  * Comparison tab : independent tabs, reference curves (loaded from
                  file or manual entry), comparison and error vs reference.

All computation (common/, direct/, indirect/) is reused unchanged --
this file is only the interface layer.

Usage:
    python main_pyqt.py
"""

import os
import sys
import shutil
import traceback

import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

# matplotlib only for figures generated inside this interface
# (Mayneord and comparison); plot modules already fix the Agg backend.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# For interactive 3D plots
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def _resource_dir():
    """Read-only resource directory. When frozen (PyInstaller),
    points to the temporary `_internal` folder where data was embedded;
    in development, points to the script's own folder."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", BASE_DIR)
    return BASE_DIR


def _app_dir():
    """Writable directory for outputs. When frozen, tries the executable's
    folder (portable distribution); if not writable (e.g., installed in
    a protected location), falls back to ~/PDD_Toolkit. In development,
    uses the project folder."""
    if not getattr(sys, "frozen", False):
        return BASE_DIR
    candidato = os.path.dirname(os.path.abspath(sys.executable))
    try:
        teste = os.path.join(candidato, ".pdd_write_test")
        with open(teste, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(teste)
        return candidato
    except OSError:
        alternativo = os.path.join(os.path.expanduser("~"), "PDD_Toolkit")
        os.makedirs(alternativo, exist_ok=True)
        return alternativo


RESOURCE_DIR = _resource_dir()
APP_DIR = _app_dir()
DATA_DIR = RESOURCE_DIR  # folder the file dialogs open in

from common.io_utils import load_csv, save_df
from direct.fitting import fit_all_curves
from direct.surfaces import detect_dmax_xy
from direct.reconstruction import (
    carregar_ou_construir_interpoladores, curva_pdd_interpolada,
    gerar_superficie_pdd, referencia_mais_proxima, cache_existe,
)
from direct.plotting import (
    plotar_superficie_ab, plotar_superficie_pdd, plotar_familia_curvas, plotar_curva,
)
from indirect.kerma_batch import gerar_kermas_batch
from indirect.kerma_variation import (
    tabelar_coeficientes, ajustar_modelo_kerma, FORMULA_KERMA, FORMULA_CONSTANTES,
)
from indirect.reconstruction import (
    reconstruir_pdd_indireta, reconstruir_pdd_indireta_simples,
)
from indirect.plotting import (
    plotar_kermas, plotar_coeficiente_vs_campo_ssd, plotar_ajuste_modelo,
    plotar_pdd_reconstruida,
)

APP_TITLE = "PDD Toolkit"


def _executar_autoteste():
    """Headless self-test (--selftest), used to validate the frozen package.

    Exercises what packaging can break: imports of all layers, CSV parsing,
    numpy/scipy/pandas, matplotlib (fonts/backends) e openpyxl.

    Nao depende de dados de exemplo no repositorio: a curva usada e sintetica
    e gerada aqui mesmo (os dados de laboratorio nao sao publicos).
    """
    import tempfile
    from direct.fitting import fit_pdp, pdp_model
    from common.kernel import build_kernel_matrix

    print("[selftest] resource =" , RESOURCE_DIR)
    print("[selftest] app      =", APP_DIR)

    # Curva sintetica no mesmo formato dos CSVs de entrada (';', decimal '.').
    tmp = tempfile.mkdtemp(prefix="pdd_selftest_")
    csv_path = os.path.join(tmp, "synthetic.csv")
    prof = np.arange(0.0, 31.0)
    dose = 100.0 * np.exp(-0.035 * prof) * (1.0 + 0.05 * prof)
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("x (cm);6x6\n")
        for z, d in zip(prof, dose):
            fh.write(f"{z:g};{d:.6g}\n")

    datasets, params, ptype, punit = load_csv(csv_path)
    if not datasets:
        raise RuntimeError("synthetic CSV has no curves: " + csv_path)

    label = next(iter(datasets))
    xm, ym = datasets[label]
    a, b, escala = fit_pdp(xm, ym)

    x = np.linspace(0.0, 30.0, 64)
    y, K = build_kernel_matrix(x, b)
    if not np.all(np.isfinite(K)):
        raise RuntimeError("kernel matrix has non-finite values")

    # matplotlib + bundled fonts
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(xm, ym, label=label)
    ax.plot(x, pdp_model(x, a, b) * escala, label="fit")
    ax.legend()
    png = os.path.join(tmp, "selftest.png")
    fig.savefig(png, dpi=80)
    plt.close(fig)

    # openpyxl
    df = pd.DataFrame({"x": x, "pdd": pdp_model(x, a, b) * escala})
    xlsx = os.path.join(tmp, "selftest.xlsx")
    df.to_excel(xlsx, index=False)

    # indirect layer imports
    from indirect import hopfield, kerma_batch, kerma_variation  # noqa: F401
    from indirect import reconstruction as _rec, plotting as _plot  # noqa: F401

    print(f"[selftest] csv={os.path.basename(csv_path)} label={label} "
          f"type={ptype} a={a:.4f} b={b:.4f} K={K.shape}")
    print(f"[selftest] png={os.path.getsize(png)} bytes")
    print(f"[selftest] xlsx={os.path.getsize(xlsx)} bytes")

    # planilha de entrada manual (aba Comparison): colagem estilo Excel
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tbl = PasteTableWidget(8, 2)
    colagem = "0\t49,8\n1\t98,1\n1,5\t100\n"
    if not tbl.paste_text(colagem):
        raise RuntimeError("clipboard paste returned False")
    if (tbl.item(0, 0).text(), tbl.item(0, 1).text()) != ("0", "49,8"):
        raise RuntimeError("clipboard paste wrote wrong values")
    if tbl.item(2, 0).text() != "1,5" or tbl.rowCount() < 4:
        raise RuntimeError("clipboard paste did not grow the grid")
    if PasteTableWidget.parse_clipboard("0;49.8\n1;98.1") != \
            [["0", "49.8"], ["1", "98.1"]]:
        raise RuntimeError("csv paste parsing failed")
    app.processEvents()
    print(f"[selftest] paste rows={tbl.rowCount()} cols={tbl.columnCount()}")

    print("SELFTEST OK")
    return 0


# ============================================================
#  GENERAL HELPERS
# ============================================================

def subdir(base_dir, *partes):
    """Creates (and returns) a subfolder inside the output directory."""
    caminho = os.path.join(base_dir, *partes)
    os.makedirs(caminho, exist_ok=True)
    return caminho


def fmt_num(v):
    """Formats `v` as compact number (:g) or str(v) if not numeric."""
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)


def _rotulo_param(param_type):
    """Readable axis label for the sweep parameter (axis name)."""
    if param_type == "ssd":
        return "SSD (cm)"
    return "Field size (cm)"


def linha_treino_mais_proxima(df_coef, param_type, campo, ssd):
    """Row in df_coef closest to (campo, ssd) in Euclidean distance
    (used only for advanced reconstruction diagnostics)."""
    campo_f = pd.to_numeric(df_coef[param_type], errors='coerce').astype(float)
    ssd_f = pd.to_numeric(df_coef['ssd'], errors='coerce').astype(float)
    try:
        campo_alvo = float(campo)
    except (TypeError, ValueError):
        campo_alvo = np.nan
    d2 = (campo_f - campo_alvo) ** 2 + (ssd_f - float(ssd)) ** 2
    idx = d2.idxmin()
    return df_coef.loc[idx], float(d2.loc[idx] ** 0.5)


def dspin(minimum, maximum, value, decimals=2, step=1.0, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(minimum, maximum)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    if suffix:
        w.setSuffix(suffix)
    return w


def ispin(minimum, maximum, value, step=1):
    w = QtWidgets.QSpinBox()
    w.setRange(minimum, maximum)
    w.setSingleStep(step)
    w.setValue(value)
    return w


def titulo(texto):
    lbl = QtWidgets.QLabel(texto)
    f = lbl.font()
    f.setBold(True)
    lbl.setFont(f)
    lbl.setWordWrap(True)
    return lbl


# ============================================================
#  WIDGETS REUTILIZAVEIS
# ============================================================

class StreamEmitter(QtCore.QObject):
    """Redireciona sys.stdout para o painel de log (thread-safe via signal)."""
    text = QtCore.pyqtSignal(str)

    def write(self, s):
        self.text.emit(str(s))

    def flush(self):
        pass

    def isatty(self):
        return False


class TaskWorker(QtCore.QThread):
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            resultado = self.fn()
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.finished_ok.emit(resultado)


class ImageView(QtWidgets.QLabel):
    """Mostra um PNG salvo em disco, reescalando conforme o tamanho do widget.
    Duplo-clique abre a imagem no tamanho original.
    Clique com o botao direito abre menu com opcoes: Fullscreen, Salvar, Copiar.
    """

    def __init__(self, min_height=240):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(min_height)
        self.setStyleSheet("border: 1px solid #c8c8c8; background: #fafafa;")
        self._pix = None
        self._image_path = None
        self.setText("(no image)")
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def set_image(self, path):
        self._image_path = path
        if not path or not os.path.exists(path):
            self._pix = None
            self.setPixmap(QtGui.QPixmap())
            self.setText("(no image)")
            return
        pix = QtGui.QPixmap(path)
        if pix.isNull():
            self._pix = None
            self.setText("(failed to open image)")
            return
        self._pix = pix
        self.setText("")
        self._rescale()

    def _rescale(self):
        if self._pix is None:
            return
        w = max(self.width() - 6, 80)
        self.setPixmap(self._pix.scaledToWidth(w, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def mouseDoubleClickEvent(self, event):
        if self._pix is None:
            return
        self._open_fullscreen()

    def _show_context_menu(self, pos):
        if self._pix is None:
            return
        menu = QtWidgets.QMenu(self)
        act_fs = menu.addAction("Fullscreen (Fullscreen)")
        act_fs.triggered.connect(self._open_fullscreen)
        act_save = menu.addAction("Save as...")
        act_save.triggered.connect(self._save_as)
        act_copy = menu.addAction("Copy image")
        act_copy.triggered.connect(self._copy_to_clipboard)
        menu.exec_(self.mapToGlobal(pos))

    def _open_fullscreen(self):
        if self._pix is None:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Image - Fullscreen")
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowMaximizeButtonHint)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        lbl = QtWidgets.QLabel()
        lbl.setPixmap(self._pix)
        lbl.setAlignment(Qt.AlignCenter)
        scroll.setWidget(lbl)
        lay.addWidget(scroll)
        dlg.resize(1200, 900)
        dlg.showMaximized()
        dlg.exec_()

    def _save_as(self):
        if not self._image_path or not os.path.exists(self._image_path):
            return
        destino, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save image as", os.path.basename(self._image_path),
            "PNG (*.png);;All files (*)")
        if destino:
            shutil.copy2(self._image_path, destino)

    def _copy_to_clipboard(self):
        if self._pix is None:
            return
        QtWidgets.QApplication.clipboard().setPixmap(self._pix)


class Interactive3DPlot(QtWidgets.QDialog):
    """Dialog com grafico 3D matplotlib interativo (rotacao com mouse, zoom, pan).
    Recebe uma funcao de plotagem que desenha no eixo 3D fornecido.
    """

    def __init__(self, parent, title, plot_fn, figsize=(9, 7)):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 800)
        self._plot_fn = plot_fn

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._fig = plt.figure(figsize=figsize)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setFocusPolicy(Qt.ClickFocus)
        self._canvas.setFocus()
        lay.addWidget(self._canvas)

        # Toolbar com botoes de navegacao (rotacao, zoom, pan, home, salvar)
        toolbar = NavigationToolbar(self._canvas, self)
        lay.addWidget(toolbar)

        # Botao para alternar tela cheia
        btn_fs = QtWidgets.QPushButton("Fullscreen")
        btn_fs.clicked.connect(self._toggle_fullscreen)
        lay.addWidget(btn_fs)

        self._is_fullscreen = False
        self._prev_geometry = None

        # Desenha o grafico inicial
        self._draw()

    def _draw(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111, projection='3d')
        self._plot_fn(ax)
        self._fig.tight_layout()
        self._canvas.draw()

    def _toggle_fullscreen(self):
        if not self._is_fullscreen:
            self._prev_geometry = self.saveGeometry()
            self.showFullScreen()
            self._is_fullscreen = True
        else:
            self.showNormal()
            if self._prev_geometry:
                self.restoreGeometry(self._prev_geometry)
            self._is_fullscreen = False


class OutputRow(QtWidgets.QWidget):
    """Linha com os botoes de arquivo gerado: Salvar como / Open / XLSX."""

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.lbl = QtWidgets.QLabel("—")
        self.lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.b_save = QtWidgets.QPushButton("Save as…")
        self.b_open = QtWidgets.QPushButton("Open")
        self.b_xlsx = QtWidgets.QPushButton("Save XLSX…")
        self.b_xlsx.setVisible(False)
        lay.addWidget(self.lbl, 1)
        lay.addWidget(self.b_save)
        lay.addWidget(self.b_open)
        lay.addWidget(self.b_xlsx)
        self._path = None
        self.b_save.clicked.connect(lambda: self._salvar(self._path))
        self.b_open.clicked.connect(self._abrir)
        self.b_xlsx.clicked.connect(lambda: self._salvar(self._caminho_xlsx()))
        self.set_path(None)

    def _caminho_xlsx(self):
        if self._path and self._path.lower().endswith(".csv"):
            return self._path[:-4] + ".xlsx"
        return None

    def set_path(self, path):
        self._path = path
        existe = bool(path) and os.path.exists(path)
        self.lbl.setText(os.path.basename(path) if path else "—")
        self.b_save.setVisible(existe)
        self.b_open.setVisible(existe)
        xlsx = self._caminho_xlsx()
        self.b_xlsx.setVisible(bool(xlsx) and os.path.exists(xlsx))

    def _salvar(self, origem):
        if not origem or not os.path.exists(origem):
            return
        destino, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save file as", os.path.basename(origem))
        if not destino:
            return
        try:
            shutil.copyfile(origem, destino)
            QtWidgets.QMessageBox.information(self, "File saved",
                                              f"Salvo em:\n{destino}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def _abrir(self):
        if self._path and os.path.exists(self._path):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self._path))


class TableBox(QtWidgets.QWidget):
    """Tabela (somente leitura) + botao para exportar em CSV/XLSX."""

    def __init__(self, nome_export="tabela.csv"):
        super().__init__()
        self.nome_export = nome_export
        self._df = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        self.btn = QtWidgets.QPushButton("Export table (CSV/XLSX)")
        self.btn.clicked.connect(self._exportar)
        row.addWidget(self.btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.set_dataframe(None)

    def set_dataframe(self, df):
        self._df = df
        if df is None or len(df) == 0:
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.btn.setEnabled(False)
            return
        self.btn.setEnabled(True)
        self.table.clear()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                v = df.iloc[i][col]
                try:
                    s = f"{float(v):.6g}"
                except (TypeError, ValueError):
                    s = str(v)
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(s))
        self.table.resizeColumnsToContents()

    def _exportar(self):
        if self._df is None:
            return
        destino, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exportar tabela", self.nome_export, "CSV (*.csv)")
        if not destino:
            return
        try:
            save_df(self._df, destino)
            QtWidgets.QMessageBox.information(
                self, "Tabela exportada",
                f"CSV salvo em:\n{destino}\n(e o .xlsx irmao ao lado)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))


class UrlButton(QtWidgets.QPushButton):
    """Botao que abre uma pasta no gerenciador de arquivos."""

    def __init__(self, texto="Open output folder"):
        super().__init__(texto)
        self._pasta = None
        self.clicked.connect(self._abrir)

    def set_pasta(self, pasta):
        self._pasta = pasta

    def _abrir(self):
        pasta = self._pasta or APP_DIR
        os.makedirs(pasta, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(pasta))


class _PasteDelegate(QtWidgets.QStyledItemDelegate):
    """Delegate que desvia colagens de bloco do editor para a tabela.

    Sem isso, colar dentro de uma celula em edicao (duplo clique)
    despeja o texto inteiro numa celula so.
    """

    def __init__(self, tabela):
        super().__init__(tabela)
        self.tabela = tabela

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        if editor is not None:
            editor.installEventFilter(self)
        return editor

    def eventFilter(self, obj, ev):
        if (ev.type() == QtCore.QEvent.KeyPress
                and ev.matches(QtGui.QKeySequence.Paste)):
            texto = QtWidgets.QApplication.clipboard().text()
            separadores = PasteTableWidget.SEPARADORES
            if (any(s in texto for s in separadores)
                    or "\n" in texto or "\r" in texto):
                # colagem de bloco: descarta o conteudo do editor e
                # delega a colagem para a tabela
                try:
                    obj.clear()
                except AttributeError:
                    pass
                QtCore.QTimer.singleShot(
                    0, lambda: self.tabela.paste_text())
                return True
        return super().eventFilter(obj, ev)


class PasteTableWidget(QtWidgets.QTableWidget):
    """Tabela editavel com colagem no estilo planilha (Ctrl+V).

    Aceita blocos copiados do Excel, LibreOffice Calc ou de um CSV
    (colunas separadas por TAB, ';' ou '|'), com decimal por virgula ou
    ponto. A grade cresce para caber o bloco colado.
    """

    SEPARADORES = ("\t", ";", "|")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setItemDelegate(_PasteDelegate(self))

    # ---------- parsing da area de transferencia ----------
    @staticmethod
    def _limpar_campo(txt):
        txt = txt.strip()
        if len(txt) >= 2 and txt[0] == txt[-1] and txt[0] in "\"'":
            txt = txt[1:-1].strip()
        return txt.replace('""', '"')

    @classmethod
    def parse_clipboard(cls, texto):
        """Texto colado -> lista de linhas, cada uma com suas celulas."""
        txt = (texto or "").replace("\r\n", "\n").replace("\r", "\n")
        linhas = txt.split("\n")
        while linhas and not linhas[-1].strip():
            linhas.pop()
        if not linhas:
            return []
        sep = None
        for s in cls.SEPARADORES:
            if any(s in ln for ln in linhas):
                sep = s
                break
        if sep is None:
            # uma unica coluna (ou um unico valor)
            return [[cls._limpar_campo(ln)] for ln in linhas]
        return [[cls._limpar_campo(p) for p in ln.split(sep)]
                     for ln in linhas]

    # ---------- colagem ----------
    def paste_text(self, texto=None):
        """Cola o texto (padrao: area de transferencia) na celula atual."""
        if texto is None:
            texto = QtWidgets.QApplication.clipboard().text()
        blocos = self.parse_clipboard(texto)
        if not blocos:
            return False
        idx = self.currentIndex()
        r0 = idx.row() if idx.isValid() else 0
        c0 = idx.column() if idx.isValid() else 0
        r0, c0 = max(r0, 0), max(c0, 0)
        n_lin = len(blocos)
        n_col = min(max(len(b) for b in blocos), self.columnCount() - c0)
        if n_col <= 0:
            return False
        # fecha um editor aberto antes de escrever (o commit dele
        # sobrescreveria a colagem)
        self.setCurrentCell(r0, c0)
        # cresce a grade para caber o bloco, com uma linha de folga
        if r0 + n_lin + 1 > self.rowCount():
            self.setRowCount(r0 + n_lin + 1)
        for i, linha in enumerate(blocos):
            for j, valor in enumerate(linha[:n_col]):
                item = self.item(r0 + i, c0 + j)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    self.setItem(r0 + i, c0 + j, item)
                item.setText(valor)
        self.scrollToItem(self.item(r0 + n_lin - 1, c0))
        return True

    # ---------- eventos ----------
    def keyPressEvent(self, ev):
        if ev.matches(QtGui.QKeySequence.Paste):
            if self.paste_text():
                ev.accept()
                return
        super().keyPressEvent(ev)

    def contextMenuEvent(self, ev):
        menu = QtWidgets.QMenu(self)
        act_colar = menu.addAction("Paste (Ctrl+V)")
        act_inserir = menu.addAction("Insert row below")
        act_limpar = menu.addAction("Clear contents")
        escolha = menu.exec_(ev.globalPos())
        if escolha is act_colar:
            self.paste_text()
        elif escolha is act_inserir:
            linha = self.currentRow()
            self.insertRow(linha + 1 if linha >= 0 else self.rowCount())
        elif escolha is act_limpar:
            self.clearContents()


# ============================================================
#  ESTADO DA APLICACAO (equivalente ao st.session_state)
# ============================================================

class MetodoState:
    """Estado por metodo de kerma (hopfield / fourier)."""

    def __init__(self):
        self.df_tidy = None
        self.kermas = None
        self.ssds_grade = None
        self.N_val = 64
        self.kermas_csv = None
        self.kermas_png = None
        self.df_coef = None
        self.coef_csv = None
        self.coef_pngs = {}
        self.fits = {}
        self.fit_pngs = {}
        self.v2_modelo = None          # cached V2 model (built on demand)
        self.curvas_hist = []          # reconstrucao avancada
        self.curvas_hist_simples = []  # reconstrucao simples

    def reset_downstream(self):
        self.df_coef = None
        self.coef_csv = None
        self.coef_pngs = {}
        self.fits = {}
        self.fit_pngs = {}
        self.v2_modelo = None


class DirectState:
    def __init__(self):
        self.out_dir = os.path.join(APP_DIR, "output_direto")
        self.ssd1 = 100.0
        self.energia = 6.0
        self.campo_fixo = 10.0
        self.datasets = None
        self.params = None
        self.param_type = "campo"
        self.param_unit = ""
        self.fit_results = None
        self.fit_csv = None
        self.sup_a_png = None
        self.sup_b_png = None
        self.curvas_hist = []


class InverseState:
    def __init__(self):
        self.out_dir = os.path.join(APP_DIR, "output_indireto")
        self.ssd1 = 100.0
        self.energia = 6.0
        self.campo_fixo = 10.0
        self.datasets = None
        self.params = None
        self.param_type = "campo"
        self.param_unit = ""
        self.metodos = {"hopfield": MetodoState(), "fourier": MetodoState()}

    def metodo(self, nome):
        return self.metodos[nome]


class AppState:
    def __init__(self):
        self.direct = DirectState()
        self.inverse = InverseState()
        self.grade = None          # A/B surfaces compartilhadas


# ============================================================
#  ABA BASE
# ============================================================

class BaseTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.state = main.state
        self._worker = None
        self._dlg = None

    def run_task(self, label, fn, on_done=None):
        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Please wait",
                                              "A task is already running.")
            return
        self.main.statusBar().showMessage(label)
        dlg = QtWidgets.QProgressDialog(label, "", 0, 0, self.main)
        dlg.setWindowTitle("Processing…")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.show()
        self._dlg = dlg

        worker = TaskWorker(fn)

        def ok(res):
            dlg.close()
            self.main.statusBar().showMessage("Done.", 5000)
            if on_done is not None:
                try:
                    on_done(res)
                except Exception:
                    traceback.print_exc()
                    QtWidgets.QMessageBox.critical(self, "Interface error",
                                                   traceback.format_exc())

        def erro(msg):
            dlg.close()
            self.main.statusBar().showMessage("Failed.", 5000)
            QtWidgets.QMessageBox.critical(self, "Error", msg)

        worker.finished_ok.connect(ok)
        worker.failed.connect(erro)
        worker.finished.connect(lambda: self._finalizar_worker(worker))
        self._worker = worker
        worker.start()

    def _finalizar_worker(self, worker):
        if self._worker is worker:
            self._worker = None
        worker.deleteLater()

    def fixed_info(self, prefix):
        """Parametros fixos do feixe para anotar nos graficos."""
        st = getattr(self.state, "direct" if prefix == "d" else "inverse")
        info = {"energia": st.energia}
        if st.param_type == "ssd":
            info["campo"] = st.campo_fixo
        else:
            info["ssd"] = st.ssd1
        return info

    def _scroll_content(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(content)
        scroll.setWidget(content)
        return scroll, lay


# ============================================================
#  ABA DIRETO
# ============================================================

class DirectTab(BaseTab):
    def __init__(self, main):
        super().__init__(main)
        self.st = self.state.direct
        self._construido = False
        self._build()
        self._construido = True
        self._refresh_gates()

    # ---------- construcao ----------
    def _build(self):
        outer = QtWidgets.QVBoxLayout(self)
        scroll, root = self._scroll_content()
        outer.addWidget(scroll)

        # ---- 1. base ----
        self.grp_base = QtWidgets.QGroupBox("1. Configuration and base file")
        g = QtWidgets.QGridLayout(self.grp_base)
        g.addWidget(QtWidgets.QLabel("Output directory:"), 0, 0)
        self.ed_out = QtWidgets.QLineEdit(self.st.out_dir)
        b_out = QtWidgets.QPushButton("…")
        b_out.setFixedWidth(30)
        b_out.clicked.connect(self._escolher_dir)
        g.addWidget(self.ed_out, 0, 1)
        g.addWidget(b_out, 0, 2)
        g.addWidget(QtWidgets.QLabel("Reference SSD of base CSV (cm):"), 1, 0)
        self.sp_ssd1 = dspin(1.0, 1000.0, self.st.ssd1, 2, 1.0)
        g.addWidget(self.sp_ssd1, 1, 1)
        self.btn_carregar = QtWidgets.QPushButton("Load base CSV…")
        self.btn_carregar.clicked.connect(self._carregar_csv)
        g.addWidget(self.btn_carregar, 2, 0, 1, 3)
        self.lbl_base = QtWidgets.QLabel("No base loaded.")
        self.lbl_base.setWordWrap(True)
        g.addWidget(self.lbl_base, 3, 0, 1, 3)
        g.addWidget(QtWidgets.QLabel("Beam energy (MeV):"), 4, 0)
        self.sp_energia = dspin(0.0, 200.0, self.st.energia, 2, 1.0)
        g.addWidget(self.sp_energia, 4, 1)
        self.lbl_campo_fixo = QtWidgets.QLabel("Fixed field of this file:")
        self.sp_campo_fixo = dspin(0.0, 1000.0, self.st.campo_fixo, 2, 1.0)
        g.addWidget(self.lbl_campo_fixo, 5, 0)
        g.addWidget(self.sp_campo_fixo, 5, 1)
        root.addWidget(self.grp_base)

        # ---- 2. ajuste ----
        self.grp_fit = QtWidgets.QGroupBox("2. Fit (A, B) for each curve")
        lay = QtWidgets.QVBoxLayout(self.grp_fit)
        self.btn_fit = QtWidgets.QPushButton("Run fit")
        self.btn_fit.clicked.connect(self._executar_fit)
        lay.addWidget(self.btn_fit)
        self.tbl_fit = TableBox("fit_ab_por_curva.csv")
        lay.addWidget(self.tbl_fit)
        self.out_fit = OutputRow()
        lay.addWidget(self.out_fit)
        root.addWidget(self.grp_fit)

        # ---- 3. mayneord ----
        self.grp_may = QtWidgets.QGroupBox("3. View curve converted via Mayneord")
        lay = QtWidgets.QVBoxLayout(self.grp_may)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Curve from base file (field):"))
        self.cb_may = QtWidgets.QComboBox()
        row.addWidget(self.cb_may, 1)
        row.addWidget(QtWidgets.QLabel("Original SSD (cm):"))
        self.sp_may_ssd1 = dspin(1.0, 1000.0, self.st.ssd1, 2, 1.0)
        row.addWidget(self.sp_may_ssd1)
        row.addWidget(QtWidgets.QLabel("Convert to SSD (cm):"))
        self.sp_may_ssd2 = dspin(1.0, 1000.0, self.st.ssd1 + 10, 2, 1.0)
        row.addWidget(self.sp_may_ssd2)
        self.btn_may = QtWidgets.QPushButton("Generate curve plot")
        self.btn_may.clicked.connect(self._gerar_mayneord)
        row.addWidget(self.btn_may)
        lay.addLayout(row)
        self.img_may = ImageView()
        lay.addWidget(self.img_may)
        self.out_may = OutputRow()
        lay.addWidget(self.out_may)
        root.addWidget(self.grp_may)

        # ---- 4. superficies ----
        self.grp_sup = QtWidgets.QGroupBox("4. Build surfaces A(field,SSD) and B(field,SSD)")
        lay = QtWidgets.QVBoxLayout(self.grp_sup)
        self.chk_cache = QtWidgets.QCheckBox("Surface cache found — use?")
        self.chk_cache.setChecked(True)
        lay.addWidget(self.chk_cache)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Start SSD:"))
        self.sp_sup_start = dspin(1.0, 1000.0, self.st.ssd1, 2, 1.0)
        row.addWidget(self.sp_sup_start)
        row.addWidget(QtWidgets.QLabel("End SSD:"))
        self.sp_sup_end = dspin(1.0, 1000.0, 354.0, 2, 1.0)
        row.addWidget(self.sp_sup_end)
        row.addWidget(QtWidgets.QLabel("Number of SSDs:"))
        self.sp_sup_n = ispin(2, 500, 15)
        row.addWidget(self.sp_sup_n)
        lay.addLayout(row)
        self.chk_d0 = QtWidgets.QCheckBox("Use fixed d0 for all fields?")
        self.chk_d0.toggled.connect(lambda on: self.sp_d0.setEnabled(on))
        lay.addWidget(self.chk_d0)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("d0 value (cm):"))
        self.sp_d0 = dspin(0.0, 100.0, 0.0, 3, 0.5)
        self.sp_d0.setEnabled(False)
        row2.addWidget(self.sp_d0)
        row2.addStretch(1)
        lay.addLayout(row2)
        self.btn_sup = QtWidgets.QPushButton("Build / load surfaces")
        self.btn_sup.clicked.connect(self._montar_superficies)
        lay.addWidget(self.btn_sup)
        row3 = QtWidgets.QHBoxLayout()
        self.img_sup_a = ImageView()
        self.img_sup_b = ImageView()
        row3.addWidget(self.img_sup_a)
        row3.addWidget(self.img_sup_b)
        # botoes 3D interativo
        btn_sup_a_3d = QtWidgets.QPushButton("View Interactive 3D (A)")
        btn_sup_a_3d.clicked.connect(lambda: self._abrir_3d_interativo_superficie('A'))
        btn_sup_b_3d = QtWidgets.QPushButton("View Interactive 3D (B)")
        btn_sup_b_3d.clicked.connect(lambda: self._abrir_3d_interativo_superficie('B'))
        lay.addWidget(btn_sup_a_3d)
        lay.addWidget(btn_sup_b_3d)
        lay.addLayout(row3)
        root.addWidget(self.grp_sup)

        # ---- 5. point reconstruction ----
        self.grp_rec = QtWidgets.QGroupBox("5. Reconstruct PDD(x) for a specific (field, SSD)")
        lay = QtWidgets.QVBoxLayout(self.grp_rec)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Depth grid points:"))
        self.sp_rec_n = ispin(10, 100000, 300, 10)
        row.addWidget(self.sp_rec_n)
        row.addWidget(QtWidgets.QLabel("Max depth (cm):"))
        self.sp_rec_xmax = dspin(1.0, 200.0, 30.0, 2, 1.0)
        row.addWidget(self.sp_rec_xmax)
        lay.addLayout(row)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Desired field:"))
        self.sp_rec_campo = dspin(0.0, 1000.0, 10.0, 2, 0.5)
        row2.addWidget(self.sp_rec_campo)
        row2.addWidget(QtWidgets.QLabel("Desired SSD (cm):"))
        self.sp_rec_ssd = dspin(1.0, 1000.0, self.st.ssd1, 2, 1.0)
        row2.addWidget(self.sp_rec_ssd)
        self.btn_rec = QtWidgets.QPushButton("Generate curve")
        self.btn_rec.clicked.connect(self._reconstruir)
        row2.addWidget(self.btn_rec)
        lay.addLayout(row2)
        self.hist_rec = QtWidgets.QVBoxLayout()
        lay.addLayout(self.hist_rec)
        root.addWidget(self.grp_rec)

        # ---- 6. 3D surface ----
        self.grp_3d = QtWidgets.QGroupBox("6. PDD 3D Surface (variable field or variable SSD)")
        lay = QtWidgets.QVBoxLayout(self.grp_3d)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Depth grid points:"))
        self.sp_3d_n = ispin(10, 100000, 300, 10)
        row.addWidget(self.sp_3d_n)
        row.addWidget(QtWidgets.QLabel("Max depth (cm):"))
        self.sp_3d_xmax = dspin(1.0, 200.0, 30.0, 2, 1.0)
        row.addWidget(self.sp_3d_xmax)
        lay.addLayout(row)
        self.cb_3d_modo = QtWidgets.QComboBox()
        self.cb_3d_modo.addItem("Fixed SSD, variable field", "A")
        self.cb_3d_modo.addItem("Fixed field, variable SSD", "B")
        self.cb_3d_modo.currentIndexChanged.connect(self._atualizar_3d_modo)
        lay.addWidget(self.cb_3d_modo)
        row2 = QtWidgets.QHBoxLayout()
        self.lbl_3d_fixo = QtWidgets.QLabel("Fixed SSD (cm):")
        row2.addWidget(self.lbl_3d_fixo)
        self.sp_3d_fixo = dspin(1.0, 1000.0, self.st.ssd1, 2, 1.0)
        row2.addWidget(self.sp_3d_fixo)
        self.lbl_3d_nvar = QtWidgets.QLabel("Number of fields:")
        row2.addWidget(self.lbl_3d_nvar)
        self.sp_3d_nvar = ispin(3, 100000, 20)
        row2.addWidget(self.sp_3d_nvar)
        self.btn_3d = QtWidgets.QPushButton("Generate 3D surface")
        self.btn_3d.clicked.connect(self._gerar_3d)
        row2.addWidget(self.btn_3d)
        lay.addLayout(row2)
        row3 = QtWidgets.QHBoxLayout()
        self.img_3d_a = ImageView()
        self.img_3d_b = ImageView()
        row3.addWidget(self.img_3d_a)
        row3.addWidget(self.img_3d_b)
        # botao 3D interativo
        btn_3d_int = QtWidgets.QPushButton("View Interactive 3D")
        btn_3d_int.clicked.connect(self._abrir_3d_interativo_pdd)
        lay.addWidget(btn_3d_int)
        lay.addLayout(row3)
        root.addWidget(self.grp_3d)

        self.btn_pasta = UrlButton()
        root.addWidget(self.btn_pasta)
        root.addStretch(1)
        self._construido = True

    # ---------- util ----------
    def _sync_state(self):
        self.st.out_dir = self.ed_out.text().strip() or self.st.out_dir
        self.st.ssd1 = float(self.sp_ssd1.value())
        self.st.energia = float(self.sp_energia.value())
        self.st.campo_fixo = float(self.sp_campo_fixo.value())

    def _escolher_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output directory",
                                                        self.ed_out.text())
        if d:
            self.ed_out.setText(d)
            self.btn_pasta.set_pasta(d)

    def _refresh_gates(self):
        base = self.st.datasets is not None
        grade = self.state.grade is not None
        self.grp_fit.setEnabled(base)
        self.grp_may.setEnabled(base)
        self.grp_sup.setEnabled(base)
        self.grp_rec.setEnabled(base and grade)
        self.grp_3d.setEnabled(base and grade)
        self.chk_cache.setVisible(cache_existe(self.st.out_dir))
        self.btn_pasta.set_pasta(self.st.out_dir)
        # campos fixos dependem do tipo de parametro
        eh_ssd = self.st.param_type == "ssd"
        self.lbl_campo_fixo.setVisible(eh_ssd)
        self.sp_campo_fixo.setVisible(eh_ssd)
        # 3d defaults
        self._atualizar_3d_modo()

    def _atualizar_3d_modo(self):
        if not getattr(self, "_construido", False):
            return
        modo = self.cb_3d_modo.currentData()
        grade = self.state.grade
        if modo == "A":
            self.lbl_3d_fixo.setText("Fixed SSD (cm):")
            self.lbl_3d_nvar.setText("Number of fields:")
        else:
            self.lbl_3d_fixo.setText("Fixed field:")
            self.lbl_3d_nvar.setText("Number of SSDs:")
        self.sp_3d_nvar.setValue(20)

    # ---------- 1. carga ----------
    def _carregar_csv(self):
        src, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose base CSV", DATA_DIR, "CSV (*.csv)")
        if not src:
            return
        out_dir = self.ed_out.text().strip() or self.st.out_dir
        ssd1 = float(self.sp_ssd1.value())
        try:
            os.makedirs(out_dir, exist_ok=True)
            destino = os.path.join(out_dir, "base_" + os.path.basename(src))
            shutil.copyfile(src, destino)
            datasets, params, param_type, param_unit = load_csv(destino)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error loading CSV", str(e))
            return
        self.st.out_dir = out_dir
        self.st.ssd1 = ssd1
        self.st.datasets = datasets
        self.st.params = params
        self.st.param_type = param_type
        self.st.param_unit = param_unit
        self.st.fit_results = None
        self.st.curvas_hist = []
        self.lbl_base.setText(
            f"Base loaded: {len(datasets)} curves · parameter: {param_type} "
            f"({param_unit}) · file: {os.path.basename(destino)}")
        self.cb_may.clear()
        self.cb_may.addItems(list(datasets.keys()))
        self.sp_rec_ssd.setValue(ssd1)
        self.sp_may_ssd1.setValue(ssd1)
        self.sp_may_ssd2.setValue(ssd1 + 10)
        self.sp_3d_fixo.setValue(ssd1)
        self.tbl_fit.set_dataframe(None)
        self.out_fit.set_path(None)
        self._refresh_gates()

    # ---------- 2. fit ----------
    def _executar_fit(self):
        self._sync_state()
        datasets = self.st.datasets
        out_dir = self.st.out_dir
        param_type = self.st.param_type
        params = self.st.params

        def tarefa():
            fit_results = fit_all_curves(datasets)
            df_fit = pd.DataFrame([
                {"curva": nome, param_type: params[nome], "a": r["a"], "b": r["b"]}
                for nome, r in fit_results.items()
            ])
            caminho = os.path.join(subdir(out_dir, "ajuste"), "fit_ab_por_curva.csv")
            save_df(df_fit, caminho)
            return fit_results, df_fit, caminho

        def pronto(res):
            fit_results, df_fit, caminho = res
            self.st.fit_results = fit_results
            self.st.fit_csv = caminho
            self.tbl_fit.set_dataframe(df_fit)
            self.out_fit.set_path(caminho)

        self.run_task("Fitting curves (A,B)…", tarefa, pronto)

    # ---------- 3. mayneord ----------
    def _gerar_mayneord(self):
        self._sync_state()
        if self.cb_may.count() == 0:
            return
        curva = self.cb_may.currentText()
        ssd_leg = float(self.sp_may_ssd1.value())
        ssd_alvo = float(self.sp_may_ssd2.value())
        datasets = self.st.datasets
        out_dir = self.st.out_dir
        energia = self.st.energia
        from common.mayneord import apply_mayneord

        def tarefa():
            xm, ym = datasets[curva]
            d0 = detect_dmax_xy(xm, ym)
            ym2 = ym if abs(ssd_alvo - ssd_leg) < 1e-9 else apply_mayneord(
                ym, xm, ssd_leg, ssd_alvo, d0)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(xm, ym, label=f"{curva} — SSD {ssd_leg:g} cm")
            ax.plot(xm, ym2, label=f"{curva} — SSD {ssd_alvo:g} cm")
            ax.set_xlabel("Depth (cm)")
            ax.set_ylabel("PDD (%)")
            ax.set_title(f"Curve {curva}: original vs. converted (Mayneord)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            if energia is not None:
                fig.text(0.99, 0.01, f"Energy = {energia:g} MeV", ha='right',
                         va='bottom', fontsize=8, style='italic', color='dimgray')
            fig.tight_layout()
            stem = str(curva).replace("/", "-").replace(" ", "_")
            pasta = subdir(out_dir, "mayneord")
            png = os.path.join(pasta, f"mayneord_{stem}_ssd{ssd_alvo:g}.png")
            fig.savefig(png, dpi=200, bbox_inches="tight")
            plt.close(fig)
            df_vis = pd.DataFrame({
                "profundidade_cm": xm,
                f"PDD_original_SSD_{ssd_leg:g}": ym,
                f"PDD_convertida_SSD_{ssd_alvo:g}": ym2,
            })
            csv = os.path.join(pasta, f"mayneord_{stem}_ssd{ssd_alvo:g}.csv")
            save_df(df_vis, csv)
            return png, csv

        def pronto(res):
            png, csv = res
            self.img_may.set_image(png)
            self.out_may.set_path(csv)

        self.run_task("Generating Mayneord curve…", tarefa, pronto)

    # ---------- 4. superficies ----------
    def _montar_superficies(self):
        self._sync_state()
        out_dir = self.st.out_dir
        usar_cache = self.chk_cache.isChecked() and cache_existe(out_dir)
        datasets = self.st.datasets
        params = self.st.params
        param_type = self.st.param_type
        ssd1 = self.st.ssd1
        fixed_d0 = float(self.sp_d0.value()) if self.chk_d0.isChecked() else None
        start = float(self.sp_sup_start.value())
        end = float(self.sp_sup_end.value())
        n = int(self.sp_sup_n.value())
        fixed_info = self.fixed_info("d")

        def tarefa():
            dir_sup = subdir(out_dir, "superficies")
            if usar_cache:
                grade = carregar_ou_construir_interpoladores(dir_sup)
            else:
                ssds = np.linspace(start, end, n)
                grade = carregar_ou_construir_interpoladores(
                    dir_sup, datasets=datasets, params=params,
                    param_type=param_type, ssd1=ssd1, ssds_grade=ssds,
                    fixed_d0=fixed_d0, forcar_rebuild=True)
            pa = os.path.join(dir_sup, "superficie_A.png")
            pb = os.path.join(dir_sup, "superficie_B.png")
            plotar_superficie_ab(grade["superficie_a"], "A Surface", "A", pa,
                                 fixed_info=fixed_info)
            plotar_superficie_ab(grade["superficie_b"], "B Surface", "B", pb,
                                 fixed_info=fixed_info)
            return grade, pa, pb

        def pronto(res):
            grade, pa, pb = res
            self.state.grade = grade
            self.st.sup_a_png = pa
            self.st.sup_b_png = pb
            self.img_sup_a.set_image(pa)
            self.img_sup_b.set_image(pb)
            self._refresh_gates()
            self.main.comparison_tab.refresh_instances()

        self.run_task("Building A/B surfaces…", tarefa, pronto)

    # ---------- 5. reconstrucao ----------
    def _reconstruir(self):
        self._sync_state()
        grade = self.state.grade
        if grade is None:
            return
        n_pts = int(self.sp_rec_n.value())
        x_max = float(self.sp_rec_xmax.value())
        campo = float(self.sp_rec_campo.value())
        ssd = float(self.sp_rec_ssd.value())
        out_dir = self.st.out_dir
        datasets = self.st.datasets
        ssd1 = self.st.ssd1
        param_type = self.st.param_type
        energia = self.st.energia

        def tarefa():
            x_curva, pdd_conv, info = curva_pdd_interpolada(
                grade, campo, ssd, n_pts, x_max)
            ref = referencia_mais_proxima(grade, campo, ssd)
            stem = f"pdd_c{campo}_ssd{ssd}".replace(".", "p")
            pasta = subdir(out_dir, "curvas")
            csv = os.path.join(pasta, stem + ".csv")
            png = os.path.join(pasta, stem + ".png")
            save_df(pd.DataFrame({"profundidade_cm": x_curva, "pdd_conv_%": pdd_conv}), csv)
            plotar_curva(x_curva, pdd_conv, info, ref, datasets, ssd1, param_type,
                         png, fixed_info={"energia": energia})
            item = {"info": dict(info), "png": png, "csv": csv, "campo": campo,
                    "ssd": ssd, "x_curva": np.asarray(x_curva),
                    "pdd_conv": np.asarray(pdd_conv)}
            return item

        def pronto(item):
            self.st.curvas_hist.append(item)
            self._rebuild_hist(self.hist_rec, self.st.curvas_hist,
                               tipo="direto")
            self.main.comparison_tab.refresh_instances()

        self.run_task("Reconstructing PDD curve…", tarefa, pronto)

    def _rebuild_hist(self, layout, historico, tipo):
        """Reconstroi a lista de resultados (mais recente primeiro)."""
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for h in reversed(historico):
            box = QtWidgets.QGroupBox()
            v = QtWidgets.QVBoxLayout(box)
            info = h["info"]
            if tipo == "direto":
                txt = (f"Field {fmt_num(h['campo'])} · SSD {fmt_num(h['ssd'])} cm — "
                       f"a={info['a']:.6f}  b={info['b']:.6f}  d0={info['d0']:.2f} cm")
            else:
                txt = (f"Field {fmt_num(h['campo'])} · SSD {fmt_num(h['ssd'])} cm — "
                       f"b={info['b']:.6f}  d0={info['d0']:.2f} cm")
            v.addWidget(titulo(txt))
            img = ImageView()
            img.set_image(h["png"])
            v.addWidget(img)
            row = OutputRow()
            row.set_path(h["csv"])
            v.addWidget(row)
            layout.addWidget(box)

    # ---------- 6. 3D surface ----------
    def _gerar_3d(self):
        self._sync_state()
        grade = self.state.grade
        if grade is None:
            return
        n_pts = int(self.sp_3d_n.value())
        x_max = float(self.sp_3d_xmax.value())
        modo = self.cb_3d_modo.currentData()
        valor_fixo = float(self.sp_3d_fixo.value())
        n_var = int(self.sp_3d_nvar.value())
        out_dir = self.st.out_dir
        fixed_info = self.fixed_info("d")

        def tarefa():
            campos_grade = grade["campos_grade"]
            pasta = subdir(out_dir, "superficie_3d")
            if modo == "A":
                valores = np.linspace(campos_grade.min(), campos_grade.max(), n_var)
                x_prof, Y, Z = gerar_superficie_pdd(grade, "A", valor_fixo, valores,
                                                    n_pts, x_max)
                stem = f"sup3d_modoA_ssd{valor_fixo}".replace(".", "p")
            else:
                ssds_grade = grade["ssds_grade"]
                valores = np.linspace(ssds_grade.min(), ssds_grade.max(), n_var)
                x_prof, Y, Z = gerar_superficie_pdd(grade, "B", valor_fixo, valores,
                                                    n_pts, x_max)
                stem = f"sup3d_modoB_campo{valor_fixo}".replace(".", "p")
            png3d = os.path.join(pasta, stem + "_3d.png")
            png2d = os.path.join(pasta, stem + "_2d.png")
            plotar_superficie_pdd(x_prof, Y, Z, modo, valor_fixo, png3d,
                                  fixed_info=fixed_info)
            plotar_familia_curvas(x_prof, Y, Z, modo, valor_fixo, png2d,
                                  fixed_info=fixed_info)
            return png3d, png2d

        def pronto(res):
            self.img_3d_a.set_image(res[0])
            self.img_3d_b.set_image(res[1])

        self.run_task("Generating 3D surface…", tarefa, pronto)

    # ---------- 3D Interativo ----------
    def _abrir_3d_interativo_superficie(self, qual):
        """Abre dialog interativo para superficie A ou B."""
        grade = self.state.grade
        if grade is None:
            return
        fixed_info = self.fixed_info("d")
        titulo = f"Surface {qual} (Field × SSD)"
        def plot_fn(ax):
            matriz = grade[f"superficie_{qual.lower()}"]
            campos = matriz.index.values.astype(float)
            ssds = matriz.columns.values.astype(float)
            X, Y = np.meshgrid(ssds, campos)
            Z = matriz.values
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k',
                                    linewidth=0.2, antialiased=True)
            ax.set_xlabel('SSD (cm)')
            ax.set_ylabel('Field')
            ax.set_zlabel(qual)
            ax.set_title(titulo)
            ax.view_init(elev=30, azim=-60)
        dlg = Interactive3DPlot(self, titulo, plot_fn)
        dlg.exec_()

    def _abrir_3d_interativo_pdd(self):
        """Abre dialog interativo para 3D surface de PDD."""
        grade = self.state.grade
        if grade is None:
            return
        modo = self.cb_3d_modo.currentData()
        valor_fixo = float(self.sp_3d_fixo.value())
        n_var = int(self.sp_3d_nvar.value())
        n_pts = int(self.sp_3d_n.value())
        x_max = float(self.sp_3d_xmax.value())
        fixed_info = self.fixed_info("d")
        
        if modo == "A":
            valores = np.linspace(grade["campos_grade"].min(), grade["campos_grade"].max(), n_var)
            x_prof, Y, Z = gerar_superficie_pdd(grade, "A", valor_fixo, valores, n_pts, x_max)
            ylabel = 'Field'
            titulo = f"PDD(depth, field) | SSD = {valor_fixo:.1f} cm"
        else:
            valores = np.linspace(grade["ssds_grade"].min(), grade["ssds_grade"].max(), n_var)
            x_prof, Y, Z = gerar_superficie_pdd(grade, "B", valor_fixo, valores, n_pts, x_max)
            ylabel = 'SSD (cm)'
            titulo = f"PDD(depth, SSD) | Field = {valor_fixo:.2f}"
        
        def plot_fn(ax):
            Xm, Ym = np.meshgrid(x_prof, Y, indexing='ij')
            surf = ax.plot_surface(Xm, Ym, Z, cmap='viridis',
                                    edgecolor='k', linewidth=0.15, antialiased=True, alpha=0.92)
            z_floor = Z.min() - 5
            ax.contourf(Xm, Ym, Z, zdir='z', offset=z_floor, cmap='viridis', alpha=0.25, levels=15)
            ax.set_xlabel('Depth (cm)', fontsize=11, labelpad=8)
            ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
            ax.set_zlabel('PDD (%)', fontsize=11, labelpad=8)
            ax.set_title(titulo, fontsize=13, pad=14)
            ax.set_zlim(z_floor, 105)
            ax.view_init(elev=30, azim=-60)
        
        dlg = Interactive3DPlot(self, titulo, plot_fn, figsize=(11, 7))
        dlg.exec_()


# ============================================================
#  ABA INVERSO
# ============================================================

class InverseTab(BaseTab):
    COLS_COEF_PNG = ["A_kerma", "tau_kerma", "beta_kerma"]
    NOMES_FIT = ["A", "tau", "beta"]
    # rotulo visual de cada coeficiente (as chaves de NOMES_FIT/fits sao
    # os nomes usados internamente; aqui so trocamos tau->τ e beta->β)
    DISPLAY_FIT = {"A": "A", "tau": "τ", "beta": "β"}

    def __init__(self, main):
        super().__init__(main)
        self.st = self.state.inverse
        self._construido = False
        self._build()
        self._construido = True
        self._refresh_gates()
        self._refresh_methodo_view()

    # ---------- util estado ----------
    def _metodo(self):
        return self.cb_metodo.currentData()

    def _ms(self, metodo=None):
        return self.st.metodo(metodo or self._metodo())

    # ---------- construcao ----------
    def _build(self):
        outer = QtWidgets.QVBoxLayout(self)
        scroll, root = self._scroll_content()
        outer.addWidget(scroll)

        # ---- 1. base ----
        self.grp_base = QtWidgets.QGroupBox("1. Configuration and base file")
        g = QtWidgets.QGridLayout(self.grp_base)
        g.addWidget(QtWidgets.QLabel("Output directory:"), 0, 0)
        self.ed_out = QtWidgets.QLineEdit(self.st.out_dir)
        b_out = QtWidgets.QPushButton("…")
        b_out.setFixedWidth(30)
        b_out.clicked.connect(self._escolher_dir)
        g.addWidget(self.ed_out, 0, 1)
        g.addWidget(b_out, 0, 2)
        g.addWidget(QtWidgets.QLabel("Reference SSD of base CSV (cm):"), 1, 0)
        self.sp_ssd1 = dspin(1.0, 1000.0, self.st.ssd1, 2, 1.0)
        g.addWidget(self.sp_ssd1, 1, 1)
        self.btn_carregar = QtWidgets.QPushButton("Load base CSV…")
        self.btn_carregar.clicked.connect(self._carregar_csv)
        g.addWidget(self.btn_carregar, 2, 0, 1, 3)
        self.lbl_base = QtWidgets.QLabel("No base loaded.")
        self.lbl_base.setWordWrap(True)
        g.addWidget(self.lbl_base, 3, 0, 1, 3)
        g.addWidget(QtWidgets.QLabel("Beam energy (MeV):"), 4, 0)
        self.sp_energia = dspin(0.0, 200.0, self.st.energia, 2, 1.0)
        g.addWidget(self.sp_energia, 4, 1)
        self.lbl_campo_fixo = QtWidgets.QLabel("Fixed field of this file:")
        self.sp_campo_fixo = dspin(0.0, 1000.0, self.st.campo_fixo, 2, 1.0)
        g.addWidget(self.lbl_campo_fixo, 5, 0)
        g.addWidget(self.sp_campo_fixo, 5, 1)
        root.addWidget(self.grp_base)

        # ---- 1b. surfaces ----
        self.grp_sup = QtWidgets.QGroupBox(
            "1b. Build surfaces A(field,SSD) and B(field,SSD) — optional")
        lay = QtWidgets.QVBoxLayout(self.grp_sup)
        self.lbl_sup_status = QtWidgets.QLabel("")
        self.lbl_sup_status.setWordWrap(True)
        lay.addWidget(self.lbl_sup_status)
        self.chk_cache = QtWidgets.QCheckBox("Surface cache found — use?")
        self.chk_cache.setChecked(True)
        lay.addWidget(self.chk_cache)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Start SSD:"))
        self.sp_sup_start = dspin(1.0, 1000.0, self.st.ssd1, 2, 1.0)
        row.addWidget(self.sp_sup_start)
        row.addWidget(QtWidgets.QLabel("End SSD:"))
        self.sp_sup_end = dspin(1.0, 1000.0, 354.0, 2, 1.0)
        row.addWidget(self.sp_sup_end)
        row.addWidget(QtWidgets.QLabel("Number of SSDs:"))
        self.sp_sup_n = ispin(2, 500, 15)
        row.addWidget(self.sp_sup_n)
        lay.addLayout(row)
        self.chk_d0 = QtWidgets.QCheckBox("Use fixed d0 for all fields?")
        self.chk_d0.toggled.connect(lambda on: self.sp_d0.setEnabled(on))
        lay.addWidget(self.chk_d0)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("d0 value (cm):"))
        self.sp_d0 = dspin(0.0, 100.0, 0.0, 3, 0.5)
        self.sp_d0.setEnabled(False)
        row2.addWidget(self.sp_d0)
        row2.addStretch(1)
        lay.addLayout(row2)
        self.btn_sup = QtWidgets.QPushButton("Build / load surfaces")
        self.btn_sup.clicked.connect(self._montar_superficies)
        lay.addWidget(self.btn_sup)
        row3 = QtWidgets.QHBoxLayout()
        self.img_sup_a = ImageView()
        self.img_sup_b = ImageView()
        row3.addWidget(self.img_sup_a)
        row3.addWidget(self.img_sup_b)
        lay.addLayout(row3)
        root.addWidget(self.grp_sup)

        # ---- 2. kermas ----
        self.grp_kerma = QtWidgets.QGroupBox(
            "2. Calculate kermas F(y) (Hopfield / Analytical-Fourier)")
        lay = QtWidgets.QVBoxLayout(self.grp_kerma)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Method:"))
        self.cb_metodo = QtWidgets.QComboBox()
        self.cb_metodo.addItem("Hopfield (neural network)", "hopfield")
        self.cb_metodo.addItem("Analytical (Fourier)", "fourier")
        self.cb_metodo.currentIndexChanged.connect(self._refresh_methodo_view)
        row.addWidget(self.cb_metodo)
        row.addWidget(QtWidgets.QLabel("N (neurons/points):"))
        self.sp_N = ispin(4, 100000, 64)
        row.addWidget(self.sp_N)
        row.addWidget(QtWidgets.QLabel("t_end (Hopfield):"))
        self.sp_tend = dspin(0.0, 1e6, 100.0, 2, 10.0)
        row.addWidget(self.sp_tend)
        lay.addLayout(row)
        self.lbl_fourier = QtWidgets.QLabel(
            "Deconvolution in the frequency domain (FFT). Wiener/Tikhonov "
            "regularization reduces noise amplification.")
        self.lbl_fourier.setWordWrap(True)
        lay.addWidget(self.lbl_fourier)
        row2 = QtWidgets.QHBoxLayout()
        self.chk_fourier_reg = QtWidgets.QCheckBox("Apply regularization (recommended)")
        self.chk_fourier_reg.setChecked(True)
        self.chk_fourier_reg.toggled.connect(lambda on: self.sp_fourier_lam.setEnabled(on))
        row2.addWidget(self.chk_fourier_reg)
        row2.addWidget(QtWidgets.QLabel("λ:"))
        self.sp_fourier_lam = dspin(0.0, 1.0, 0.05, 5, 0.01)
        row2.addWidget(self.sp_fourier_lam)
        row2.addStretch(1)
        lay.addLayout(row2)
        row3 = QtWidgets.QHBoxLayout()
        row3.addWidget(QtWidgets.QLabel("SSD grid — start:"))
        self.sp_g_start = dspin(1.0, 1000.0, self.st.ssd1, 2, 1.0)
        row3.addWidget(self.sp_g_start)
        row3.addWidget(QtWidgets.QLabel("end:"))
        self.sp_g_end = dspin(1.0, 1000.0, 354.0, 2, 1.0)
        row3.addWidget(self.sp_g_end)
        row3.addWidget(QtWidgets.QLabel("Count:"))
        self.sp_g_n = ispin(2, 500, 15)
        row3.addWidget(self.sp_g_n)
        lay.addLayout(row3)
        self.chk_d0k = QtWidgets.QCheckBox("Use fixed d0 for all fields?")
        self.chk_d0k.toggled.connect(lambda on: self.sp_d0k.setEnabled(on))
        lay.addWidget(self.chk_d0k)
        row4 = QtWidgets.QHBoxLayout()
        row4.addWidget(QtWidgets.QLabel("d0 value (cm):"))
        self.sp_d0k = dspin(0.0, 100.0, 0.0, 3, 0.5)
        self.sp_d0k.setEnabled(False)
        row4.addWidget(self.sp_d0k)
        row4.addStretch(1)
        lay.addLayout(row4)
        self.btn_kerma = QtWidgets.QPushButton("Run kerma calculation")
        self.btn_kerma.clicked.connect(self._calcular_kermas)
        lay.addWidget(self.btn_kerma)
        self.tbl_kerma = TableBox("kermas_tidy.csv")
        lay.addWidget(self.tbl_kerma)
        self.out_kerma = OutputRow()
        lay.addWidget(self.out_kerma)
        self.img_kerma = ImageView()
        lay.addWidget(self.img_kerma)
        self.lbl_kerma_outro = QtWidgets.QLabel("")
        self.lbl_kerma_outro.setWordWrap(True)
        lay.addWidget(self.lbl_kerma_outro)
        root.addWidget(self.grp_kerma)

        # ---- 3. coefficients ----
        self.grp_coef = QtWidgets.QGroupBox(
            "3. Analyze kerma F(y) coefficients (A, τ, β)")
        lay = QtWidgets.QVBoxLayout(self.grp_coef)
        self.btn_coef = QtWidgets.QPushButton("Calculate coefficients")
        self.btn_coef.clicked.connect(self._calcular_coef)
        lay.addWidget(self.btn_coef)
        self.tbl_coef = TableBox("coefficients_kerma.csv")
        lay.addWidget(self.tbl_coef)
        self.out_coef = OutputRow()
        lay.addWidget(self.out_coef)
        self.imgs_coef = {}
        row = QtWidgets.QHBoxLayout()
        for col in self.COLS_COEF_PNG:
            img = ImageView()
            self.imgs_coef[col] = img
            row.addWidget(img)
        lay.addLayout(row)
        # botoes 3D interativo para coefficients
        row3d = QtWidgets.QHBoxLayout()
        for col in self.COLS_COEF_PNG:
            btn = QtWidgets.QPushButton(f"View Interactive 3D ({self.DISPLAY_FIT[col.replace('_kerma', '')]})")
            btn.clicked.connect(lambda checked, c=col: self._abrir_3d_interativo_coef(c))
            row3d.addWidget(btn)
        lay.addLayout(row3d)
        root.addWidget(self.grp_coef)

        # ---- 4. modelo ----
        self.grp_modelo = QtWidgets.QGroupBox(
            "4. Generalized model — stretched exponential kerma F(y)=A·exp(−(y/τ)^β)")
        lay = QtWidgets.QVBoxLayout(self.grp_modelo)
        lbl_form = QtWidgets.QLabel(FORMULA_KERMA)
        lbl_form.setWordWrap(True)
        lay.addWidget(lbl_form)
        lbl_c = QtWidgets.QLabel(FORMULA_CONSTANTES)
        lbl_c.setWordWrap(True)
        lay.addWidget(lbl_c)
        self.btn_modelo = QtWidgets.QPushButton("Fit model (A, τ, β)")
        self.btn_modelo.clicked.connect(self._ajustar_modelo)
        lay.addWidget(self.btn_modelo)
        self.fits_ui = {}
        grid = QtWidgets.QGridLayout()
        for i, nome in enumerate(self.NOMES_FIT):
            box = QtWidgets.QGroupBox(self.DISPLAY_FIT.get(nome, nome))
            v = QtWidgets.QVBoxLayout(box)
            lbl_n = QtWidgets.QLabel("—")
            lbl_n.setWordWrap(True)
            lbl_n.setTextInteractionFlags(Qt.TextSelectableByMouse)
            v.addWidget(lbl_n)
            lbl_r = QtWidgets.QLabel("")
            v.addWidget(lbl_r)
            lbl_a = QtWidgets.QLabel("")
            lbl_a.setWordWrap(True)
            lbl_a.setStyleSheet("color: #a05a00;")
            v.addWidget(lbl_a)
            img = ImageView()
            v.addWidget(img)
            self.fits_ui[nome] = (lbl_n, lbl_r, lbl_a, img)
            grid.addWidget(box, i // 3, i % 3)
        lay.addLayout(grid)
        # botoes 3D interativo para modelo ajustado
        row3d = QtWidgets.QHBoxLayout()
        for nome in self.NOMES_FIT:
            btn = QtWidgets.QPushButton(f"View Interactive 3D ({self.DISPLAY_FIT[nome]})")
            btn.clicked.connect(lambda checked, n=nome: self._abrir_3d_interativo_modelo(n))
            row3d.addWidget(btn)
        lay.addLayout(row3d)
        root.addWidget(self.grp_modelo)

        # ---- 5. reconstrucao avancada ----
        self.grp_rec = QtWidgets.QGroupBox(
            "5. ADVANCED Reconstruction — PDD(x) for any (field, SSD)")
        lay = QtWidgets.QVBoxLayout(self.grp_rec)
        self.lbl_rec_status = QtWidgets.QLabel("")
        self.lbl_rec_status.setWordWrap(True)
        lay.addWidget(self.lbl_rec_status)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Grid points:"))
        self.sp_rec_n = ispin(10, 100000, 300, 10)
        row.addWidget(self.sp_rec_n)
        row.addWidget(QtWidgets.QLabel("Max depth (cm):"))
        self.sp_rec_xmax = dspin(1.0, 200.0, 30.0, 2, 1.0)
        row.addWidget(self.sp_rec_xmax)
        row.addWidget(QtWidgets.QLabel("Desired field:"))
        self.sp_rec_campo = dspin(0.0, 1000.0, 10.0, 2, 0.5)
        row.addWidget(self.sp_rec_campo)
        row.addWidget(QtWidgets.QLabel("Desired SSD (cm):"))
        self.sp_rec_ssd = dspin(1.0, 1000.0, self.st.ssd1, 2, 1.0)
        row.addWidget(self.sp_rec_ssd)
        row.addWidget(QtWidgets.QLabel("F(y) model:"))
        self.cb_kerma_modelo = QtWidgets.QComboBox()
        self.cb_kerma_modelo.addItem("Stretched exp. (legacy)", "stretched")
        self.cb_kerma_modelo.addItem("V2 (surface zones, PDD-space fit)", "v2")
        self.cb_kerma_modelo.setToolTip("Same kernel exp(-b(x-y)^2) and same b/d0 in both. V2 keeps the surface "
                                        "cells of F(y) and is fitted in PDD space (indirect/kerma_model_v2.py).")
        row.addWidget(self.cb_kerma_modelo)
        self.btn_rec = QtWidgets.QPushButton("Generate curve")
        self.btn_rec.clicked.connect(self._reconstruir_avancada)
        row.addWidget(self.btn_rec)
        lay.addLayout(row)
        self.hist_rec = QtWidgets.QVBoxLayout()
        lay.addLayout(self.hist_rec)
        root.addWidget(self.grp_rec)

        # ---- 5b. reconstrucao simples ----
        self.grp_simples = QtWidgets.QGroupBox(
            "5b. SIMPLE Reconstruction — uses the already calculated numeric kerma directly")
        lay = QtWidgets.QVBoxLayout(self.grp_simples)
        self.lbl_simples_status = QtWidgets.QLabel("")
        self.lbl_simples_status.setWordWrap(True)
        lay.addWidget(self.lbl_simples_status)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Combination (field, SSD):"))
        self.cb_simples = QtWidgets.QComboBox()
        row.addWidget(self.cb_simples, 1)
        row.addWidget(QtWidgets.QLabel("Points:"))
        self.sp_simples_n = ispin(10, 100000, 300, 10)
        row.addWidget(self.sp_simples_n)
        row.addWidget(QtWidgets.QLabel("Max depth (cm):"))
        self.sp_simples_xmax = dspin(1.0, 200.0, 30.0, 2, 1.0)
        row.addWidget(self.sp_simples_xmax)
        self.btn_simples = QtWidgets.QPushButton("Generate curve (simple mode)")
        self.btn_simples.clicked.connect(self._reconstruir_simples)
        row.addWidget(self.btn_simples)
        lay.addLayout(row)
        self.hist_simples = QtWidgets.QVBoxLayout()
        lay.addLayout(self.hist_simples)
        root.addWidget(self.grp_simples)

        self.btn_pasta = UrlButton()
        root.addWidget(self.btn_pasta)
        root.addStretch(1)
        self._construido = True

    # ---------- util ----------
    def _sync_state(self):
        self.st.out_dir = self.ed_out.text().strip() or self.st.out_dir
        self.st.ssd1 = float(self.sp_ssd1.value())
        self.st.energia = float(self.sp_energia.value())
        self.st.campo_fixo = float(self.sp_campo_fixo.value())

    def _escolher_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output directory",
                                                        self.ed_out.text())
        if d:
            self.ed_out.setText(d)
            self.btn_pasta.set_pasta(d)

    def _refresh_gates(self):
        base = self.st.datasets is not None
        self.grp_sup.setEnabled(base)
        self.grp_kerma.setEnabled(base)
        ms = self._ms()
        tem_kerma = ms.df_tidy is not None
        tem_coef = ms.df_coef is not None
        tem_modelo = bool(ms.fits)
        self.grp_coef.setEnabled(base and tem_kerma)
        self.grp_modelo.setEnabled(base and tem_coef)
        self.grp_rec.setEnabled(base and tem_modelo)
        self.grp_simples.setEnabled(base and tem_coef)
        self.chk_cache.setVisible(cache_existe(self.st.out_dir))
        self.btn_pasta.set_pasta(self.st.out_dir)
        eh_ssd = self.st.param_type == "ssd"
        self.lbl_campo_fixo.setVisible(eh_ssd)
        self.sp_campo_fixo.setVisible(eh_ssd)
        # fourier widgets
        fourier = self._metodo() == "fourier"
        self.sp_tend.setEnabled(not fourier)
        self.lbl_fourier.setVisible(fourier)
        self.chk_fourier_reg.setVisible(fourier)
        self.sp_fourier_lam.setEnabled(fourier and self.chk_fourier_reg.isChecked())
        self.sp_fourier_lam.setVisible(fourier)
        # status surfaces / reconstruction
        if self.state.grade is not None:
            self.lbl_sup_status.setText("A/B surfaces already available "
                                        "(built here or in the Direct tab).")
        else:
            self.lbl_sup_status.setText("No A/B surfaces built yet.")
        if self.state.grade is not None:
            self.lbl_rec_status.setText(
                "Surface B(field,SSD) found — b and d0 come from there "
                "(recommended).")
        else:
            self.lbl_rec_status.setText(
                "Surface B(field,SSD) not yet built. Build it in item 1b "
                "above (or in the Direct tab → item 4); without it, b and d0 use a "
                "fallback griddata on the sparse kerma table.")
        if tem_coef:
            self.lbl_simples_status.setText(
                "Reconstructs using the already calculated numeric kerma F(y) "
                "directly — only for (field, SSD) combinations already extracted.")
        else:
            self.lbl_simples_status.setText(
                "Available after calculating coefficients (item 3).")

    # ---------- 1. carga ----------
    def _carregar_csv(self):
        src, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose base CSV", DATA_DIR, "CSV (*.csv)")
        if not src:
            return
        out_dir = self.ed_out.text().strip() or self.st.out_dir
        ssd1 = float(self.sp_ssd1.value())
        try:
            os.makedirs(out_dir, exist_ok=True)
            destino = os.path.join(out_dir, "base_" + os.path.basename(src))
            shutil.copyfile(src, destino)
            datasets, params, param_type, param_unit = load_csv(destino)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error loading CSV", str(e))
            return
        self.st.out_dir = out_dir
        self.st.ssd1 = ssd1
        self.st.datasets = datasets
        self.st.params = params
        self.st.param_type = param_type
        self.st.param_unit = param_unit
        self._refresh_gates()
        self.lbl_base.setText(
            f"Base loaded: {len(datasets)} curves · parameter: {param_type} "
            f"({param_unit}) · file: {os.path.basename(destino)}")
        self.sp_g_start.setValue(ssd1)
        self.sp_sup_start.setValue(ssd1)
        self.sp_rec_ssd.setValue(ssd1)

    # ---------- 1b. superficies ----------
    def _montar_superficies(self):
        self._sync_state()
        out_dir = self.st.out_dir
        usar_cache = self.chk_cache.isChecked() and cache_existe(out_dir)
        datasets = self.st.datasets
        params = self.st.params
        param_type = self.st.param_type
        ssd1 = self.st.ssd1
        fixed_d0 = float(self.sp_d0.value()) if self.chk_d0.isChecked() else None
        start = float(self.sp_sup_start.value())
        end = float(self.sp_sup_end.value())
        n = int(self.sp_sup_n.value())
        fixed_info = self.fixed_info("i")

        def tarefa():
            dir_sup = subdir(out_dir, "superficies")
            if usar_cache:
                grade = carregar_ou_construir_interpoladores(dir_sup)
            else:
                ssds = np.linspace(start, end, n)
                grade = carregar_ou_construir_interpoladores(
                    dir_sup, datasets=datasets, params=params,
                    param_type=param_type, ssd1=ssd1, ssds_grade=ssds,
                    fixed_d0=fixed_d0, forcar_rebuild=True)
            pa = os.path.join(dir_sup, "superficie_A.png")
            pb = os.path.join(dir_sup, "superficie_B.png")
            plotar_superficie_ab(grade["superficie_a"], "A Surface", "A", pa,
                                 fixed_info=fixed_info)
            plotar_superficie_ab(grade["superficie_b"], "B Surface", "B", pb,
                                 fixed_info=fixed_info)
            return grade, pa, pb

        def pronto(res):
            grade, pa, pb = res
            self.state.grade = grade
            self.img_sup_a.set_image(pa)
            self.img_sup_b.set_image(pb)
            self._refresh_gates()
            self.main.comparison_tab.refresh_instances()

        self.run_task("Building A/B surfaces…", tarefa, pronto)

    # ---------- 2. kermas ----------
    def _calcular_kermas(self):
        self._sync_state()
        metodo = self._metodo()
        N = int(self.sp_N.value())
        t_end = float(self.sp_tend.value())
        regularizar = self.chk_fourier_reg.isChecked()
        lam = float(self.sp_fourier_lam.value())
        start = float(self.sp_g_start.value())
        end = float(self.sp_g_end.value())
        n = int(self.sp_g_n.value())
        fixed_d0 = float(self.sp_d0k.value()) if self.chk_d0k.isChecked() else None
        datasets = self.st.datasets
        params = self.st.params
        param_type = self.st.param_type
        ssd1 = self.st.ssd1
        out_dir = self.st.out_dir
        fixed_info = self.fixed_info("i")

        def tarefa():
            ssds = np.linspace(start, end, n)
            df_tidy, kermas = gerar_kermas_batch(
                datasets, params, param_type, ssd1, ssds, fixed_d0=fixed_d0,
                N=N, t_end=t_end, metodo=metodo,
                regularizar_fourier=regularizar, lam_fourier=lam)
            pasta = subdir(out_dir, "kermas")
            csv = os.path.join(pasta, f"kermas_tidy_{metodo}.csv")
            save_df(df_tidy, csv)
            png = os.path.join(pasta, f"kermas_todos_{metodo}.png")
            plotar_kermas(kermas, png, fixed_info=fixed_info)
            return metodo, df_tidy, kermas, ssds, N, csv, png

        def pronto(res):
            metodo_r, df_tidy, kermas, ssds, N_r, csv, png = res
            ms = self.st.metodo(metodo_r)
            ms.df_tidy = df_tidy
            ms.kermas = kermas
            ms.ssds_grade = ssds
            ms.N_val = N_r
            ms.kermas_csv = csv
            ms.kermas_png = png
            ms.reset_downstream()
            if metodo_r == self._metodo():
                self._refresh_methodo_view()
            self._refresh_gates()

        self.run_task(f"Calculating kermas ({metodo})…", tarefa, pronto)

    # ---------- 3. coefficients ----------
    def _calcular_coef(self):
        self._sync_state()
        metodo = self._metodo()
        ms = self._ms()
        if ms.df_tidy is None:
            return
        df_tidy = ms.df_tidy
        kermas = ms.kermas
        param_type = self.st.param_type
        out_dir = self.st.out_dir
        fixed_info = self.fixed_info("i")

        def tarefa():
            df_coef = tabelar_coeficientes(df_tidy, kermas, param_type)
            pasta = subdir(out_dir, "coefficients")
            csv = os.path.join(pasta, f"coefficients_kerma_{metodo}.csv")
            save_df(df_coef, csv)
            pngs = {}
            for coluna in self.COLS_COEF_PNG:
                png = os.path.join(pasta, f"{coluna}_vs_campo_ssd_{metodo}.png")
                plotar_coeficiente_vs_campo_ssd(df_coef, coluna, png,
                                                param_type=param_type,
                                                fixed_info=fixed_info)
                pngs[coluna] = png
            return metodo, df_coef, csv, pngs

        def pronto(res):
            metodo_r, df_coef, csv, pngs = res
            ms = self.st.metodo(metodo_r)
            ms.df_coef = df_coef
            ms.coef_csv = csv
            ms.coef_pngs = pngs
            ms.fits = {}
            ms.fit_pngs = {}
            if metodo_r == self._metodo():
                self._refresh_methodo_view()
            self._refresh_gates()

        self.run_task(f"Calculating coefficients ({metodo})…", tarefa, pronto)

    # ---------- 4. modelo ----------
    def _ajustar_modelo(self):
        self._sync_state()
        metodo = self._metodo()
        ms = self._ms()
        if ms.df_coef is None:
            return
        df_coef = ms.df_coef
        param_type = self.st.param_type
        out_dir = self.st.out_dir
        fixed_info = self.fixed_info("i")

        def tarefa():
            fits = ajustar_modelo_kerma(df_coef, param_type)
            fit_A, fit_tau, fit_beta = fits
            pares = [("A", fit_A), ("tau", fit_tau), ("beta", fit_beta)]
            pasta = subdir(out_dir, "modelo")
            pngs = {}
            for nome, fit in pares:
                png = os.path.join(pasta, f"modelo_{nome}_{metodo}.png")
                plotar_ajuste_modelo(df_coef, fit, png, param_type=param_type,
                                     fixed_info=fixed_info)
                pngs[nome] = png
            # devolve fits como dict + ordem
            dict_fits = {"A": fit_A, "tau": fit_tau, "beta": fit_beta}
            return metodo, dict_fits, pngs

        def pronto(res):
            metodo_r, dict_fits, pngs = res
            ms = self.st.metodo(metodo_r)
            ms.fits = dict_fits
            ms.fit_pngs = pngs
            if metodo_r == self._metodo():
                self._refresh_methodo_view()
            self._refresh_gates()

        self.run_task(f"Fitting generalized model ({metodo})…", tarefa, pronto)

    # ---------- 5. reconstrucao avancada ----------
    def _reconstruir_avancada(self):
        self._sync_state()
        metodo = self._metodo()
        ms = self._ms()
        kmodelo = self.cb_kerma_modelo.currentData() or "stretched"
        if kmodelo == "v2":
            if ms.kermas is None or ms.df_tidy is None:
                return
        elif not ms.fits or ms.df_coef is None:
            return
        n_pts = int(self.sp_rec_n.value())
        x_max = float(self.sp_rec_xmax.value())
        campo = float(self.sp_rec_campo.value())
        ssd = float(self.sp_rec_ssd.value())
        out_dir = self.st.out_dir
        energia = self.st.energia
        grade = self.state.grade
        df_coef = ms.df_coef
        fits = ms.fits
        N_val = ms.N_val
        param_type = self.st.param_type

        def tarefa():
            if kmodelo == "v2":
                if ms.v2_modelo is None:      # per-curve PDD-space fits (~20-30 s for 16 fields x 15 SSD), cached
                    df_v2 = tabelar_coeficientes(ms.df_tidy, ms.kermas, param_type, modelo="v2")
                    ms.v2_modelo = ajustar_modelo_kerma(df_v2, param_type, modelo="v2")
                x_curva, pdd_ind, info = reconstruir_pdd_indireta(
                    df_coef, None, None, None, campo, ssd, n_pts=n_pts, x_max=x_max,
                    param_type=param_type, grade_direto=grade, modelo="v2", modelo_v2=ms.v2_modelo)
            else:
                x_curva, pdd_ind, info = reconstruir_pdd_indireta(
                    df_coef, fits["A"], fits["tau"], fits["beta"],
                    campo, ssd, n_pts=n_pts, x_max=x_max, N=N_val,
                    grade_direto=grade)
            stem = f"pdd_indireta_{metodo}_c{campo}_ssd{ssd}".replace(".", "p")
            pasta = subdir(out_dir, "curvas")
            csv = os.path.join(pasta, stem + ".csv")
            png = os.path.join(pasta, stem + ".png")
            save_df(pd.DataFrame({"profundidade_cm": x_curva, "pdd_%": pdd_ind}), csv)
            plotar_pdd_reconstruida(x_curva, pdd_ind, info, png, campo=campo,
                                    ssd=ssd, fixed_info={"energy": energia})
            return metodo, {
                "info": dict(info), "png": png, "csv": csv, "campo": campo,
                "ssd": ssd, "metodo": metodo, "x_curva": np.asarray(x_curva),
                "pdd_curva": np.asarray(pdd_ind),
            }

        def pronto(res):
            metodo_r, item = res
            ms = self.st.metodo(metodo_r)
            ms.curvas_hist.append(item)
            if metodo_r == self._metodo():
                self._refresh_methodo_view()
            self.main.comparison_tab.refresh_instances()

        try:
            self.run_task(f"Reconstructing advanced inverse PDD ({metodo})…",
                          tarefa, pronto)
        except ValueError as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    # ---------- 5b. reconstrucao simples ----------
    def _reconstruir_simples(self):
        self._sync_state()
        metodo = self._metodo()
        ms = self._ms()
        if ms.df_coef is None or ms.kermas is None:
            return
        idx = self.cb_simples.currentIndex()
        combos = sorted(ms.kermas.keys())
        if idx < 0 or idx >= len(combos):
            return
        campo_s, ssd_s = combos[idx]
        n_pts = int(self.sp_simples_n.value())
        x_max = float(self.sp_simples_xmax.value())
        out_dir = self.st.out_dir
        energia = self.st.energia
        df_coef = ms.df_coef
        kermas = ms.kermas
        param_type = self.st.param_type

        def tarefa():
            x_curva, pdd, info = reconstruir_pdd_indireta_simples(
                df_coef, kermas, campo_s, ssd_s, n_pts=n_pts, x_max=x_max,
                param_type=param_type)
            stem = f"pdd_indireta_simples_{metodo}_c{campo_s}_ssd{ssd_s}".replace(".", "p")
            pasta = subdir(out_dir, "curvas")
            csv = os.path.join(pasta, stem + ".csv")
            save_df(pd.DataFrame({"profundidade_cm": x_curva, "pdd_%": pdd}), csv)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(x_curva, pdd, "orange", lw=2.2,
                    label=f"Inverse PDD (simple, {metodo}) — field={campo_s}  SSD={ssd_s:g} cm")
            ax.axvline(info["d0"], color="green", lw=1, ls=":", alpha=0.7,
                       label=f"d0 = {info['d0']:.1f} cm")
            ax.set_xlabel("Depth (cm)")
            ax.set_ylabel("PDD (%)")
            ax.set_title(f"Simple reconstruction ({metodo}) — b={info['b']:.4f}")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            if energia is not None:
                fig.text(0.99, 0.01, f"Energy = {energia:g} MeV", ha='right',
                         va='bottom', fontsize=8, style='italic', color='dimgray')
            png = os.path.join(pasta, stem + ".png")
            fig.tight_layout()
            fig.savefig(png, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return metodo, {
                "info": dict(info), "png": png, "csv": csv, "campo": campo_s,
                "ssd": ssd_s, "metodo": metodo, "x_curva": np.asarray(x_curva),
                "pdd_curva": np.asarray(pdd),
            }

        def pronto(res):
            metodo_r, item = res
            ms = self.st.metodo(metodo_r)
            ms.curvas_hist_simples.append(item)
            if metodo_r == self._metodo():
                self._refresh_methodo_view()
            self.main.comparison_tab.refresh_instances()

        self.run_task(f"Simple reconstruction ({metodo})…", tarefa, pronto)

    # ---------- refrescar visualizacao do metodo atual ----------
    def _refresh_methodo_view(self):
        if not getattr(self, "_construido", False):
            return
        ms = self._ms()
        # section 2
        self.tbl_kerma.set_dataframe(ms.df_tidy)
        self.out_kerma.set_path(ms.kermas_csv)
        self.img_kerma.set_image(ms.kermas_png)
        outro = "fourier" if self._metodo() == "hopfield" else "hopfield"
        ms_outro = self.st.metodo(outro)
        if ms_outro.df_tidy is not None:
            self.lbl_kerma_outro.setText(
                f"(Kermas also calculated by {outro} — available for "
                "reconstruction and comparison; switch the method above to see them.)")
        else:
            self.lbl_kerma_outro.setText("")
        # section 3
        self.tbl_coef.set_dataframe(ms.df_coef)
        self.out_coef.set_path(ms.coef_csv)
        for col, img in self.imgs_coef.items():
            img.set_image(ms.coef_pngs.get(col))
        # section 4
        for nome in self.NOMES_FIT:
            lbl_n, lbl_r, lbl_a, img = self.fits_ui[nome]
            fit = ms.fits.get(nome)
            if fit is None:
                lbl_n.setText("—")
                lbl_r.setText("")
                lbl_a.setText("")
                img.set_image(None)
                continue
            lbl_n.setText(fit["formula_numerica"])
            r2_txt = (f"{fit['r2_treino']:.4f}"
                      if fit.get("r2_treino") is not None else "—")
            lbl_r.setText(f"RMSE treino = {fit['rmse_treino']:.4g}  ·  R² treino = {r2_txt}")
            aviso = fit.get("aviso", "")
            lbl_a.setText(aviso if "AVISO" in aviso else "")
            img.set_image(ms.fit_pngs.get(nome))
        # section 5b combo
        self.cb_simples.clear()
        if ms.kermas:
            for c, s in sorted(ms.kermas.keys()):
                self.cb_simples.addItem(f"field {c} · SSD {s:g} cm")
        # historicos
        self._rebuild_hist_avancada()
        self._rebuild_hist_simples()
        self._refresh_gates()

    def _rebuild_hist_avancada(self):
        ms = self._ms()
        layout = self.hist_rec
        while layout.count():
            it = layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        for h in reversed(ms.curvas_hist):
            info = h["info"]
            box = QtWidgets.QGroupBox()
            v = QtWidgets.QVBoxLayout(box)
            v.addWidget(titulo(
                f"[{h.get('metodo', '—')}] Field {fmt_num(h['campo'])} · "
                f"SSD {fmt_num(h['ssd'])} cm — "
                f"A_previsto={info['A_previsto']:.6f}  "
                f"tau_previsto={info['tau_previsto']:.6f}  "
                f"beta_previsto={info['beta_previsto']:.6f}  "
                f"b={info['b']:.6f}  d0={info['d0']:.2f} cm  "
                f"· b vindo de: {info.get('b_origem', '—')}"))
            # diagnostico vs ponto tabulado mais proximo
            if ms.df_coef is not None:
                try:
                    linha, dist = linha_treino_mais_proxima(
                        ms.df_coef, self.st.param_type, h["campo"], h["ssd"])
                    diffs = []
                    for nome_curto, chave_info, chave_tab in [
                        ("A", "A_previsto", "A_kerma"),
                        ("τ", "tau_previsto", "tau_kerma"),
                        ("β", "beta_previsto", "beta_kerma"),
                    ]:
                        if chave_info not in info or chave_tab not in linha:
                            continue
                        tab = float(linha[chave_tab])
                        dif = (100.0 * (info[chave_info] - tab) / tab) if tab else float('nan')
                        diffs.append((nome_curto, tab, dif))
                    diffs_txt = "  ".join(
                        f"{n} tabulado={t:.6f} (Δ={d:+.1f}%)" for n, t, d in diffs)
                    lbl = QtWidgets.QLabel(
                        f"Nearest tabulated point: field={fmt_num(linha[self.st.param_type])}  "
                        f"ssd={fmt_num(linha['ssd'])}  (distance={dist:.3g}) — {diffs_txt}")
                    lbl.setWordWrap(True)
                    v.addWidget(lbl)
                    if any(abs(d[2]) > 15 for d in diffs):
                        aviso = QtWidgets.QLabel(
                            "⚠️ Large difference in some parameter — the generalized "
                            "model probably does not capture this point well.")
                        aviso.setWordWrap(True)
                        aviso.setStyleSheet("color: #b00020;")
                        v.addWidget(aviso)
                except Exception:
                    pass
            img = ImageView()
            img.set_image(h["png"])
            v.addWidget(img)
            row = OutputRow()
            row.set_path(h["csv"])
            v.addWidget(row)
            layout.addWidget(box)

    def _rebuild_hist_simples(self):
        ms = self._ms()
        layout = self.hist_simples
        while layout.count():
            it = layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        for h in reversed(ms.curvas_hist_simples):
            info = h["info"]
            box = QtWidgets.QGroupBox()
            v = QtWidgets.QVBoxLayout(box)
            v.addWidget(titulo(
                f"[Simple · {h.get('metodo', '—')}] Field {fmt_num(h['campo'])} · "
                f"SSD {fmt_num(h['ssd'])} cm — b={info['b']:.6f}  d0={info['d0']:.2f} cm"))
            img = ImageView()
            img.set_image(h["png"])
            v.addWidget(img)
            row = OutputRow()
            row.set_path(h["csv"])
            v.addWidget(row)
            layout.addWidget(box)

    # ---------- 3D Interativo ----------
    def _abrir_3d_interativo_coef(self, coluna):
        """Abre dialog interativo para scatter 3D de coeficiente."""
        ms = self._ms()
        if ms.df_coef is None:
            return
        df_coef = ms.df_coef
        param_type = self.st.param_type
        fixed_info = self.fixed_info("i")
        rotulo_coef = self.DISPLAY_FIT.get(coluna.replace("_kerma", ""), coluna)
        titulo = f"Variation of {rotulo_coef} with ({_rotulo_param(param_type)}, SSD)"
        
        def plot_fn(ax):
            rotulo_param = _rotulo_param(param_type)
            sc = ax.scatter(df_coef[param_type], df_coef['ssd'], df_coef[coluna],
                             c=df_coef[coluna], cmap='viridis', s=40, edgecolor='k', linewidth=0.3)
            ax.set_xlabel(rotulo_param)
            ax.set_ylabel('SSD (cm)')
            ax.set_zlabel(rotulo_coef)
            ax.set_title(titulo)
        
        dlg = Interactive3DPlot(self, titulo, plot_fn)
        dlg.exec_()

    def _abrir_3d_interativo_modelo(self, nome):
        """Abre dialog interativo para superficie do modelo ajustado."""
        ms = self._ms()
        if not ms.fits or ms.df_coef is None:
            return
        fit = ms.fits.get(nome)
        if fit is None:
            return
        df_coef = ms.df_coef
        param_type = self.st.param_type
        fixed_info = self.fixed_info("i")
        
        campo_vals = np.linspace(df_coef[param_type].min(), df_coef[param_type].max(), 30)
        ssd_vals = np.linspace(df_coef['ssd'].min(), df_coef['ssd'].max(), 30)
        Campo, Ssd = np.meshgrid(campo_vals, ssd_vals)
        Z = fit['prever'](Campo.ravel(), Ssd.ravel()).reshape(Campo.shape)
        
        rotulo_param = _rotulo_param(param_type)
        r2_txt = f", R²={fit['r2_treino']:.4g}" if fit.get('r2_treino') is not None else ""
        titulo = f"Model for {nome}  (training RMSE={fit['rmse_treino']:.4g}{r2_txt})"
        
        def plot_fn(ax):
            ax.plot_surface(Campo, Ssd, Z, cmap='viridis', alpha=0.55, edgecolor='none')
            ax.scatter(df_coef[param_type], df_coef['ssd'], df_coef[fit['coluna']],
                       color='red', s=25, label='Computed values (per-curve stretched-exponential fit)')
            ax.set_xlabel(rotulo_param)
            ax.set_ylabel('SSD (cm)')
            ax.set_zlabel(fit['coluna'])
            ax.set_title(titulo)
            ax.legend(fontsize=8, loc='upper left')
        
        dlg = Interactive3DPlot(self, titulo, plot_fn)
        dlg.exec_()


# ============================================================
#  ABA COMPARACAO
# ============================================================

class ComparisonInstance(QtWidgets.QWidget):
    """Uma aba de comparacao independente."""

    def __init__(self, main, inst_id):
        super().__init__()
        self.main = main
        self.inst_id = inst_id
        self.state = main.state
        self.curvas_extra = []
        self.resultado = None
        self._build()
        self.refresh()

    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)

        # ---- adicionar referencia ----
        self.grp_ref = QtWidgets.QGroupBox("Add reference curve (optional)")
        v = QtWidgets.QVBoxLayout(self.grp_ref)
        self.cb_modo = QtWidgets.QComboBox()
        self.cb_modo.addItem("From an already loaded file (Direct/Inverse)")
        self.cb_modo.addItem("Manual entry (spreadsheet)")
        self.cb_modo.currentIndexChanged.connect(self._trocar_modo)
        v.addWidget(self.cb_modo)

        self.stack = QtWidgets.QStackedWidget()
        # painel arquivo
        p1 = QtWidgets.QWidget()
        g1 = QtWidgets.QHBoxLayout(p1)
        g1.addWidget(QtWidgets.QLabel("Source:"))
        self.cb_fonte = QtWidgets.QComboBox()
        self.cb_fonte.currentIndexChanged.connect(self._update_source_curves)
        g1.addWidget(self.cb_fonte, 1)
        g1.addWidget(QtWidgets.QLabel("Curve (field):"))
        self.cb_curva_ref = QtWidgets.QComboBox()
        g1.addWidget(self.cb_curva_ref, 1)
        g1.addWidget(QtWidgets.QLabel("Convert to SSD:"))
        self.sp_ssd_ref = dspin(1.0, 1000.0, 100.0, 2, 1.0)
        g1.addWidget(self.sp_ssd_ref)
        b_add_ref = QtWidgets.QPushButton("Add curve")
        b_add_ref.clicked.connect(self._add_ref_file)
        g1.addWidget(b_add_ref)
        self.stack.addWidget(p1)
        # painel manual
        p2 = QtWidgets.QWidget()
        v2 = QtWidgets.QVBoxLayout(p2)
        self.tbl_manual = PasteTableWidget(8, 2)
        self.tbl_manual.setHorizontalHeaderLabels(["Depth (cm)", "PDD (%)"])
        self.tbl_manual.horizontalHeader().setStretchLastSection(True)
        self.tbl_manual.setAlternatingRowColors(True)
        v2.addWidget(self.tbl_manual)
        lbl_dica = QtWidgets.QLabel(
            "Tip: click the first cell and press Ctrl+V (or right-click → "
            "Paste) to paste a whole block copied from Excel, Calc or a "
            "CSV (decimal comma or dot).")
        lbl_dica.setWordWrap(True)
        v2.addWidget(lbl_dica)
        h2 = QtWidgets.QHBoxLayout()
        b_row = QtWidgets.QPushButton("Add row")
        b_row.clicked.connect(self._add_linha_manual)
        h2.addWidget(b_row)
        h2.addWidget(QtWidgets.QLabel("Curve name:"))
        self.ed_nome_manual = QtWidgets.QLineEdit("Manual curve")
        h2.addWidget(self.ed_nome_manual, 1)
        b_add_manual = QtWidgets.QPushButton("Add manual curve")
        b_add_manual.clicked.connect(self._add_manual)
        h2.addWidget(b_add_manual)
        v2.addLayout(h2)
        self.stack.addWidget(p2)
        v.addWidget(self.stack)
        lay.addWidget(self.grp_ref)

        # ---- selecao ----
        grp_sel = QtWidgets.QGroupBox("Curves to compare")
        v3 = QtWidgets.QVBoxLayout(grp_sel)
        b_atualiza = QtWidgets.QPushButton("Refresh curve list")
        b_atualiza.clicked.connect(self.refresh)
        v3.addWidget(b_atualiza)
        self.lst = QtWidgets.QListWidget()
        self.lst.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        v3.addWidget(self.lst)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Reference curve for error calculation:"))
        self.cb_ref_erro = QtWidgets.QComboBox()
        h.addWidget(self.cb_ref_erro, 1)
        self.btn_gerar = QtWidgets.QPushButton("Generate comparison")
        self.btn_gerar.clicked.connect(self._gerar)
        h.addWidget(self.btn_gerar)
        v3.addLayout(h)
        lay.addWidget(grp_sel)

        # ---- resultados ----
        grp_res = QtWidgets.QGroupBox("Result")
        v4 = QtWidgets.QVBoxLayout(grp_res)
        self.img = ImageView()
        v4.addWidget(self.img)
        self.out_csv = OutputRow()
        v4.addWidget(self.out_csv)
        self.tbl_erros = TableBox(f"erros_comparacao_cmp{self.inst_id}.csv")
        v4.addWidget(self.tbl_erros)
        lay.addWidget(grp_res)

    # ---- util ----
    def _trocar_modo(self, idx):
        self.stack.setCurrentIndex(idx)

    def _available_bases(self):
        bases = []
        if self.state.direct.datasets is not None:
            bases.append("Base from Direct file")
        if self.state.inverse.datasets is not None:
            bases.append("Base from Inverse file")
        return bases

    def _update_source_curves(self):
        self.cb_curva_ref.clear()
        bases = self._available_bases()
        if not bases:
            return
        fonte = self.cb_fonte.currentText()
        ds = (self.state.direct.datasets if fonte.startswith("Base from Direct file")
              else self.state.inverse.datasets)
        if ds:
            self.cb_curva_ref.addItems(list(ds.keys()))

    def _add_linha_manual(self):
        self.tbl_manual.insertRow(self.tbl_manual.rowCount())

    def _add_ref_file(self):
        bases = self._available_bases()
        if not bases or self.cb_curva_ref.count() == 0:
            QtWidgets.QMessageBox.information(
                self, "Sem base",
                "Load a base CSV in Direct and/or Inverse, or type the curve "
                "manualmente.")
            return
        fonte = self.cb_fonte.currentText()
        if fonte.startswith("Base from Direct file"):
            ds, ssd1 = self.state.direct.datasets, self.state.direct.ssd1
        else:
            ds, ssd1 = self.state.inverse.datasets, self.state.inverse.ssd1
        nome = self.cb_curva_ref.currentText()
        ssd_alvo = float(self.sp_ssd_ref.value())
        from common.mayneord import apply_mayneord
        xm, ym = ds[nome]
        d0 = detect_dmax_xy(xm, ym)
        if abs(ssd_alvo - ssd1) > 1e-9:
            ym = apply_mayneord(ym, xm, ssd1, ssd_alvo, d0)
        self.curvas_extra.append({
            "label": f"[Experimental] {nome} · SSD {ssd_alvo:g} cm",
            "x": np.asarray(xm), "y": np.asarray(ym), "campo": nome,
            "ssd": ssd_alvo, "origem": "Experimental",
        })
        self.refresh()

    def _add_manual(self):
        xs, ys = [], []
        for i in range(self.tbl_manual.rowCount()):
            it_x = self.tbl_manual.item(i, 0)
            it_y = self.tbl_manual.item(i, 1)
            if it_x is None or it_y is None:
                continue
            try:
                x = float(str(it_x.text()).strip().replace(",", "."))
                y = float(str(it_y.text()).strip().replace(",", "."))
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
        if len(xs) < 2:
            QtWidgets.QMessageBox.warning(
                self, "Few points",
                "Enter at least 2 valid points (depth and PDD).")
            return
        ordem = np.argsort(xs)
        nome = self.ed_nome_manual.text().strip() or "Manual curve"
        self.curvas_extra.append({
            "label": f"[Manual] {nome}", "x": np.asarray(xs)[ordem],
            "y": np.asarray(ys)[ordem], "campo": nome, "ssd": None,
            "origem": "Manual",
        })
        # limpa a planilha (grid novo, como o "bump" do editor no Streamlit)
        self.tbl_manual.clearContents()
        self.tbl_manual.setRowCount(8)
        self.refresh()

    def collect_curves(self):
        curvas = []
        for h in self.state.direct.curvas_hist:
            curvas.append({
                "label": f"[Direct] field {fmt_num(h['campo'])} · SSD {fmt_num(h['ssd'])} cm",
                "x": np.asarray(h["x_curva"]), "y": np.asarray(h["pdd_conv"]),
                "campo": h["campo"], "ssd": h["ssd"], "origem": "Direct",
            })
        for ms_metodo in ("hopfield", "fourier"):
            ms = self.state.inverse.metodo(ms_metodo)
            for h in ms.curvas_hist:
                metodo_h = h.get("metodo", ms_metodo)
                curvas.append({
                    "label": f"[Advanced Inverse · {metodo_h}] field {fmt_num(h['campo'])} · SSD {fmt_num(h['ssd'])} cm",
                    "x": np.asarray(h["x_curva"]), "y": np.asarray(h["pdd_curva"]),
                    "campo": h["campo"], "ssd": h["ssd"], "origem": "Inverse advanced",
                })
            for h in ms.curvas_hist_simples:
                metodo_h = h.get("metodo", ms_metodo)
                curvas.append({
                    "label": f"[Simple Inverse · {metodo_h}] field {fmt_num(h['campo'])} · SSD {fmt_num(h['ssd'])} cm",
                    "x": np.asarray(h["x_curva"]), "y": np.asarray(h["pdd_curva"]),
                    "campo": h["campo"], "ssd": h["ssd"], "origem": "Inverse simple",
                })
        curvas.extend(self.curvas_extra)
        return curvas

    def refresh(self):
        """Rebuilds the curve list and the error reference combo."""
        # fontes de arquivo
        self.cb_fonte.blockSignals(True)
        fonte_atual = self.cb_fonte.currentText()
        self.cb_fonte.clear()
        bases = self._available_bases()
        self.cb_fonte.addItems(bases if bases else ["(no base loaded)"])
        if fonte_atual in bases:
            self.cb_fonte.setCurrentText(fonte_atual)
        self.cb_fonte.blockSignals(False)
        self._update_source_curves()
        self.sp_ssd_ref.setValue(
            self.state.direct.ssd1 if self.state.direct.datasets is not None
            else self.state.inverse.ssd1)

        self.curvas = self.collect_curves()
        marcados = {self.lst.item(i).text() for i in range(self.lst.count())
                    if self.lst.item(i).checkState() == Qt.Checked}
        self.lst.blockSignals(True)
        self.lst.clear()
        labels = [c["label"] for c in self.curvas]
        for lbl in labels:
            it = QtWidgets.QListWidgetItem(lbl)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if lbl in marcados else Qt.Unchecked)
            self.lst.addItem(it)
        # marca por padrao as ultimas ate 3, se nada estava marcado
        if not marcados and labels:
            for lbl in labels[-min(3, len(labels)):]:
                for i in range(self.lst.count()):
                    if self.lst.item(i).text() == lbl:
                        self.lst.item(i).setCheckState(Qt.Checked)
        self.lst.blockSignals(False)

        # combo de referencia de erro
        atual = self.cb_ref_erro.currentText()
        self.cb_ref_erro.clear()
        self.cb_ref_erro.addItem("(none)")
        self.cb_ref_erro.addItems(labels)
        if atual in labels:
            self.cb_ref_erro.setCurrentText(atual)

    def _selecionadas(self):
        sel = []
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            if it.checkState() == Qt.Checked:
                sel.append(it.text())
        return [c for c in self.curvas if c["label"] in sel]

    def _gerar(self):
        curvas_sel = self._selecionadas()
        if not curvas_sel:
            QtWidgets.QMessageBox.information(self, "No curves",
                                              "Select at least one curve.")
            return
        ref_label = self.cb_ref_erro.currentText()
        out_dir = self.state.direct.out_dir or os.path.join(APP_DIR, "output_direto")
        inst_id = self.inst_id

        def tarefa():
            fig, ax = plt.subplots(figsize=(10, 5.5))
            cores = plt.cm.tab10(np.linspace(0, 1, len(curvas_sel)))
            for c, cor in zip(curvas_sel, cores):
                if c["origem"] in ("Experimental", "Manual"):
                    ax.scatter(c["x"], c["y"], s=22, color=cor, zorder=5,
                               label=c["label"])
                else:
                    ax.plot(c["x"], c["y"], color=cor, lw=2.0, label=c["label"])
            ax.set_xlabel("Depth (cm)")
            ax.set_ylabel("PDD (%)")
            ax.set_title("Curve comparison")
            ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pasta = subdir(out_dir, "comparacoes")
            png = os.path.join(pasta, f"comparacao_curvas_cmp{inst_id}.png")
            fig.savefig(png, dpi=200, bbox_inches="tight")
            plt.close(fig)

            df_erros = None
            if ref_label != "(none)":
                ref = next((c for c in curvas_sel if c["label"] == ref_label), None)
                if ref is not None:
                    linhas = []
                    for c in curvas_sel:
                        if c["label"] == ref_label:
                            continue
                        y_interp = np.interp(ref["x"], c["x"], c["y"])
                        erro = y_interp - ref["y"]
                        rmse = float(np.sqrt(np.mean(erro ** 2)))
                        linhas.append({"curva": c["label"],
                                       "erro_max": float(np.max(np.abs(erro))),
                                       "rmse": rmse})
                    df_erros = pd.DataFrame(linhas)

            x_base = curvas_sel[0]["x"]
            df_tabela = pd.DataFrame({"profundidade_cm": x_base})
            for c in curvas_sel:
                df_tabela[c["label"]] = np.interp(x_base, c["x"], c["y"])
            csv = os.path.join(pasta, f"comparacao_curvas_cmp{inst_id}.csv")
            save_df(df_tabela, csv)
            return png, csv, df_erros

        def pronto(res):
            png, csv, df_erros = res
            self.resultado = {"png": png, "csv": csv, "df_erros": df_erros}
            self.img.set_image(png)
            self.out_csv.set_path(csv)
            self.tbl_erros.set_dataframe(df_erros)

        self.main.direct_tab.run_task("Generating comparison…", tarefa, pronto)


class ComparisonTab(BaseTab):
    def __init__(self, main):
        super().__init__(main)
        self.state = main.state
        self.instances = []
        self._next_id = 0
        self._build()
        self.add_instance()

    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.btn_nova = QtWidgets.QPushButton("➕ New comparison tab")
        self.btn_nova.clicked.connect(self.add_instance)
        top.addWidget(self.btn_nova)
        top.addStretch(1)
        lay.addLayout(top)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._fechar_aba)
        lay.addWidget(self.tabs)

    def add_instance(self):
        inst_id = self._next_id
        self._next_id += 1
        inst = ComparisonInstance(self.main, inst_id)
        self.instances.append(inst)
        self.tabs.addTab(inst, f"Comparison {len(self.instances)}")
        self.tabs.setCurrentWidget(inst)
        return inst

    def _fechar_aba(self, index):
        if self.tabs.count() <= 1:
            QtWidgets.QMessageBox.information(
                self, "Keep tab", "At least one comparison tab must be kept.")
            return
        w = self.tabs.widget(index)
        self.tabs.removeTab(index)
        if w in self.instances:
            self.instances.remove(w)
        w.deleteLater()

    def refresh_instances(self):
        for inst in self.instances:
            inst.refresh()


# ============================================================
#  JANELA PRINCIPAL
# ============================================================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1280, 860)
        self.state = AppState()

        self.tabs = QtWidgets.QTabWidget()
        self.direct_tab = DirectTab(self)
        self.inverse_tab = InverseTab(self)
        self.comparison_tab = ComparisonTab(self)
        self.tabs.addTab(self.direct_tab, "🔵 Direct")
        self.tabs.addTab(self.inverse_tab, "🟠 Inverse")
        self.tabs.addTab(self.comparison_tab, "🟢 Comparison")
        self.tabs.currentChanged.connect(self._tab_mudou)
        self.setCentralWidget(self.tabs)

        # log dock
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        dock = QtWidgets.QDockWidget("Log / progress", self)
        dock.setWidget(self.log)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.dock_log = dock

        # menu
        m_arq = self.menuBar().addMenu("File")
        act_open_d = QtWidgets.QAction("Open Direct output folder", self)
        act_open_d.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(self.state.direct.out_dir)))
        act_open_i = QtWidgets.QAction("Open Inverse output folder", self)
        act_open_i.triggered.connect(
            lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(self.state.inverse.out_dir)))
        act_sair = QtWidgets.QAction("Exit", self)
        act_sair.triggered.connect(self.close)
        m_arq.addAction(act_open_d)
        m_arq.addAction(act_open_i)
        m_arq.addSeparator()
        m_arq.addAction(act_sair)

        m_view = self.menuBar().addMenu("View")
        m_view.addAction(dock.toggleViewAction())

        self.statusBar().showMessage("Pronto.")

    def _tab_mudou(self, idx):
        # equivale ao rerun do Streamlit: reavalia os gates ao trocar de aba
        self.direct_tab._refresh_gates()
        self.inverse_tab._refresh_gates()
        if self.tabs.widget(idx) is self.comparison_tab:
            self.comparison_tab.refresh_instances()

    def append_log(self, s):
        self.log.moveCursor(QtGui.QTextCursor.End)
        self.log.insertPlainText(s)
        self.log.moveCursor(QtGui.QTextCursor.End)


def main():
    if "--selftest" in sys.argv:
        try:
            return _executar_autoteste()
        except Exception:
            traceback.print_exc()
            return 1

    # Em builds --windowed (Windows) sys.stdout/stderr podem ser None, o que
    # quebraria os muitos print() dos modulos de calculo. Garante um destino
    # valido antes de construir a janela; depois o painel de log assume.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)

    emitter = StreamEmitter()
    win = MainWindow()
    emitter.text.connect(win.append_log)
    sys.stdout = emitter
    sys.stderr = emitter

    def excepthook(tipo, valor, tb):
        txt = "".join(traceback.format_exception(tipo, valor, tb))
        win.append_log(txt)
        QtWidgets.QMessageBox.critical(win, "Unexpected error", txt)

    sys.excepthook = excepthook

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    sys.exit(main())
