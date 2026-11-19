# MCMEG - Hopfield Network

Desktop tool (PyQt5) for working with percentage depth dose (PDD) curves using a Hopfield network.

- **Method 1** (direct) – fits each measured curve with a closed-form model (parameters *a*, *b*), applies the Mayneord factor to move between SSDs, builds A/B surfaces over field size and SSD, and reconstructs PDD curves for arbitrary field/SSD.
- **Method 2** (inverse) – extracts the numerical kerma F(y) from each curve (Hopfield network or Fourier/OLS). Two ways to reconstruct a PDD from it:
  - **Generalized** – fits a generalized model of F(y) over field size and SSD (stretched exponential by default, or `v2`) and reconstructs any (field, SSD).
  - **Simple** – uses the already extracted kerma directly, only for (field, SSD) combinations that were extracted.
- **Comparison** – overlays reconstructed curves against reference curves (loaded from a file or pasted in) and reports the error.

## Install

### Windows: download the .exe (no Python needed)

1. Open the [**Releases**](https://github.com/lsbeloti/Neural-Network/releases/latest) page and download `MCMEG_Hopfield.exe`.
2. Double-click it.

To also open it by typing `mcmeg-hopfield` in a terminal, use the pip install below instead.

### Any system: install and run from the terminal

Requires Python 3.9+. The recommended way is [pipx](https://pipx.pypa.io), which installs the program in its own environment and puts the `mcmeg-hopfield` command on your PATH:

```bash
pip install pipx
pipx ensurepath
pipx install git+https://github.com/lsbeloti/Neural-Network.git
```

Close and reopen the terminal, then start the program the same way you would type `firefox`:

```bash
mcmeg-hopfield
```

Without pipx:

```bash
pip install git+https://github.com/lsbeloti/Neural-Network.git
mcmeg-hopfield            # or: python -m mcmeg_hopfield
```

If the command is not found, the folder pip installs scripts into is not on your PATH. Use `python -m mcmeg_hopfield` (on Windows: `py -m mcmeg_hopfield`) or use pipx.

Update later with `pipx upgrade mcmeg-hopfield`; remove with `pipx uninstall mcmeg-hopfield`.

On Linux, if Qt fails to start with an "xcb" error: `sudo apt install libgl1 libxcb-cursor0 libxcb-xinerama0`.

Results are written to `output_method1/` and `output_method2/` in the folder you launch the program from.

### From a clone (development)

```bash
git clone https://github.com/lsbeloti/Neural-Network.git
cd Neural-Network
pip install -e .
```

## Input format

A CSV whose first column is depth (cm) and whose other columns are one curve each, named by field size (`10x10`), SSD (`SSD 100`) or energy (`6MV`). The separator (`;` or `,`) is detected automatically and the decimal mark is a comma or a dot. See [`examples/synthetic_example.csv`](examples/synthetic_example.csv) (synthetic data, not a real measurement).

## Building the executable yourself

The Windows `.exe` is built automatically by GitHub Actions (`.github/workflows/build.yml`):

- **Release:** push a tag such as `v0.1.0` (`git tag v0.1.0 && git push origin v0.1.0`) and the Windows `.exe` and a Linux package are attached to a new release.
- **Without a release:** open the repo's **Actions** tab, pick **Build**, click **Run workflow**, then download the files from the finished run under *Artifacts*.

To build locally (PyInstaller only builds for the OS it runs on, so build on Windows to get a `.exe`):

```bash
pip install ".[build]"
pyinstaller --noconfirm mcmeg_hopfield.spec
```

The app is created in `dist/MCMEG_Hopfield/`. Environment variables for the spec: `MCMEG_ONEFILE=1` (single file, which is what the Windows release uses) and `MCMEG_CONSOLE=1` (keep the console for debugging).

## Self-test

```bash
mcmeg-hopfield --selftest
```

Runs headless checks (imports, CSV parsing, fitting, matplotlib, Excel export, clipboard paste) and prints `SELFTEST OK`.

## Layout

```
src/mcmeg_hopfield/
  common/     CSV I/O, kernel matrix, Mayneord correction
  direct/     (Method 1) fitting, A/B surfaces, reconstruction, plotting
  indirect/   (Method 2) Hopfield/Fourier kerma, kerma models, reconstruction, plotting
  gui.py      PyQt5 interface
examples/     synthetic example CSV
```

## License

MIT, see [LICENSE](LICENSE).
