"""PyInstaller entry point (absolute import so the frozen app resolves the package)."""
import sys

from mcmeg_hopfield.gui import main

if __name__ == "__main__":
    sys.exit(main())
