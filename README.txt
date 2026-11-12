Copy these 5 files into your PDD4 folder, keeping the paths (overwrite when asked):

  indirect/kerma_model_v2.py     NEW
  indirect/kerma_batch.py        modified (stores target curve in kermas)
  indirect/kerma_variation.py    modified (modelo='v2' switch + docstring note)
  indirect/reconstruction.py     modified (modelo='v2' switch)
  main_pyqt.py                   modified ("F(y) model" combo in the indirect reconstruction row)

Do NOT replace indirect/hopfield.py (not part of this change).
Use: Extract kermas (again, with this version) -> choose "F(y) model: V2" -> Generate curve.
The legacy model stays the default.
