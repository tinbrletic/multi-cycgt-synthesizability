# Running the Synthesizability Adaptation

This branch adapts Multi_CycGT's `gcn_transformer_fc` variant to predict linear
peptide synthesizability instead of cyclic peptide membrane permeability. The
code edits are done; this file walks through the env setup and training run.

## What's already done on this branch

- `data_process/prepare_synthesis_data.py` — converts `peptide_baza.csv`
  (semicolon-delimited, target_col 0/1) into Multi_CycGT's expected schema.
  Tested under Python 3.13 with `py -m pytest data_process/tests/`.
- `data_process/data_processing.py` — patched to read the adapted CSV and skip
  the upstream Permeability >= -6 binarization (our `label` column is already 0/1).
- `model/deep_learning/gcn_transformer_fc/models.py` — `nn.Linear(103, 128)`
  changed to `nn.Linear(25, 128)` to match our descriptor count (25 numeric
  features after dropping the 7 metadata columns).
- `model/deep_learning/gcn_transformer_fc/model_concat.py` — added
  `WeightedRandomSampler` for 87/13 class imbalance and per-fold AUC/F1/accuracy
  logging to `results_per_fold.csv`.

## What you still need to do

The training pipeline depends on `dgl==1.1.0` + `dgllife==0.3.2` + the patched
`gcn_predictor.py`. Neither has a wheel for Python 3.13. The smoke test was
therefore not run on the host that wrote this branch.

### 1. Set up a Python 3.10 venv

`dgl 1.1.0` has wheels for Python 3.7/3.8/3.9/3.10/3.11 on Windows. Pick 3.10
because it matches the upstream Linux/docker reference (`replace.sh` hardcodes
`python3.9`, but 3.10 site-packages layout is identical and the patch script
is trivially adaptable).

```powershell
# install python 3.10 if you don't already have it (winget):
winget install --id Python.Python.3.10 -e

# create + activate venv
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -V    # should print Python 3.10.x

# install deps in the order torch -> dgl -> dgllife -> rest, because each
# layer pins specific versions of the one below
pip install --upgrade pip
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install dgl==1.1.0 -f https://data.dgl.ai/wheels/repo.html
pip install dgllife==0.3.2
pip install rdkit==2023.9.5 pandas==2.0.2 scikit-learn==1.2.2 numpy==1.24.3
```

If you have CUDA 11.7 / 11.8 hardware, replace the torch + dgl lines with the
CUDA wheels — see the upstream README. Multi_CycGT's `Model_TGCN.forward()`
calls `.to("cuda")` unconditionally inside the model, so **CPU-only training
will crash** until those calls are guarded. Easiest path is to actually have
a GPU; second-easiest is to grep `to\("cuda"\)` and `.cuda\(\)` in
`model/deep_learning/gcn_transformer_fc/` and replace with a `device` constant.

### 2. Apply the dgllife monkey-patch

The repo ships `gcn_predictor.py` at the root — a modified copy of
`dgllife.model.model_zoo.gcn_predictor.GCNPredictor` that returns the
intermediate `graph_feats` when called with `model_use='a'`. Every
training script depends on this; without it the model errors on the first
GCN forward pass.

The upstream `replace.sh` is Linux + Python 3.9 only. On Windows / Python 3.10:

```powershell
# DGL prints a one-shot backend-selection notice to stdout on first import.
# If you do not set this, that notice pollutes any $variable that captures the
# next `python -c` output and Copy-Item will fail with a "drive not found" error.
$env:DGLBACKEND = 'pytorch'

$site = python -c "import dgllife, os; print(os.path.dirname(dgllife.__file__))"
Write-Output "site=$site"   # eyeball: should be a clean path, not multi-line text
Copy-Item gcn_predictor.py "$site\model\model_zoo\gcn_predictor.py" -Force
```

To make the env var permanent across future shells:

```powershell
[Environment]::SetEnvironmentVariable('DGLBACKEND', 'pytorch', 'User')
```

Verify the patch by running the helper script:

```powershell
python verify_patch.py
```

Should print `PATCH OK`. (Windows PowerShell 5.1 strips quotes when forwarding
to native exes; a `python -c "..."` one-liner with embedded quotes is
unreliable. The helper script sidesteps the issue.)

### 3. Regenerate the splits in your env

The splits in `data/data_splitClassifier/` were generated on the host that
wrote this branch and are already correct, but rebuilding under the venv
verifies the env is wired up:

```powershell
cd data_process
python data_processing.py
cd ..
```

Expected output: ten `experiment_{n}_data_save` lines, no errors. 60 files
appear in `data/data_splitClassifier/`.

### 4. Smoke-test (recommended before the full run)

Edit `model/deep_learning/gcn_transformer_fc/model_concat.py`:

- line `for num in range(1, 11):` → `for num in range(1, 2):` (1 fold)
- line `for epoch in range(1, 501):` → `for epoch in range(1, 3):` (2 epochs)

Run:

```powershell
cd model/deep_learning/gcn_transformer_fc
python model_concat.py
```

Expected: 2 training epochs print, prediction CSVs appear under
`pred_data_origin/gcn_transformer_fc/1/{test,val}/`, then one row appears in
`results_per_fold.csv`. Metric values will be garbage (2 epochs is nothing)
but **no errors** = the pipeline is wired correctly.

Common smoke-test failure modes:

| Symptom | Likely cause |
|---|---|
| `forward() got unexpected keyword argument 'model_use'` | Step 2 patch not applied or not picked up — check `inspect.getsource` |
| `mat1 and mat2 shapes cannot be multiplied (128x25 and 103x128)` | You forgot the `Linear(103,128) -> Linear(25,128)` patch in `models.py` (this branch has it; if you reverted, re-apply) |
| `RuntimeError: CUDA error: no kernel image is available...` | torch/dgl built for a CUDA version your driver doesn't support — try CPU wheels |
| `KeyError: 'label'` in `create_dataset_number` | You're reading the upstream CSV instead of the adapted one — re-run step 3 |

### 5. Full run

Revert the smoke-test edits (`range(1, 11)` and `range(1, 501)`) and run from
the same directory:

```powershell
cd model/deep_learning/gcn_transformer_fc
python model_concat.py 2>&1 | Tee-Object -FilePath ..\..\..\training_log.txt
```

Estimated runtime: 30 min – several hours on CPU (full 10 folds × 500 epochs
× ~11 train batches/epoch). On a 6 GB+ consumer GPU each fold should finish
in a few minutes.

Per-epoch checkpoints land in `model_origin/`, per-epoch predictions in
`pred_data_origin/`, and per-fold metrics in `results_per_fold.csv`.

### 6. Aggregate the results

```powershell
python -c "import pandas as pd; df = pd.read_csv('model/deep_learning/gcn_transformer_fc/results_per_fold.csv'); print(df.to_string(index=False)); print(); print('Mean AUC:', round(df.test_auc.mean(), 4), '+/-', round(df.test_auc.std(), 4)); print('Mean F1:', round(df.test_f1.mean(), 4), '+/-', round(df.test_f1.std(), 4)); print('Mean ACC:', round(df.test_accuracy.mean(), 4), '+/-', round(df.test_accuracy.std(), 4))"
```

Drop those numbers into `model/deep_learning/gcn_transformer_fc/comparison.md`
(template on this branch) alongside the UOOP-Project classical baselines.

## Known caveats

- **Test set truncation.** `model_concat.py`'s `collate` reshapes the SMILES
  tokens to `[batch_size, 128]`, which requires `drop_last=True` on every
  loader. With ~177 test samples per fold and `batch_size=128`, only 128 are
  evaluated (last ~28% dropped). The `results_per_fold.csv` records both
  `n_test_eval` and `n_test_total` so this is visible. The proper fix is to
  rewrite `collate` to handle variable batch sizes — non-trivial because the
  fused-head `fc1 = Linear(batch_size*128 + 40 + 128, 1)` is also coupled to
  `batch_size`. Out of scope for this adaptation.
- **Cross-task transfer caveat.** Multi_CycGT was trained on cyclic peptides
  predicting membrane permeability; we're using the same architecture for
  linear peptides predicting synthesizability. The architecture's design
  biases (graph + sequence + descriptors) may not transfer 1:1. A result of
  AUC 0.6–0.7 would be a defensible academic finding, not a bug.
- **CV-split mismatch with UOOP baselines.** Multi_CycGT uses
  `KFold(n_splits=10, shuffle=True, random_state=3407)`. UOOP-Project uses
  `RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)`. Fold
  scores are not paired across the two pipelines — compare only at the
  mean ± std level. Paired comparison would need a separate, larger change to
  make Multi_CycGT consume external split files.
- **SMOTE vs WeightedRandomSampler.** UOOP-Project uses real SMOTE (synthetic
  minority samples). This branch uses `WeightedRandomSampler` (oversamples
  existing minority samples). The latter is weaker — no new feature-space
  coverage — and may overfit faster on the few minority examples.
