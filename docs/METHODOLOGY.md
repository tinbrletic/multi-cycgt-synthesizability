# Methodology — Applying Multi_CycGT to Linear-Peptide Synthesizability

> **Purpose of this document.** A full, self-contained rundown of *what problem we
> are solving*, *what data we use*, **exactly how the data is divided for training
> and evaluation**, how the model is trained, and how results are scored. It is
> written so that the data-splitting question ("do you use the whole dataset for
> both training and testing?") can be answered precisely, with the actual code as
> the source of truth.
>
> Companion files: [`../SETUP_AND_RUN.md`](../SETUP_AND_RUN.md) (how to run),
> [`../model/deep_learning/gcn_transformer_fc/comparison.md`](../model/deep_learning/gcn_transformer_fc/comparison.md)
> (results vs. classical baselines).

---

## 1. The problem

**Goal.** Take **Multi_CycGT** — a multimodal deep-learning model originally
designed to predict *membrane permeability of cyclic peptides* — and test how
well it predicts **synthesizability of linear peptides** (i.e. whether a given
peptide sequence can be successfully synthesized).

This is a **transfer / repurposing experiment**: the architecture stays the same,
but the input chemistry (linear vs. cyclic peptides) and the prediction target
(synthesis success vs. permeability) both change. The point is to measure whether
the rich multimodal representation buys anything over the classical
feature-based machine-learning baselines from the UOOP project.

**Task type.** Binary classification.
`label = 1` ⇒ synthesis succeeded, `label = 0` ⇒ synthesis failed.

---

## 2. The dataset

| Property | Value |
|---|---|
| Source file | `data_process/peptide_baza.csv` (semicolon-delimited) |
| Origin | Linear peptides (Gutman et al., 2022) |
| Samples | **1771 peptides** (after RDKit SMILES generation; 0 dropped) |
| Label column | `target_col` → renamed to `label` (binary 0/1) |
| Class balance | **1539 positive / 232 negative ≈ 87 % / 13 %** (strong imbalance) |
| Raw columns | 30 (1 id, 1 sequence, 1 length, 1 raw flag, 25 handcrafted descriptors, 1 target) |

The 25 handcrafted descriptors are physicochemical/sequence features:
`peptide_len`, several hydrophobicity scales (`hydrophobic_janin`,
`hydrophobic_engleman`, `hydrophobic_kyte-doolittle`, `hydrophobic_hopp-woods`,
`hydrophobic_eisenberg`, `hydrophobic_roseman`, `hydrophobic_moment`),
`aliphatic_index`, `isoelectric_point`, `charge`, amino-acid group counts
(`tiny_group`, `small_group`, `aliphatic_group`, `aromatic_group`,
`non-polar_group`, `polar_group`, `charged_group`, `basic_group`,
`acidic_group`), `cruciani_prp1..3`, `instability_index`, and `boman`.

> These are **the same handcrafted features** the UOOP classical baselines use,
> so the deep-learning model is not given less information than the baselines —
> it gets these features *plus* two additional structural views (graph and
> sequence). That makes the comparison fair.

---

## 3. From raw CSV to model input (the adapter)

The raw file does not match the column schema Multi_CycGT expects.
`data_process/prepare_synthesis_data.py` performs a one-time conversion:

```
peptide_baza.csv  ──prepare_synthesis_data.py──►  peptide_synthesis_adapted.csv
```

What it does, per peptide:

1. **Generate a molecule** from the one-letter sequence with RDKit
   `Chem.MolFromSequence(seq)` (standard L-amino acids).
2. **Generate SMILES** from that molecule (`Chem.MolToSmiles`) → new `SMILES`
   column. This is what feeds the graph and sequence branches.
3. Compute molecule-level `Sequence_LogP` (Crippen logP) and `Sequence_TPSA`
   (topological polar surface area). *(These two columns are written but, in the
   current pipeline, end up unused — see §5 caveat 4.)*
4. **Rename / add metadata** columns so the schema matches upstream
   (`target_col → label`, `id → CycPeptMPDB_ID`, plus dummy `Year`,
   `Structurally_Unique_ID`).
5. **Drop** `peptide_seq`, `synthesis_flag`, and `hydrophobic_cornette`
   (the last is constant 0.0 across all rows → zero variance → no information).

Result: `peptide_synthesis_adapted.csv` — one row per peptide, with `SMILES`,
`label`, the 25 numeric descriptors, and metadata columns.

---

## 4. Data division for training and evaluation ⟵ **the key section**

This is the precise answer to "how is the data split, and do we use the whole
dataset for both training and testing?" The splitting happens in
`data_process/data_processing.py`.

### 4.1 The scheme: 10-fold cross-validation with an inner validation split

```python
kf = KFold(n_splits=10, shuffle=True, random_state=3407)
for train_index, test_index in kf.split(X_stand):       # outer 10-fold loop
    X_train, X_test = X_stand[train_index], X_stand[test_index]
    X_train, X_val, ... = train_test_split(X_train, ...,  # inner 90/10 split
                                           test_size=0.1, random_state=3407)
```

For **each** of the 10 folds the 1771 peptides are partitioned into **three
disjoint sets**:

| Set | How it is chosen | Role | Size (fold 1) | Size (folds 2–10) |
|---|---|---|---|---|
| **Test** | the held-out 1/10 of the outer KFold | final evaluation; never seen in training | **178** | **177** |
| **Validation** | inner 10 % of the remaining 9/10 | model selection / early-stopping signal; weights are **never** updated on it | **160** | **160** |
| **Train** | inner 90 % of the remaining 9/10 | the only data the optimizer updates weights on | **1433** | **1434** |

(Verified directly from the generated split files:
`X_train1.csv = 1433`, `X_val1.csv = 160`, `X_test1.csv = 178`;
`X_train2.csv = 1434`, `X_val2.csv = 160`, `X_test2.csv = 177`.)

Roughly **81 % train / 9 % validation / 10 % test** within every fold.
These are written to `data/data_splitClassifier/X_{train,test,val}{1..10}.csv`.

### 4.2 The two facts that answer the mentor's question

**(a) Within a single fold, the three sets never overlap.** A peptide that is in
the training set of fold *k* is not in that fold's validation or test set. So
there is **no train/test leakage within a fold** — the test AUC for fold *k* is
measured on peptides the model never trained on.

**(b) Across the 10 folds, the whole dataset is used for both training and
testing — but never at the same time.** That is the whole point of k-fold
cross-validation:

- Each peptide is placed in the **test set of exactly one fold** and in the
  **training/validation set of the other nine**.
- So over the full experiment, *every* peptide is eventually used to train
  (9 times) and to test (once).
- The headline number (e.g. AUC 0.765 ± 0.063) is the **mean ± standard
  deviation across the 10 fold-level test scores**. The ± std is a measure of how
  stable the model is across different train/test partitions.

> **One-sentence answer for your mentor:** *"I use 10-fold cross-validation.
> In each fold the data is split into disjoint train (~81 %), validation (~9 %),
> and test (~10 %) sets, so there is no leakage within a fold. Across the ten
> folds every peptide is tested exactly once and trained on the other nine times,
> so the whole dataset is used for both training and testing, just never
> simultaneously in the same fold. The reported score is the mean ± std of the
> ten held-out test scores."*

### 4.3 Reproducibility

`shuffle=True, random_state=3407` for the outer KFold and `random_state=3407` for
the inner split make the partition deterministic — re-running
`data_processing.py` reproduces the identical splits. The seed `3407` is used
throughout the project (`torch.manual_seed(3407)` as well).

### 4.4 Class imbalance is handled at *training time only* (no leakage)

The 87/13 imbalance is corrected with a **`WeightedRandomSampler`** applied to the
**training DataLoader only** (`model_concat.py`). Each class is sampled with
probability ∝ `1 / class_count`, so a training batch is roughly class-balanced.
Crucially:

- The sampler touches **only the training set** — validation and test loaders use
  `shuffle=False` and are left at the true 87/13 distribution. So the imbalance
  correction does **not** leak into evaluation and the reported metrics reflect
  real-world class proportions.
- This is preferable to oversampling the whole dataset *before* splitting (which
  would copy the same peptide into both train and test and inflate scores).

---

## 5. Important caveats about the split / evaluation (be ready to discuss these)

These are real properties of the current pipeline. None of them invalidate the
experiment, but an examiner may ask, so state them up front.

1. **`drop_last=True` silently discards part of each evaluation set.** Every
   DataLoader uses a fixed `batch_size = 128` with `drop_last=True`, because the
   fusion head hard-codes `reshape([128, 128])` (the batch size is wired into the
   layer dimensions). Consequence: a test set of 177–178 peptides is evaluated on
   only the **first 128** (1 full batch); the remaining ~50 (~28 %) are dropped.
   Validation (160 → 128) and training (1433 → 1408) are truncated the same way.
   So per-fold test metrics are computed on 128 samples, not the full ~177.
   *Fixing this requires decoupling `batch_size` from the model dimensions.*

2. **Standardization is fit per file, not fit-on-train-applied-to-test.**
   `create_dataset_number` calls `sklearn.scale()` independently on each of
   train/test/val. This is **not leakage** (test statistics do not flow into
   train), but it is a methodological inconsistency: the test set is standardized
   using its *own* mean/std rather than the training set's. The cleaner practice
   is to fit a `StandardScaler` on train and `transform` val/test with it.

3. **The SMILES vocabulary is rebuilt per file.** `create_dataset_seq` writes a
   fresh `vocab.txt` for each of train/test/val, so a SMILES token can map to a
   different integer ID across the three sets. Because the transformer branch
   consumes those integer IDs (no shared embedding table is persisted across
   splits), the encodings are not perfectly aligned between train and test.
   (Inherited from the upstream code.)

4. **The graph + sequence branches are derived from the sequence, not
   independent of it.** `SMILES` is generated deterministically from the
   peptide sequence, so the GCN (graph) and Transformer (sequence) branches are
   alternative encodings of the *same* sequence information. The truly
   independent signal is the 25 handcrafted descriptors in the FC branch. The
   per-molecule `Sequence_LogP`/`Sequence_TPSA` columns are produced by the
   adapter but **not consumed** in this pipeline (`create_dataset_list` is never
   called); the descriptor branch uses the 25 numeric features instead.

---

## 6. The model (multimodal late fusion)

The same architecture as upstream Multi_CycGT — three independent encoders whose
outputs are concatenated just before a sigmoid classifier
(`model/deep_learning/gcn_transformer_fc/`).

| Branch | Input | Encoder | Output width |
|---|---|---|---|
| **Graph** | RDKit complete graph of the generated SMILES, canonical atom features | DGL-Life `GCNPredictor` (`hidden_feats=[60,20]`), returning the intermediate `graph_feats` | 40 |
| **Sequence** | SMILES tokenized → integer IDs, padded to length 128 | `Transformer_test` (2 encoder layers, 8 heads, dim 128) | 128×128 |
| **Descriptors** | the 25 handcrafted numeric features (standardized) | `Linear(25, 128) → BatchNorm1d` | 128 |

**Fusion** (`Model_TGCN.forward`): concatenate the three branch outputs along the
last dimension and pass through `fc1 = Linear(batch_size*128 + 40 + 128, 1)` →
`Sigmoid` → probability of synthesis success. With `batch_size=128` the fused
input width is `128*128 + 40 + 128 = 16552`. (This is why `batch_size` is wired
into the layer dimensions and `drop_last=True` is mandatory — see caveat 1.)

> **Note on the "GCN" patch.** `gcn_predictor.py` at the repo root is a modified
> copy of the DGL-Life `GCNPredictor` whose `forward` returns the intermediate
> `graph_feats` (size 40) when called with `model_use='a'`, instead of the final
> prediction. `replace.sh` installs it into site-packages. Without the patch the
> training script errors. (See `CLAUDE.md` and `SETUP_AND_RUN.md`.)

---

## 7. Training protocol

Defined in `model/deep_learning/gcn_transformer_fc/model_concat.py`, run once per
fold (`for num in range(1, 11)`), training a fresh model from scratch each time.

| Setting | Value |
|---|---|
| Epochs | **500** (fixed; **no early stopping currently** — see §9) |
| Optimizer | Adam, learning rate `1e-3`, over all three branches jointly |
| Loss | `BCELoss` on the sigmoid output |
| Batch size | 128 (`drop_last=True`) |
| Imbalance | `WeightedRandomSampler` on train loader only |
| Device | CUDA required (parts of the forward pass call `.cuda()` directly) |
| Per-epoch outputs | test/val predictions → `pred_data_origin/gcn_transformer_fc/<fold>/{test,val}/experiment_<epoch>_predicted_*.csv` |
| Per-fold summary | final-epoch metrics appended to `results_per_fold.csv` |

Each epoch evaluates on both the validation and test loaders and writes the raw
predictions, so metrics at any epoch can be recomputed afterward (this is what
makes the post-hoc "best validation epoch" analysis possible).

---

## 8. Evaluation and metrics

**Metrics** (`sklearn`): **ROC-AUC** (primary — threshold-independent, robust to
the 87/13 imbalance), plus **F1** and **accuracy** at the 0.5 threshold.

Because the dataset is 87 % positive, accuracy and F1 are misleading on their own
(a "always predict success" baseline already scores ~0.86 accuracy / ~0.93 F1).
**AUC is the apples-to-apples number** for comparing against the classical
baselines.

### Two reported numbers — final-epoch vs. best-validation-epoch

Training runs a fixed 500 epochs with no early stopping, but test AUC for most
folds **peaks early (epoch ~6–22) and then degrades** as the model overfits
(train loss falls to ~0.21 while val loss climbs to 2.5–3.7). We therefore report
two aggregations:

| Aggregation | What it is | Headline AUC |
|---|---|---|
| **Best-validation-epoch** | for each fold, pick the epoch with the highest **validation** AUC, then report that epoch's **test** AUC. Legitimate because validation labels never updated weights — equivalent to having run early stopping. | **0.765 ± 0.063** |
| **Final epoch (500)** | the model actually produced at the end of training; knowably overfit. Reported for transparency. | 0.661 ± 0.089 |

The post-hoc "best-val-epoch" selection is done by `analyze_best_epoch.py`
(writes `results_per_fold_best.csv`); the final-epoch numbers come straight out of
`results_per_fold.csv`. Full per-fold tables and the baseline comparison are in
[`comparison.md`](../model/deep_learning/gcn_transformer_fc/comparison.md).

**Bottom line of the experiment so far:** at best-val-epoch the multimodal model
reaches **AUC 0.765**, statistically indistinguishable from the best classical
baseline (Random Forest, AUC 0.764). The rich multimodal representation achieves
**parity, not a win**, at this dataset size (N = 1771) — consistent with a small
dataset, an inductive-bias mismatch (the architecture was designed for
permeability, not synthesizability), and aggressive overfitting from the
unregularized 500-epoch schedule.

---

## 9. Known limitation that motivates the next steps

The ~0.10 AUC gap between the final epoch (0.661) and the best-validation epoch
(0.765) is **pure overfitting**: 8 of 10 folds have already peaked by epoch ~22,
yet training continues to epoch 500 and the test AUC decays. Right now the "early
stopping" is done *after the fact* in analysis, not *during* training, so the
model that gets saved is the bad one.

**Improvements:**

1. **Early stopping during training — implemented.** A small `EarlyStopping`
   helper (`gcn_transformer_fc/early_stopping.py`) now monitors **validation
   loss** and halts a fold after **50 epochs** without improvement; the
   **last (stop-epoch)** weights are kept (no best-weight restore), with the
   500-epoch loop retained only as a safety ceiling. Per-fold results now also
   log `best_val_loss_epoch` / `best_val_loss` so the keep-last gap is visible.
   On a realistic overfit curve (val-loss minimum ~epoch 16) this stops near
   epoch 66 instead of running to 500. See the design spec at
   [`superpowers/specs/2026-06-12-early-stopping-design.md`](superpowers/specs/2026-06-12-early-stopping-design.md).
   Note the accepted trade-off (§8): val-loss-min and val-AUC-max are usually
   different epochs, so the per-epoch prediction CSVs are still written and
   `analyze_best_epoch.py` remains available for the AUC-optimal report.
2. **Training-curve and ROC visualizations — planned (next task).** Plot
   train/val/test loss and AUC vs. epoch (to *see* the overfitting crossover)
   and ROC curves per fold (to visualize the 0.765 AUC), so the behavior is
   inspectable rather than inferred from CSVs.

These directly address the observation that "some folds do better than the final
epoch": the final epoch was suboptimal *by construction*, and early stopping
(plus the planned plots) makes that visible and fixable.

---

## 10. End-to-end reproduction (commands)

```powershell
# 0. (one time) generate model input from the raw peptide CSV
cd data_process
python prepare_synthesis_data.py     # peptide_baza.csv -> peptide_synthesis_adapted.csv

# 1. generate the 10-fold train/val/test splits
python data_processing.py            # -> data/data_splitClassifier/X_{train,val,test}{1..10}.csv

# 2. train + evaluate all 10 folds (CUDA + patched dgllife required)
cd ../model/deep_learning/gcn_transformer_fc
python model_concat.py               # -> results_per_fold.csv + pred_data_origin/...

# 3. post-hoc best-validation-epoch aggregation
python ../../../analyze_best_epoch.py  # -> results_per_fold_best.csv
```

See [`../SETUP_AND_RUN.md`](../SETUP_AND_RUN.md) for environment setup (CUDA,
torch 2.0.1+cu117, dgl 1.1.0, the dgllife patch via `replace.sh`/`verify_patch.py`).
