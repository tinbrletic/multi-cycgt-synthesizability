# Onboarding Presentation — Peptide Synthesizability Prediction

> Slide-by-slide content for onboarding new contributors. Focus: **the dataset and
> its features**, then implementation and results. Each slide is one `###` section.
> `[FIGURE: ...]` lines tell you which image to drop on that slide.
>
> Sources: [`METHODOLOGY.md`](METHODOLOGY.md),
> [`../model/deep_learning/gcn_transformer_fc/comparison.md`](../model/deep_learning/gcn_transformer_fc/comparison.md),
> and the generated figures.

---

## SECTION 0 — Introduction

---

### Slide 1 — Title

**Predicting Peptide Synthesizability**
From handcrafted features to multimodal deep learning

A comparison of two approaches on the same dataset:
- **UOOP-Project** — classical machine learning on engineered features
- **Multi_CycGT** — a repurposed multimodal deep-learning model

Onboarding deck · Focus: the dataset and its features

---

### Slide 2 — The problem we are solving

- **Goal:** given a peptide's amino-acid sequence, predict whether it can be **successfully synthesized** in the lab.
- **Why it matters:** failed syntheses waste time, reagents, and money. A reliable predictor lets chemists **prioritize peptides** that are likely to succeed before touching the bench.
- **Task type:** binary classification.
  - `label = 1` → synthesis **succeeded**
  - `label = 0` → synthesis **failed**
- **The scientific question of this project:** does a heavy, modern **multimodal deep-learning** model beat well-tuned **classical ML** on this task — or just match it?

---

### Slide 3 — Two projects, one shared dataset

| | UOOP-Project | Multi_CycGT |
|---|---|---|
| Approach | Classical ML (6 algorithms) | Multimodal deep learning |
| Features | ~233 engineered tabular features | 25 descriptors + graph + SMILES |
| Origin | Built for this task | **Repurposed** from a cyclic-peptide permeability paper (Cao et al. 2024) |
| Role here | The **baseline** to beat | The **experiment** |

- Both train and evaluate on the **exact same peptides** → a fair head-to-head.
- Multi_CycGT is a **transfer experiment**: same architecture as the original permeability paper, but new chemistry (linear peptides) and new target (synthesis success).

---

### Slide 4 — What "repurposing" means (Multi_CycGT)

- Original Multi_CycGT predicted **membrane permeability of cyclic peptides**.
- We keep the **architecture unchanged** but swap:
  - **Input chemistry:** cyclic → **linear** peptides
  - **Prediction target:** permeability → **synthesizability**
- This tests whether the model's rich representation **transfers** to a new biochemical question, or whether its built-in assumptions ("inductive biases") are too specialized to help.

---

## SECTION 1 — The Dataset (main focus)

---

### Slide 5 — Dataset at a glance

- **Source file:** `data_process/peptide_baza.csv` (semicolon-delimited)
- **Origin:** linear peptides, Gutman et al. (2022)
- **Size:** **1771 peptides**, one row per peptide
- **Label:** `target_col` (binary) → renamed to `label`
- **Class balance:** **1539 positive / 232 negative ≈ 87% / 13%**
  - Strongly **imbalanced** — most peptides in the dataset *can* be synthesized.
  - This single fact drives many design decisions later (sampling, metrics).

---

### Slide 6 — One row = one peptide (the raw table)

`peptide_baza.csv` has **30 columns**. Conceptually four blocks:

1. **Identity** — `id`, `peptide_seq` (the amino-acid string), `peptide_len`
2. **Raw flag** — `synthesis_flag` (TRUE/FALSE, the human-readable label)
3. **25 physicochemical descriptors** — the scientific "fingerprint" of the peptide (next slide)
4. **Target** — `target_col` (the 0/1 learning label)

Example sequence: `SELLTPLGIDLDEW` (length 14), `synthesis_flag = TRUE`, `target_col = 1`.

> This same raw table is the **starting point for both projects.** Everything else is feature engineering on top of it.

---

### Slide 7 — The 25 physicochemical descriptors (the shared core)

These are computed from the sequence and describe its chemistry. Grouped by what they capture:

- **Length:** `peptide_len`
- **Hydrophobicity** (7 different scales — how "water-avoiding" the peptide is): `hydrophobic_janin`, `hydrophobic_engleman`, `hydrophobic_kyte-doolittle`, `hydrophobic_hopp-woods`, `hydrophobic_eisenberg`, `hydrophobic_roseman`, `hydrophobic_moment`
- **Structure / stability:** `aliphatic_index`, `instability_index`, `boman` (protein-binding potential)
- **Charge:** `isoelectric_point`, `charge`
- **Amino-acid group counts** (composition): `tiny_group`, `small_group`, `aliphatic_group`, `aromatic_group`, `non-polar_group`, `polar_group`, `charged_group`, `basic_group`, `acidic_group`
- **Cruciani properties** (3 latent descriptors): `cruciani_prp1`, `cruciani_prp2`, `cruciani_prp3`

> Note: `hydrophobic_cornette` exists in the raw file but is **constant (0.0)** for every peptide → zero information → dropped. So the usable descriptor count is 25.

---

### Slide 8 — UOOP widens the table: sequence-position features

UOOP does **not** stop at the 25 descriptors. Its model input (`peptide_baza_formatted.csv`) is **470 columns**, adding two big families of one-hot "is this amino acid here?" features:

- **`X4_*` — position-4 identity (20 columns):** `X4_A`, `X4_C`, … `X4_Y` — which of the 20 amino acids sits at sequence position 4.
- **`X5_*` — position-5 identity (20 columns):** same idea for position 5.
- **`X8_*` — dipeptide composition (400 columns):** `X8_AA`, `X8_AC`, … `X8_YY` — counts/indicators for every possible **amino-acid pair** (20 × 20 = 400).

Why these? Synthesis problems often come from **specific residues in specific places** and from **hard-to-couple neighboring pairs** — exactly what position and dipeptide features encode.

---

### Slide 9 — Feature-space summary (the key comparison)

| Feature family | Columns | Used by UOOP | Used by Multi_CycGT |
|---|---|---|---|
| 25 physicochemical descriptors | 25 | ✅ | ✅ (FC branch) |
| Position-4 one-hot (`X4_*`) | 20 | ✅ | ❌ |
| Position-5 one-hot (`X5_*`) | 20 | ✅ | ❌ |
| Dipeptide composition (`X8_*`) | 400 | ✅ | ❌ |
| Molecular graph (from SMILES) | — | ❌ | ✅ (GCN branch) |
| Tokenized SMILES sequence | — | ❌ | ✅ (Transformer branch) |

- **UOOP:** ~466 raw engineered columns → **≈233 features kept** after dropping all-zero / constant columns (the dipeptide matrix is very sparse for short peptides).
- **Multi_CycGT:** only the **25 descriptors as tabular input**, but **two extra structural views** of the same molecule.

> Takeaway: UOOP grows the table **wider**; Multi_CycGT goes **deeper** on representation. Neither is given less information — that's what makes the comparison fair.

---

### Slide 10 — How Multi_CycGT "sees" a peptide: three views

Multi_CycGT turns each peptide into **three parallel representations**, then fuses them:

1. **Graph view** — the sequence → RDKit molecule → **molecular graph** (atoms + bonds), encoded by a Graph Convolutional Network (GCN).
2. **Sequence view** — the molecule → **SMILES string** → tokenized → encoded by a **Transformer** (the same family of model behind modern language models).
3. **Descriptor view** — the **25 physicochemical numbers** → a small fully-connected network.

A caveat worth knowing: the graph and SMILES are both **derived from the sequence**, so they re-encode the *same* underlying information in different shapes. The truly independent signal is still the 25 descriptors.

---

### Slide 11 — Handling the 87/13 imbalance (no cheating)

Because 87% of peptides are positive, a lazy "always say success" model already looks ~86% accurate. Both projects correct for this **without leaking** test information:

- **UOOP:** **SMOTE** — synthesizes new minority-class examples, applied **inside each training fold only**.
- **Multi_CycGT:** **WeightedRandomSampler** — during training, rare (negative) peptides are drawn more often so each batch is roughly balanced.

Critically, in **both** cases the correction touches the **training data only**. Validation and test sets keep the real 87/13 distribution, so reported scores reflect reality.

---

### Slide 12 — How the data is split (no leakage)

**10-fold cross-validation** with an inner validation split (Multi_CycGT):

- The 1771 peptides are split into **10 folds**. Each fold = disjoint **train (~81%) / validation (~9%) / test (~10%)**.
- **Within a fold:** the three sets never overlap → no train/test leakage.
- **Across all 10 folds:** every peptide is tested **exactly once** and trained on the other nine times → the whole dataset is used for both, **never simultaneously**.
- Reported score = **mean ± std across the 10 held-out test scores**.
- Seed `3407` everywhere → reproducible splits.

> One-line answer to "do you train and test on the same data?": *No — within any fold they're disjoint; across folds every peptide is held out once.*

---

## SECTION 2 — Implementation

---

### Slide 13 — UOOP implementation (classical ML)

- **6 classifiers:** Random Forest, SVM, Naive Bayes, Logistic Regression, K-Nearest Neighbors, Decision Tree.
- **Feature selection per classifier** — tried multiple strategies and kept the best:
  - Statistical filters: Chi-square, Kruskal-Wallis, Kolmogorov-Smirnov, Mann-Whitney, Wilcoxon (all *inside* CV)
  - Model-based selectors: L1-Logistic-Regression, RFECV, Random-Forest importance
  - `All` = no selection (use every feature)
- **Validation:** `RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)` → 100 fits per setting (tight error bars).
- **Imbalance:** SMOTE inside each CV training fold.
- **Statistical rigor:** Friedman + Nemenyi + pairwise Wilcoxon-Holm / Conover-Holm tests to check whether classifier differences are *real* or noise.

---

### Slide 14 — Multi_CycGT implementation (multimodal late fusion)

Three encoders run in parallel, then their outputs are **concatenated before a single sigmoid classifier** ("late fusion"):

| Branch | Input | Encoder | Output size |
|---|---|---|---|
| Graph | Molecular graph of the SMILES | DGL-Life `GCNPredictor` | 40 |
| Sequence | Tokenized SMILES (len 128) | Transformer (2 layers, 8 heads) | 128×128 |
| Descriptors | 25 numeric features | `Linear(25,128) → BatchNorm` | 128 |

- **Fusion:** concatenate all three → `Linear(16552, 1)` → **Sigmoid** → probability of synthesis success.
- **Note:** the batch size (128) is **wired into** the fused layer's dimensions — a quirk inherited from the original paper.

---

### Slide 15 — Multi_CycGT training protocol

- **Optimizer:** Adam, learning rate 1e-3 · **Loss:** Binary Cross-Entropy on the sigmoid output.
- **Imbalance:** WeightedRandomSampler on the training loader only.
- **Early stopping (key improvement):** monitors **validation loss**, **patience = 50** epochs, keeps the last (stop-epoch) weights, with a 500-epoch safety ceiling.
  - Before early stopping the model overfit hard (ran all 500 epochs, test AUC decayed to ~0.66).
  - With it, every fold stopped at **epoch 52–68** (~7–8× less compute) and recovered the lost performance automatically.
- **Per epoch:** writes test/val predictions to CSV → lets us recompute any metric and draw the result curves afterward.
- **Device:** CUDA required (RTX 3050 Ti; torch 2.0.1+cu117).

---

## SECTION 3 — Results

---

### Slide 16 — How we measure success

- **Primary metric: ROC-AUC** — threshold-independent and robust to the 87/13 imbalance. It measures **ranking quality** (does the model rank a synthesizable peptide above a non-synthesizable one?).
- **Why not accuracy / F1?** On 87% positives, "always predict success" already scores ~0.86 accuracy / ~0.93 F1 — those metrics flatter a useless model. AUC is the **apples-to-apples** number.
- We also report F1 and accuracy for completeness, but **AUC is the headline**.

---

### Slide 17 — Headline result: parity, not a win

| Model | Mean AUC ± std |
|---|---|
| **Random Forest** (UOOP best) | **0.764 ± 0.058** |
| **Multi_CycGT** (early-stopped) | **0.757 ± 0.052** |
| Naive Bayes (UOOP) | 0.756 ± 0.058 |
| Logistic Regression (UOOP) | 0.749 ± 0.055 |
| SVM (UOOP) | 0.734 ± 0.060 |
| KNN (UOOP) | 0.732 ± 0.060 |
| Decision Tree (UOOP) | 0.726 ± 0.051 |
| Majority-class baseline | 0.500 |

- Multi_CycGT (0.757) is **statistically indistinguishable** from the best classical model, Random Forest (0.764).
- **The rich multimodal model reaches parity — it does not beat handcrafted features at this dataset size (N=1771).**

---

### Slide 18 — Multi_CycGT: training dynamics (loss)

`[FIGURE: model/deep_learning/gcn_transformer_fc/figures/loss_curves.png]`

- 2×5 grid — one panel per fold. Lines: **train / validation / test loss** vs epoch; red dashed line = early-stopping cut-off.
- **What to see:** train loss keeps falling while validation/test loss bottom out early then **rise** → classic **overfitting**.
- Early stopping cuts each fold off shortly after the validation-loss minimum, before the damage grows.

---

### Slide 19 — Multi_CycGT: AUC over time

`[FIGURE: model/deep_learning/gcn_transformer_fc/figures/auc_curves.png]`

- 2×5 grid — **validation and test AUC** vs epoch per fold.
- Markers: purple = best-validation-loss epoch, red = stop epoch.
- **What to see:** test **AUC** stays high even as **loss** rises — because AUC measures *ranking*, which is robust to the model becoming overconfident (a *calibration* problem, not a ranking one).
- This is why early stopping on loss still produces a strong AUC model.

---

### Slide 20 — Multi_CycGT: ROC curves

`[FIGURE: model/deep_learning/gcn_transformer_fc/figures/roc_curves.png]`

- One panel: 10 thin per-fold ROC curves + a bold **mean ROC** + the diagonal "chance" line.
- Pooled AUC ≈ **0.751** (across the 1280 evaluated test samples).
- Each curve bows above the diagonal → the model ranks better than chance on every fold, with fold-to-fold spread that matches the ±0.052 std.

---

### Slide 21 — UOOP: per-model ROC figures

`[FIGURE: UOOP-Project/results/smote_in_cv/20260523_144512/roc_curves_all_models_20260523_144512.png]` — master overlay of all classifiers.

Per-classifier ROC figures (each overlays that model's feature-selection variants):
- `roc_group_StatSelect_Random_Forest_20260523_144512.png`
- `roc_group_StatSelect_SVM_20260523_144512.png`
- `roc_group_StatSelect_Naive_Bayes_20260523_144512.png`
- `roc_group_StatSelect_Logistic_Regression_20260523_144512.png`
- `roc_group_StatSelect_K_Neighbors_20260523_144512.png`
- `roc_group_StatSelect_Decision_Tree_20260523_144512.png`

(All in `UOOP-Project/results/smote_in_cv/20260523_144512/`.)

- **What to see:** all classifiers cluster in the **0.73–0.76 AUC** band — the same band Multi_CycGT lands in.

---

### Slide 22 — Takeaways

1. **Same dataset, two philosophies:** UOOP widens the feature table (position + dipeptide one-hots, ~233 features); Multi_CycGT keeps 25 descriptors but adds graph + sequence views.
2. **Result: parity.** Multi_CycGT AUC **0.757** ≈ Random Forest **0.764** — a statistical tie. The heavy model does **not** beat well-engineered classical features here.
3. **Why no uplift?** Most likely (a) **small data** (1771 samples) favors classical models, (b) **inductive-bias mismatch** — the architecture was built for permeability, not synthesis, (c) lightly-regularized large fusion head.
4. **Overfitting was the main failure mode** — early stopping contained it (~7–8× less compute, AUC recovered from 0.66 → 0.757).
5. **Open improvement:** calibration is still poor (predictions are overconfident); for risk-scoring you'd add temperature/Platt scaling.
