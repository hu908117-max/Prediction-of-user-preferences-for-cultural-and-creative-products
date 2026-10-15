# Museum Cultural-Creative Product User-Preference Prediction

A complete, runnable Python project implementing the multimodal large language
model based method for predicting user aesthetic-emotional preferences for museum
cultural-creative products, as described in the accompanying manuscript
(*A Multimodal Large Language Model Based Method for Predicting User Preferences
for Museum Cultural-Creative Products*). The source dataset is **ArtEmis**
(Affective Language for Visual Art).

The project ships two coordinated deliverables:

1. **Publication result figures (Figs. 2–8)** — a faithful Python / matplotlib
   translation of the original MATLAB result-figure code. These are the figures
   reported in the manuscript (`outputs/figures/`).
2. **An executable reproduction of the proposed method** — a PyTorch implementation
   of Stages 1–4 that trains on the ArtEmis-derived dataset and writes the
   intermediate stage artifacts (loss curve, t-SNE, InfoNCE similarity,
   classification, ranking, ROC) together with `metrics.json`
   (`outputs/metrics/`).

---

## 1. Method overview

The proposed pipeline (see manuscript Sections 2–4):

| Stage | Module | Equations | Source file |
|-------|--------|-----------|------------|
| 1 | Multimodal feature extraction + gated fusion | (1)–(8)  | `src/method/gated_fusion.py` |
| 2 | User-preference-guided cross-attention        | (9)–(18) | `src/method/user_preference.py` |
| 3 | Bilinear interaction + InfoNCE matching       | (19)–(27)| `src/method/preference_matching.py` |
| 4 | Joint loss + LoRA-efficient fine-tuning       | (28)–(30)| `src/method/preference_matching.py`, `src/method/lora.py` |

* **Stage 1** projects region-level visual tokens and word-level textual tokens
  from the backbone into a shared space, performs bidirectional cross-modal
  attention, pools each modality, learns a dynamic gating vector, and produces the
  gated joint product representation.
* **Stage 2** builds the individual user preference representation from the
  viewer's historical feedback: each feedback is encoded from its emotion label and
  interpretation-text vector (Eq. 9–10), attended to the historical product tokens
  through preference-guided cross-attention (Eq. 11–16), and aggregated with
  learnable importance weights (Eq. 17–18).
* **Stage 3** maps the user and product representations into a matching space
  (Eq. 19), models second-order feature cross-links with a low-rank bilinear
  interaction (Eq. 20), passes `[user; product; bilinear]` through an MLP
  (Eq. 21–22) and a sigmoid (Eq. 23) to obtain the preference probability. A
  user-conditional InfoNCE loss (Eq. 24–26) and a symmetric visual–textual
  contrastive loss (Eq. 27) strengthen preference discrimination and cross-modal
  consistency.
* **Stage 4** combines a binary cross-entropy prediction loss (Eq. 28) with the two
  contrastive losses into a joint objective (Eq. 29) and fine-tunes the Qwen2.5-VL
  backbone with LoRA (Eq. 30) while keeping the visual encoder frozen.

## 2. Project structure

```
museum_preference_project/
├── run_all.py                 # one-command runner (figures + demo training)
├── run_figures.py             # generate publication result figures (Figs. 2-8)
├── requirements.txt
├── configs/default.yaml
├── src/
│   ├── data/artemis.py        # ArtEmis Table-1 mapping, demo dataset, loaders
│   ├── model/
│   │   ├── backbone.py        # Qwen2.5-VL backbone + lightweight stand-in
│   │   └── model.py           # end-to-end MuseumPreferenceModel
│   ├── method/
│   │   ├── gated_fusion.py    # Stage 1 (Eqs. 1-8)
│   │   ├── user_preference.py # Stage 2 (Eqs. 9-18)
│   │   ├── preference_matching.py # Stage 3-4 (Eqs. 19-29)
│   │   └── lora.py           # LoRA (Eq. 30)
│   ├── figures/               # Figs. 2-8 (MATLAB -> Python)
│   └── train.py               # training / evaluation harness + stage artifacts
├── outputs/
│   ├── figures/               # publication figures (Fig_*.png)
│   ├── metrics/               # demo-run stage artifacts + metrics.json
│   ├── checkpoints/           # best.pt / last.pt
│   └── data/                  # demo_dataset.pt (generated)
└── results/                   # (reserved for exported tables / extra reports)
```

## 3. Data

ArtEmis provides artwork images, titles, artists, styles, types, and viewer
emotion tags with interpretation texts. The eight emotion categories are mapped to
a binary aesthetic-emotional preference label through **Table 1**:

| Preference | ArtEmis emotions |
|------------|-----------------|
| High (1)   | amusement, awe, contentment, excitement |
| Low  (0)   | anger, disgust, fear, sadness |
| Dropped    | something-else (ambiguous) |

Per the manuscript, viewers are grouped into historical and target feedback sets,
and artwork-level de-duplication prevents the same artwork from appearing in both
training and test partitions.

> **Demo dataset.** Downloading the ~30 GB ArtEmis image release is optional. The
> project includes a deterministic generator (`src/data/artemis.py`,
> `build_demo_dataset`) that reproduces the statistical structure of ArtEmis (four
> latent aesthetic-emotion user groups, correlated visual / textual product
> features, emotion-conditioned preference labels) so the full method can be
> executed end-to-end on CPU. A helper to fetch the real ArtEmis annotation file
> via Hugging Face is provided as `download_artemis_hf`.

## 4. Installation

```bash
cd museum_preference_project
pip install -r requirements.txt
```

To use the real Qwen2.5-VL-7B-Instruct backbone you additionally need
`transformers`, `qwen-vl-utils`, and the model checkpoint, and you must run on a
CUDA GPU. Set `model.backbone: qwen2.5-vl` and `model.use_qwen: true` in
`configs/default.yaml` (or pass them through the training entry point).

## 5. Running

Generate **everything** (publication figures + demo training):

```bash
python run_all.py                # CPU-friendly demo (hidden_dim 256)
python run_all.py --epochs 45    # longer demo training
python run_all.py --sweep        # also run the parameter-sensitivity sweep
```

Generate only the **publication figures** (Figs. 2–8):

```bash
python run_figures.py
```

Run only the **method reproduction** (training + stage artifacts):

```bash
python -m src.train --epochs 45 --hidden_dim 256 --batch_size 64
```

The first run builds `outputs/data/demo_dataset.pt` automatically.

## 6. Outputs

* `outputs/figures/Fig_*.png` — the seven manuscript result figures.
* `outputs/metrics/metrics.json` — classification (Accuracy / Macro-F1 /
  Weighted-F1), Top-K ranking (NDCG@K, MRR, Hit@K), ROC / AUC, and the
  InfoNCE positive/negative similarity means (before / after training).
* `outputs/metrics/*.png` — the corresponding stage artifacts computed from the
  actual run.
* `outputs/checkpoints/best.pt`, `last.pt` — trained model weights.

## 7. Reproducibility notes

* All randomness is seeded (`configs/default.yaml` → `data.seed` and the training
  harness). Setting the same seed reproduces the demo dataset and the run.
* The publication figures use the reported-experiment numbers from the manuscript
  (the MATLAB result code translated to Python). The demo training produces its
  own run-specific metrics and artifacts, which are close to but not identical to
  the reported values because the demo uses a compact backbone and a generated
  ArtEmis-structured dataset rather than the full 7B model on the raw release.
* Hyperparameters follow the manuscript: τ = 0.07, λ₁ = 0.5, λ₂ = 0.3, LoRA rank
  8, AdamW with LoRA lr 2e-4 and task-module lr 1e-3, batch size 64, early
  stopping patience 10.
