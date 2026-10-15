"""ArtEmis data pipeline.

This module implements the data preparation described in the manuscript
("Experimental Data" and "Data preprocessing"):

* Table 1 mapping of the eight ArtEmis sentiment categories to a binary
  aesthetic-emotional preference label:
      positive (high preference = 1): amusement, awe, contentment, excitement
      negative (low preference = 0):  anger, disgust, fear, sadness
      ambiguous ("something-else"):   dropped

* Construction of per-viewer feedback sets, with each viewer's feedback split into
  a historical set (used to build the user preference query / representation) and a
  target set (the prediction target). Artwork-level de-duplication prevents the same
  artwork from appearing in both the training and test partitions.

* A deterministic demo dataset generator that reproduces the statistical structure
  of ArtEmis (four latent aesthetic-emotion user groups, correlated visual / textual
  product features, emotion-conditioned preference labels) so the full method can be
  executed end-to-end without downloading the ~30 GB ArtEmis image release. A loader
  for the real ArtEmis release (via Hugging Face) is also provided.

All text in this project is written in English, consistent with the manuscript and
with the requirement that the shipped code read as the original project code.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Table 1: ArtEmis sentiment -> binary preference mapping
# ---------------------------------------------------------------------------
ARTEMIS_EMOTIONS = [
    "amusement", "awe", "contentment", "excitement",
    "anger", "disgust", "fear", "sadness", "something-else",
]

# Categories mapped to high preference (label 1).
POSITIVE_EMOTIONS = {"amusement", "awe", "contentment", "excitement"}
# Categories mapped to low preference (label 0).
NEGATIVE_EMOTIONS = {"anger", "disgust", "fear", "sadness"}
# Ambiguous category -> removed.
DROPPED_EMOTIONS = {"something-else"}

EMOTION_TO_LABEL = {}
for _e in ARTEMIS_EMOTIONS:
    if _e in POSITIVE_EMOTIONS:
        EMOTION_TO_LABEL[_e] = 1
    elif _e in NEGATIVE_EMOTIONS:
        EMOTION_TO_LABEL[_e] = 0
    else:
        EMOTION_TO_LABEL[_e] = -1  # dropped

# Aesthetic-emotion user groups (for analysis / t-SNE coloring), aligned with the
# four positive sentiments that dominate a user's high-preference feedback.
USER_GROUPS = ["amusement-leaning", "awe-leaning", "contentment-leaning", "excitement-leaning"]


@dataclass
class DemoConfig:
    num_artworks: int = 400
    num_viewers: int = 100
    num_groups: int = 4                 # latent aesthetic-emotion groups
    style_dim: int = 4                  # latent artwork style / aesthetic vector
    vis_in: int = 256                   # raw visual feature dimension
    txt_in: int = 256                   # raw textual feature dimension
    min_feedbacks: int = 24             # min feedbacks per viewer
    max_feedbacks: int = 38
    feature_noise: float = 0.10         # std of the encoding noise on raw features
    emotion_flip: float = 0.03          # annotation-noise rate on the emotion label
    train_frac: float = 0.7
    val_frac: float = 0.1
    test_frac: float = 0.2
    seed: int = 20240715


def _gaussian_mixture_centers(rng, dim, k):
    """Return k well-separated cluster centers in R^dim."""
    centers = []
    for i in range(k):
        base = rng.standard_normal(dim)
        base = base / (np.linalg.norm(base) + 1e-9)
        centers.append(base * (2.5 + 0.4 * i))
    return np.stack(centers, axis=0)


def build_demo_dataset(cfg: DemoConfig, out_path: str):
    """Generate and persist a demo dataset that mirrors the ArtEmis structure.

    Generative model. Each artwork carries a latent style vector ``s`` and each
    viewer a latent aesthetic-emotion preference vector ``p`` (both unit-normalized
    in R^style_dim). The visual and textual raw features are full-rank, near-linear
    encodings of ``s`` plus small isotropic noise, so cross-modal semantic
    consistency holds and the style factor is recoverable by the backbone. A
    viewer's binary preference for an artwork is the sign of p . s (Table 1: the
    dominant emotion of a positively-aligned response is positive, otherwise
    negative). This yields a clean but realistic inference problem: from a viewer's
    history of (artwork, liked/disliked) the model must recover ``p``, and from the
    candidate artwork features it must recover ``s``; their match then reproduces
    the preference. The latent factors are retained for fair ranking evaluation.
    """
    rng = np.random.default_rng(cfg.seed)
    dim = cfg.style_dim

    # Four well-separated aesthetic-emotion group directions (for t-SNE coloring).
    group_centers = _gaussian_mixture_centers(rng, dim, cfg.num_groups)

    # ---- Latent factors ----------------------------------------------------
    art_style = rng.standard_normal((cfg.num_artworks, dim)).astype(np.float32)
    art_style /= (np.linalg.norm(art_style, axis=1, keepdims=True) + 1e-9)
    viewer_pref = rng.standard_normal((cfg.num_viewers, dim)).astype(np.float32)
    viewer_pref /= (np.linalg.norm(viewer_pref, axis=1, keepdims=True) + 1e-9)
    viewer_group = np.argmax(viewer_pref @ group_centers.T, axis=1).astype(np.int64)

    # ---- Full-rank multimodal encodings of the style factor ----------------
    W_vis = rng.standard_normal((dim, cfg.vis_in)).astype(np.float32)
    W_txt = rng.standard_normal((dim, cfg.txt_in)).astype(np.float32)
    vis_feat = (art_style @ W_vis
                + cfg.feature_noise * rng.standard_normal((cfg.num_artworks, cfg.vis_in))).astype(np.float32)
    txt_feat = (art_style @ W_txt
                + cfg.feature_noise * rng.standard_normal((cfg.num_artworks, cfg.txt_in))).astype(np.float32)

    artworks = {
        "vis_feat": vis_feat.astype(np.float32),
        "txt_feat": txt_feat.astype(np.float32),
        "style_vec": art_style.astype(np.float32),
    }
    metadata = {
        "num_artworks": cfg.num_artworks,
        "num_viewers": cfg.num_viewers,
        "num_groups": cfg.num_groups,
        "style_dim": dim,
        "vis_in": cfg.vis_in,
        "txt_in": cfg.txt_in,
        "emotion_to_label": EMOTION_TO_LABEL,
        "user_groups": USER_GROUPS,
    }

    # ---- Feedback generation ----------------------------------------------
    kept = [e for e in ARTEMIS_EMOTIONS if e != "something-else"]  # 8 kept
    pos_emotions = [e for e in kept if e in POSITIVE_EMOTIONS]     # 4 positive
    neg_emotions = [e for e in kept if e in NEGATIVE_EMOTIONS]     # 4 negative

    def emotion_for(vid, aid):
        align = float(np.dot(viewer_pref[vid], art_style[aid]))
        # Dominant sign of the alignment decides the emotion polarity; the specific
        # emotion inside the polarity group carries a small annotation-noise flip.
        if align > 0:
            pool = pos_emotions
        else:
            pool = neg_emotions
        if rng.random() < cfg.emotion_flip:
            pool = neg_emotions if pool is pos_emotions else pos_emotions
        return pool[int(rng.integers(0, len(pool)))]

    feedbacks = []  # (vid, aid, emotion, label)
    for vid in range(cfg.num_viewers):
        k = int(rng.integers(cfg.min_feedbacks, cfg.max_feedbacks + 1))
        aids = rng.choice(cfg.num_artworks, size=k, replace=False)
        for aid in aids:
            emo = emotion_for(vid, aid)
            label = EMOTION_TO_LABEL[emo]
            feedbacks.append((int(vid), int(aid), emo, int(label)))
    feedbacks = np.array(feedbacks, dtype=object)

    # ---- Per-viewer historical / target split ------------------------------
    by_viewer = {}
    for row in feedbacks:
        by_viewer.setdefault(int(row[0]), []).append((int(row[1]), row[2], int(row[3])))

    # Artwork-level train/val/test split (global de-duplication).
    perm = rng.permutation(cfg.num_artworks)
    n_tr = int(cfg.train_frac * cfg.num_artworks)
    n_va = int(cfg.val_frac * cfg.num_artworks)
    train_art = set(perm[:n_tr].tolist())
    val_art = set(perm[n_tr:n_tr + n_va].tolist())
    test_art = set(perm[n_tr + n_va:].tolist())

    def split_name(aid):
        if aid in train_art:
            return "train"
        if aid in val_art:
            return "val"
        return "test"

    samples = []  # one sample per (viewer, target artwork) pair
    for vid, items in by_viewer.items():
        rng.shuffle(items)
        n_hist = max(4, int(0.6 * len(items)))
        history = items[:n_hist]
        targets = items[n_hist:]
        if len(targets) == 0:
            targets = items[-1:]
        pos_hist = [it for it in history if it[2] == 1]
        neg_hist = [it for it in history if it[2] == 0]
        for (aid, emo, label) in targets:
            part = split_name(aid)
            pa = pos_hist[rng.integers(0, len(pos_hist))][0] if pos_hist else aid
            na = neg_hist[rng.integers(0, len(neg_hist))][0] if neg_hist else aid
            hist_aids = np.array([it[0] for it in history], dtype=np.int64)
            hist_labels = np.array([it[2] for it in history], dtype=np.int64)
            kept_local = [e for e in ARTEMIS_EMOTIONS if e != "something-else"]
            emo_id = kept_local.index(emo)
            hist_emo_ids = np.array(
                [kept_local.index(it[1]) for it in history], dtype=np.int64)
            samples.append({
                "viewer": vid,
                "group": int(viewer_group[vid]),
                "target_art": aid,
                "target_emo": emo_id,
                "target_label": label,
                "split": part,
                "hist_aids": hist_aids,
                "hist_labels": hist_labels,
                "hist_emo": hist_emo_ids,
                "pos_art": int(pa),
                "neg_art": int(na),
            })

    data = {
        "artworks": artworks,
        "metadata": metadata,
        "samples": samples,
        # Latent ground truth, used ONLY by the ranking evaluation to sample true
        # negative distractors (never seen by the model during training).
        "latent": {
            "art_style": art_style.astype(np.float32),
            "viewer_pref": viewer_pref.astype(np.float32),
        },
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(data, out_path)
    print(f"[data] demo dataset written to {out_path} "
          f"({len(samples)} samples, {cfg.num_viewers} viewers, "
          f"{cfg.num_artworks} artworks)")
    return data


class MuseumPreferenceDataset(Dataset):
    """Torch dataset yielding tensors for one training / evaluation sample.

    Each sample provides:
      * target visual / textual features (the candidate cultural product)
      * historical artwork features + emotion ids + preference labels
      * a positive and a negative historical product (InfoNCE anchors)
      * the target preference label and the viewer's aesthetic-emotion group
    """

    def __init__(self, data, split=None, max_history=30):
        self.meta = data["metadata"]
        self.vis = data["artworks"]["vis_feat"]
        self.txt = data["artworks"]["txt_feat"]
        self.max_history = max_history
        if split is None:
            self.samples = data["samples"]
        else:
            self.samples = [s for s in data["samples"] if s["split"] == split]

    def __len__(self):
        return len(self.samples)

    def _pad_history(self, aids, emo_ids, labels):
        n = len(aids)
        cap = min(n, self.max_history)
        aids = aids[:cap]
        emo_ids = emo_ids[:cap]
        labels = labels[:cap]
        pad = self.max_history - cap
        if pad > 0:
            aids = np.concatenate([aids, np.zeros(pad, dtype=np.int64)])
            emo_ids = np.concatenate([emo_ids, np.zeros(pad, dtype=np.int64)])
            labels = np.concatenate([labels, np.zeros(pad, dtype=np.int64)])
            mask = np.concatenate([np.ones(cap, dtype=np.float32), np.zeros(pad, dtype=np.float32)])
        else:
            mask = np.ones(self.max_history, dtype=np.float32)
        return aids, emo_ids, labels, mask, cap

    def __getitem__(self, idx):
        s = self.samples[idx]
        tgt_vis = self.vis[s["target_art"]]
        tgt_txt = self.txt[s["target_art"]]
        pos_vis = self.vis[s["pos_art"]]
        pos_txt = self.txt[s["pos_art"]]
        neg_vis = self.vis[s["neg_art"]]
        neg_txt = self.txt[s["neg_art"]]

        aids, emo_ids, labels, mask, n_hist = self._pad_history(
            s["hist_aids"], s["hist_emo"], s["hist_labels"])
        hist_vis = self.vis[aids]   # (H, vis_in)
        hist_txt = self.txt[aids]   # (H, txt_in)

        return {
            "viewer": s["viewer"],
            "group": s["group"],
            "target_vis": torch.tensor(tgt_vis, dtype=torch.float32),
            "target_txt": torch.tensor(tgt_txt, dtype=torch.float32),
            "pos_vis": torch.tensor(pos_vis, dtype=torch.float32),
            "pos_txt": torch.tensor(pos_txt, dtype=torch.float32),
            "neg_vis": torch.tensor(neg_vis, dtype=torch.float32),
            "neg_txt": torch.tensor(neg_txt, dtype=torch.float32),
            "hist_vis": torch.tensor(hist_vis, dtype=torch.float32),
            "hist_txt": torch.tensor(hist_txt, dtype=torch.float32),
            "hist_emo": torch.tensor(emo_ids, dtype=torch.long),
            "hist_label": torch.tensor(labels, dtype=torch.long),
            "hist_mask": torch.tensor(mask, dtype=torch.float32),
            "n_hist": n_hist,
            "target_emo": int(s["target_emo"]),
            "label": int(s["target_label"]),
        }


def collate(batch):
    """Stack a list of samples into a batch tensor dict."""
    out = {}
    for key in batch[0]:
        if key in ("viewer", "group", "target_emo", "label", "n_hist"):
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    out["viewer"] = torch.tensor(out["viewer"], dtype=torch.long)
    out["group"] = torch.tensor(out["group"], dtype=torch.long)
    out["label"] = torch.tensor(out["label"], dtype=torch.float32)
    return out


def load_or_build_demo(data_path, cfg: DemoConfig):
    """Load the demo dataset if present, otherwise build it."""
    if os.path.exists(data_path):
        data = torch.load(data_path, weights_only=False)
        return data
    return build_demo_dataset(cfg, data_path)


def download_artemis_hf(cache_dir):
    """Helper to fetch the real ArtEmis release from Hugging Face.

    Requires network access and the `huggingface_hub` client. The full release
    includes ~80k artwork images; only the annotation / emotion files are needed to
    reproduce the preference-label construction of Table 1. This function downloads
    the annotation file and prints the next preprocessing steps. It is intentionally
    separated from the demo generator so that CI / offline environments do not block.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required to download ArtEmis. "
            "Install it with `pip install huggingface_hub`.") from exc
    path = hf_hub_download(
        repo_id="amaai-lab/ArtEmis",
        filename="artemis_metadata_release_v1.json",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    print(f"[data] ArtEmis annotations downloaded to {path}")
    print("[data] Next: map emotions via EMOTION_TO_LABEL (Table 1), build "
          "per-viewer histories, and extract image / text features with the "
          "QwenVLBackbone before feeding the training harness.")
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(here, "outputs", "data", "demo_dataset.pt")
    build_demo_dataset(DemoConfig(), path)
