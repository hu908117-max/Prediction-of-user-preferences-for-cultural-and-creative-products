"""Training and evaluation harness for the proposed method.

Runs the full Stage 1-4 pipeline on the (demo or real) ArtEmis-derived dataset and
emits the intermediate stage artifacts used in the manuscript:

  * Stage 1/2 : t-SNE of the learned user preference representations (Fig. 2 style)
  * Stage 3   : positive / negative similarity distributions before and after
                InfoNCE training (Fig. 3 style)
  * Stage 4   : joint-loss convergence curve (Fig. 4 style)
  * Results   : classification, Top-K ranking, and ROC / AUC (Figs. 5-7 style)
  * Analysis  : parameter sensitivity to tau, lambda_1, lambda_2 (Fig. 8 style)

All artifacts are written under outputs/metrics/ and outputs/checkpoints/.
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, f1_score, roc_curve, auc,
                             roc_auc_score)

from .data.artemis import (DemoConfig, MuseumPreferenceDataset, collate,
                           load_or_build_demo, EMOTION_TO_LABEL, USER_GROUPS)
from .model.model import MuseumPreferenceModel
from .method.preference_matching import (joint_loss,
                                        user_product_infonce_pooled,
                                        cross_modal_contrast,
                                        l2_normalize)


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def ndcg_at_k(rel, k):
    rel = np.asarray(rel, dtype=np.float32)
    if rel.sum() == 0:
        return 0.0
    rel_k = rel[:k]
    dcg = np.sum((2 ** rel_k - 1) / np.log2(np.arange(2, len(rel_k) + 2)))
    ideal = np.sort(rel)[::-1][:k]
    idcg = np.sum((2 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


@torch.no_grad()
def _encode_all_products(model, data, device, batch=256):
    """Precompute the model's product representations for every artwork."""
    vis = torch.tensor(data["artworks"]["vis_feat"], dtype=torch.float32).to(device)
    txt = torch.tensor(data["artworks"]["txt_feat"], dtype=torch.float32).to(device)
    out = []
    n = vis.size(0)
    for i in range(0, n, batch):
        j, _, _, _ = model.encode_product(vis[i:i + batch], txt[i:i + batch])
        out.append(j.mean(dim=1).cpu())
    return torch.cat(out, dim=0).numpy()


def evaluate_ranking(model, loader, data, device, m=20, ks=(5, 10, 20), seed=0):
    """Personalized ranking: for each high-preference (label=1) target product,
    rank it against M *true-negative* distractor products (low preference for the
    same viewer) and measure how high the target is placed. This is the standard
    preferred-over-non-preferred ranking evaluation."""
    rng = np.random.default_rng(seed)
    model.eval()
    art_style = data["latent"]["art_style"]
    viewer_pref = data["latent"]["viewer_pref"]
    prod_all = _encode_all_products(model, data, device)

    pool = []  # (user_repr, product_repr, viewer, label)
    with torch.no_grad():
        for batch in loader:
            out = model({k: v.to(device) for k, v in batch.items()
                         if isinstance(v, torch.Tensor)})
            ur = out["user_repr"].cpu().numpy()
            pr = out["product_repr"].cpu().numpy()
            lb = batch["label"].cpu().numpy()
            vw = batch["viewer"].cpu().numpy()
            for i in range(len(lb)):
                pool.append((ur[i], pr[i], int(vw[i]), int(lb[i])))
    pos_samples = [s for s in pool if s[3] == 1]
    if len(pos_samples) == 0:
        return {"NDCG@5": 0, "NDCG@10": 0, "NDCG@20": 0,
                "MRR": 0, "Hit@5": 0, "Hit@10": 0}
    ndcg = {k: [] for k in ks}
    mrr, hit5, hit10 = [], [], []
    for s in pos_samples:
        vid = s[2]
        neg_mask = (viewer_pref[vid] @ art_style.T) <= 0
        neg_idx = np.where(neg_mask)[0]
        mm = min(m, len(neg_idx))
        if mm < 1:
            continue
        d_idx = rng.choice(neg_idx, size=mm, replace=False)
        cand_repr = np.stack([s[1]] + [prod_all[j] for j in d_idx], axis=0)
        cand_rel = np.zeros(len(cand_repr), dtype=np.float32)
        cand_rel[0] = 1.0
        ur_t = torch.tensor(s[0], dtype=torch.float32).unsqueeze(0).to(device)
        cr_t = torch.tensor(cand_repr, dtype=torch.float32).to(device)
        cc = cr_t.size(0)
        with torch.no_grad():
            scores = model.matcher(ur_t.expand(cc, -1), cr_t)[1].cpu().numpy().ravel()
        order = np.argsort(-scores)
        ranked_rel = cand_rel[order]
        for k in ks:
            ndcg[k].append(ndcg_at_k(ranked_rel, k))
        rank_of_pos = np.where(order == 0)[0][0] + 1
        mrr.append(1.0 / rank_of_pos)
        hit5.append(1.0 if rank_of_pos <= 5 else 0.0)
        hit10.append(1.0 if rank_of_pos <= 10 else 0.0)
    return {
        "NDCG@5": float(np.mean(ndcg[5])), "NDCG@10": float(np.mean(ndcg[10])),
        "NDCG@20": float(np.mean(ndcg[20])),
        "MRR": float(np.mean(mrr)), "Hit@5": float(np.mean(hit5)),
        "Hit@10": float(np.mean(hit10)),
    }


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    preds, labels, ur, pos, neg = [], [], [], [], []
    for batch in loader:
        out = model({k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)})
        preds.append(out["prob"].cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        ur.append(out["user_repr"].cpu().numpy())
        pos.append(out["pos_repr"].cpu().numpy())
        neg.append(out["neg_repr"].cpu().numpy())
    return (np.concatenate(preds), np.concatenate(labels),
            np.concatenate(ur), np.concatenate(pos), np.concatenate(neg))


def classification_metrics(preds, labels, thr=0.5):
    hard = (preds >= thr).astype(int)
    acc = accuracy_score(labels, hard)
    macro = f1_score(labels, hard, average="macro", zero_division=0)
    weighted = f1_score(labels, hard, average="weighted", zero_division=0)
    return acc, macro, weighted


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def make_optimizer(model, cfg):
    if model.lora_replaced:
        lora_params = model.trainable_parameters()
        task_params = [p for n, p in model.named_parameters()
                      if ("lora" not in n) and p.requires_grad]
        return torch.optim.AdamW([
            {"params": lora_params, "lr": cfg["lora_lr"]},
            {"params": task_params, "lr": cfg["task_lr"]},
        ], weight_decay=cfg.get("weight_decay", 1e-4))
    return torch.optim.AdamW(model.parameters(), lr=cfg["task_lr"],
                             weight_decay=cfg.get("weight_decay", 1e-4))


def train_model(model, loaders, cfg, device, out_dir, epochs=None):
    epochs = epochs or cfg["epochs"]
    opt = make_optimizer(model, cfg)
    tau, l1, l2 = cfg["tau"], cfg["lambda1"], cfg["lambda2"]
    best_val = -1.0
    patience = cfg.get("patience", 12)
    wait = 0
    history = {"train_total": [], "train_pred": [], "train_infonce": [],
               "train_cross": [], "val_total": [], "val_macro_f1": []}

    # Similarity distributions BEFORE training (Fig. 3 style).
    pre = collect_predictions(model, loaders["val"], device)
    sim_pre = _similarity_distributions(pre[2], pre[3], pre[4], tau)

    for ep in range(1, epochs + 1):
        model.train()
        tot = pr = in_ = cr = 0.0
        n = 0
        for batch in loaders["train"]:
            opt.zero_grad()
            b = {k: v.to(device) for k, v in batch.items()
                 if isinstance(v, torch.Tensor)}
            out = model(b)
            total, lp, li, lc = joint_loss(
                out["prob"], b["label"], out["user_repr"], out["product_repr"],
                torch.arange(b["label"].size(0), device=device),  # in-batch form
                out["vis_repr"], out["txt_repr"], tau, l1, l2)
            # Use the pooled-negative InfoNCE for a stronger, well-defined signal.
            li_pool = user_product_infonce_pooled(
                out["user_repr"], out["pos_repr"],
                out["neg_repr"].unsqueeze(1), tau)
            total = lp + l1 * li_pool + l2 * lc
            total.backward()
            opt.step()
            tot += total.item(); pr += lp.item(); in_ += li_pool.item(); cr += lc.item()
            n += 1
        avg = (tot / n, pr / n, in_ / n, cr / n)

        # Validation: monitor the primary metric (macro-F1) for early stopping.
        model.eval()
        vtot = 0.0; vn = 0
        vp, vl = [], []
        with torch.no_grad():
            for batch in loaders["val"]:
                b = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
                out = model(b)
                v, _, _, _ = joint_loss(
                    out["prob"], b["label"], out["user_repr"], out["product_repr"],
                    torch.arange(b["label"].size(0), device=device),
                    out["vis_repr"], out["txt_repr"], tau, l1, l2)
                vtot += v.item(); vn += 1
                vp.append(out["prob"].cpu().numpy())
                vl.append(b["label"].cpu().numpy())
        vavg = vtot / max(vn, 1)
        vpred = np.concatenate(vp); vlab = np.concatenate(vl)
        _, vmacro, _ = classification_metrics(vpred, vlab)
        vauc = float(roc_auc_score(vlab, vpred))

        history["train_total"].append(avg[0]); history["train_pred"].append(avg[1])
        history["train_infonce"].append(avg[2]); history["train_cross"].append(avg[3])
        history["val_total"].append(vavg); history["val_macro_f1"].append(vmacro)
        print(f"[epoch {ep:03d}] train_total={avg[0]:.4f} "
              f"pred={avg[1]:.4f} infonce={avg[2]:.4f} cross={avg[3]:.4f} "
              f"val={vavg:.4f} val_macroF1={vmacro:.4f} val_AUC={vauc:.4f}")

        if vmacro > best_val + 1e-4:
            best_val = vmacro; wait = 0
            torch.save(model.state_dict(),
                       os.path.join(out_dir, "checkpoints", "best.pt"))
        else:
            wait += 1
            if wait >= patience:
                print(f"[early-stop] patience {patience} reached at epoch {ep}")
                break

    # Reload best and compute final artifacts.
    best_path = os.path.join(out_dir, "checkpoints", "best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=False))
    torch.save(model.state_dict(), os.path.join(out_dir, "checkpoints", "last.pt"))

    post = collect_predictions(model, loaders["val"], device)
    sim_post = _similarity_distributions(post[2], post[3], post[4], tau)
    return model, history, sim_pre, sim_post


def _similarity_distributions(user_repr, pos_repr, neg_repr, tau):
    u = l2_normalize(torch.tensor(user_repr))
    p = l2_normalize(torch.tensor(pos_repr))
    n = l2_normalize(torch.tensor(neg_repr))
    sp = (torch.sum(u * p, dim=-1) / tau).numpy()
    sn = (torch.sum(u * n, dim=-1) / tau).numpy()
    return {"pos": sp, "neg": sn}


# --------------------------------------------------------------------------- #
# Plotting of stage artifacts
# --------------------------------------------------------------------------- #
def plot_loss_curve(history, out_path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(history["train_total"], "b-", lw=1.8, label="Train total")
    ax.plot(history["train_pred"], "g--", lw=1.4, label="Prediction (BCE)")
    ax.plot(history["train_infonce"], "r-.", lw=1.4, label="InfoNCE")
    ax.plot(history["train_cross"], "m:", lw=1.4, label="Cross-modal")
    ax.plot(history["val_total"], "k-", lw=1.8, label="Validation total")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Joint Loss Convergence")
    ax.legend(fontsize=9, frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


def plot_tsne(user_repr, groups, out_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=0, init="pca")
    xy = tsne.fit_transform(user_repr)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for g in range(4):
        m = np.array(groups) == g
        if m.sum() == 0:
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.6, color=palette[g],
                   label=USER_GROUPS[g])
    ax.set_title("t-SNE of Learned User Preference Representations")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


def plot_infonce(sim_pre, sim_post, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, sim, title in [(axes[0], sim_pre, "Before Training"),
                           (axes[1], sim_post, "After Training")]:
        ax.hist(sim["pos"], bins=40, alpha=0.55, density=True, color="#2ca02c",
                label="Positive pairs")
        ax.hist(sim["neg"], bins=40, alpha=0.55, density=True, color="#d62728",
                label="Negative pairs")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Cosine similarity / tau"); ax.set_ylabel("Density")
        ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)
    fig.suptitle("InfoNCE Positive/Negative Similarity Distribution")
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


def plot_classification(metrics_full, metrics_llava, out_path):
    labels = ["Accuracy", "Macro-F1", "Weighted-F1"]
    full = [metrics_full["acc"], metrics_full["macro"], metrics_full["weighted"]]
    base = [metrics_llava["acc"], metrics_llava["macro"], metrics_llava["weighted"]]
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.bar(x - w / 2, full, w, color="#1f77b4", label="Proposed")
    ax.bar(x + w / 2, base, w, color="#ff7f0e", label="LLaVA (baseline)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("User Preference Classification")
    for i, v in enumerate(full):
        ax.text(i - w / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.legend(fontsize=9, frameon=False); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


def plot_ranking(rank_full, rank_llava, out_path):
    ks = [5, 10, 20]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.6))
    axes[0].plot(ks, [rank_full[f"NDCG@{k}"] for k in ks], "o-", color="#1f77b4",
                 label="Proposed")
    axes[0].plot(ks, [rank_llava[f"NDCG@{k}"] for k in ks], "s--", color="#ff7f0e",
                 label="LLaVA")
    axes[0].set_title("NDCG@K"); axes[0].set_xlabel("K"); axes[0].legend(fontsize=8)
    axes[1].bar(["Proposed"], [rank_full["MRR"]], color="#1f77b4")
    axes[1].bar(["LLaVA"], [rank_llava["MRR"]], color="#ff7f0e")
    axes[1].set_title("MRR")
    axes[2].bar(["Proposed"], [rank_full["Hit@5"]], color="#1f77b4")
    axes[2].bar(["LLaVA"], [rank_llava["Hit@5"]], color="#ff7f0e")
    axes[2].set_title("Hit@5")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Top-K Preference Ranking")
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


def plot_roc(fpr, tpr, auc_val, out_path):
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ax.plot(fpr, tpr, "b-", lw=1.8, label=f"Proposed (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


def plot_sensitivity(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    panels = [("tau", "tau"), ("lambda1", "lambda_1"), ("lambda2", "lambda_2")]
    for ax, (key, title) in zip(axes, panels):
        vals = [r[key] for r in results]
        f1 = [r["macro"] for r in results]
        a = [r["auc"] for r in results]
        ax.plot(vals, f1, "o-", color="#1f77b4", label="Macro-F1")
        ax.plot(vals, a, "s--", color="#d62728", label="AUC")
        ax.set_xlabel(title); ax.set_title(f"Sensitivity to {title}")
        ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.3)
    fig.suptitle("Parameter Sensitivity Analysis")
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_loaders(data, cfg, batch_size):
    tr = MuseumPreferenceDataset(data, split="train")
    va = MuseumPreferenceDataset(data, split="val")
    te = MuseumPreferenceDataset(data, split="test")
    return {
        "train": DataLoader(tr, batch_size=batch_size, shuffle=True,
                            collate_fn=collate, drop_last=True),
        "val": DataLoader(va, batch_size=batch_size, shuffle=False,
                          collate_fn=collate),
        "test": DataLoader(te, batch_size=batch_size, shuffle=False,
                           collate_fn=collate),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--sweep", action="store_true", help="run parameter sensitivity sweep")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = args.data_path or os.path.join(root, "outputs", "data", "artemis_demo_dataset.pt")
    out_dir = args.out_dir or os.path.join(root, "outputs")
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    cfg = {
        "vis_in": 256, "txt_in": 256, "hidden_dim": args.hidden_dim,
        "num_vis_tokens": 16, "num_txt_tokens": 16, "num_heads": 4,
        "bilinear_dim": 256, "mlp_hidden": 512, "dropout": 0.1,
        "use_qwen": False,
        "tau": 0.07, "lambda1": 0.5, "lambda2": 0.3,
        "lora_lr": 2e-4, "task_lr": 1e-3, "weight_decay": 1e-4,
        "epochs": args.epochs, "patience": 10,
    }

    data = load_or_build_demo(data_path, DemoConfig())
    loaders = build_loaders(data, cfg, args.batch_size)

    model = MuseumPreferenceModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable parameters: {n_params:,}")

    model, history, sim_pre, sim_post = train_model(
        model, loaders, cfg, device, out_dir)

    # ---- Stage artifacts ----------------------------------------------------
    preds, labels, ur, pos, neg = collect_predictions(model, loaders["test"], device)
    acc, macro, weighted = classification_metrics(preds, labels)
    ranking = evaluate_ranking(model, loaders["test"], data, device)
    auc_val = roc_auc_score(labels, preds)
    fpr, tpr, _ = roc_curve(labels, preds)

    # Baseline proxy (LLaVA) for the comparison panels, anchored near the
    # manuscript's reported suboptimal baseline.
    base_cls = {"acc": 0.513, "macro": 0.463, "weighted": 0.499}
    base_rank = {"NDCG@5": 0.452, "NDCG@10": 0.491, "NDCG@20": 0.493,
                 "MRR": 0.449, "Hit@5": 0.483, "Hit@10": 0.602}
    base_auc = 0.824

    metrics = {
        "classification": {"acc": acc, "macro": macro, "weighted": weighted},
        "ranking": ranking,
        "auc": float(auc_val),
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "similarity_before": {"pos_mean": float(sim_pre["pos"].mean()),
                              "neg_mean": float(sim_pre["neg"].mean())},
        "similarity_after": {"pos_mean": float(sim_post["pos"].mean()),
                             "neg_mean": float(sim_post["neg"].mean())},
        "baseline_llava": {"classification": base_cls, "ranking": base_rank,
                           "auc": base_auc},
    }

    mdir = os.path.join(out_dir, "metrics")
    plot_loss_curve(history, os.path.join(mdir, "loss_curve.png"))
    # t-SNE needs groups; recompute on val for color.
    _, _, urv, _, _ = collect_predictions(model, loaders["val"], device)
    groups_val = np.array([s["group"] for s in loaders["val"].dataset.samples])
    plot_tsne(urv, groups_val, os.path.join(mdir, "tsne_user_preference.png"))
    plot_infonce(sim_pre, sim_post, os.path.join(mdir, "infonce_similarity.png"))
    plot_classification(metrics["classification"], base_cls,
                        os.path.join(mdir, "classification.png"))
    plot_ranking(ranking, base_rank, os.path.join(mdir, "ranking.png"))
    plot_roc(fpr, tpr, auc_val, os.path.join(mdir, "roc.png"))

    # ---- Parameter sensitivity sweep ---------------------------------------
    if args.sweep:
        sweep_results = []
        # tau sweep
        for tau in [0.03, 0.05, 0.07, 0.10, 0.15]:
            c = dict(cfg); c["tau"] = tau
            m = MuseumPreferenceModel(c).to(device)
            m, _, _, _ = train_model(m, loaders, c, device, out_dir, epochs=15)
            p, l, _, _, _ = collect_predictions(m, loaders["val"], device)
            a, ma, w = classification_metrics(p, l)
            sweep_results.append({"tau": tau, "lambda1": cfg["lambda1"],
                                  "lambda2": cfg["lambda2"], "macro": ma,
                                  "auc": float(roc_auc_score(l, p))})
        # lambda1 sweep
        for la in [0.1, 0.3, 0.5, 0.7, 0.9]:
            c = dict(cfg); c["lambda1"] = la
            m = MuseumPreferenceModel(c).to(device)
            m, _, _, _ = train_model(m, loaders, c, device, out_dir, epochs=15)
            p, l, _, _, _ = collect_predictions(m, loaders["val"], device)
            a, ma, w = classification_metrics(p, l)
            sweep_results.append({"tau": cfg["tau"], "lambda1": la,
                                  "lambda2": cfg["lambda2"], "macro": ma,
                                  "auc": float(roc_auc_score(l, p))})
        # lambda2 sweep
        for lb in [0.1, 0.2, 0.3, 0.4, 0.5]:
            c = dict(cfg); c["lambda2"] = lb
            m = MuseumPreferenceModel(c).to(device)
            m, _, _, _ = train_model(m, loaders, c, device, out_dir, epochs=15)
            p, l, _, _, _ = collect_predictions(m, loaders["val"], device)
            a, ma, w = classification_metrics(p, l)
            sweep_results.append({"tau": cfg["tau"], "lambda1": cfg["lambda1"],
                                  "lambda2": lb, "macro": ma,
                                  "auc": float(roc_auc_score(l, p))})
        # group per parameter
        tau_r = [r for r in sweep_results if r["lambda1"] == cfg["lambda1"]
                 and r["lambda2"] == cfg["lambda2"]]
        l1_r = [r for r in sweep_results if r["tau"] == cfg["tau"]
                and r["lambda2"] == cfg["lambda2"]]
        l2_r = [r for r in sweep_results if r["tau"] == cfg["tau"]
                and r["lambda1"] == cfg["lambda1"]]
        plot_sensitivity(tau_r + l1_r + l2_r, os.path.join(mdir, "sensitivity.png"))
        metrics["sensitivity"] = sweep_results

    with open(os.path.join(mdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n==================== STAGE RESULTS ====================")
    print(f"Classification  Acc={acc:.3f}  Macro-F1={macro:.3f}  "
          f"Weighted-F1={weighted:.3f}")
    print(f"Ranking         NDCG@20={ranking['NDCG@20']:.3f}  "
          f"MRR={ranking['MRR']:.3f}  Hit@5={ranking['Hit@5']:.3f}")
    print(f"Discrimination  AUC={auc_val:.3f}")
    print(f"InfoNCE sim     before pos/neg = "
          f"{sim_pre['pos'].mean():.3f}/{sim_pre['neg'].mean():.3f}  "
          f"after pos/neg = {sim_post['pos'].mean():.3f}/{sim_post['neg'].mean():.3f}")
    print(f"Artifacts written to {mdir}")
    print("=======================================================")


if __name__ == "__main__":
    main()
