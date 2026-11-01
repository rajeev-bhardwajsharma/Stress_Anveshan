"""
Regenerate every reported number in the paper from the saved LOSO outputs.

    python reproduce/paper_results.py            # tables + checks
    python reproduce/paper_results.py --figures  # also writes Figs. 2-5

Inputs  : results/dl_<model>_predictions.csv  (one row per test window per fold)
          results/dl_<model>_results_detailed.csv (one row per fold)
Outputs : reproduce/output/paper_results.md, paper_results.json, fig*.png

Each number is compared with the value printed in the paper; the script exits
with status 1 if anything differs by more than the rounding tolerance.

No training happens here. To regenerate the CSVs themselves, run
notebooks/01_cnn_lstm_loso.ipynb (see README, "Full reproduction").
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "output"

THRESHOLD = 0.5
CHEST_BUNDLES = ["chest_acc", "chest_eda", "chest_emg", "chest_heart", "chest_statistical"]
WRIST_BUNDLES = ["wrist_acc", "wrist_eda", "wrist_heart", "wrist_statistical"]

# Values as printed in the paper: (value, absolute tolerance)
PAPER = {
    "tableI.chest.acc": (93.59, 0.006), "tableI.chest.std": (9.92, 0.006),
    "tableI.chest.f1": (0.8855, 0.00006), "tableI.chest.worst": (66.67, 0.006),
    "tableI.wrist.acc": (92.06, 0.006), "tableI.wrist.std": (7.07, 0.006),
    "tableI.wrist.f1": (0.8893, 0.00006), "tableI.wrist.worst": (76.56, 0.006),
    "tableI.fused.acc": (95.34, 0.006), "tableI.fused.std": (7.34, 0.006),
    "tableI.fused.f1": (0.9196, 0.00006), "tableI.fused.worst": (71.43, 0.006),
    "tableII.cnn.acc": (89.9, 0.06), "tableII.cnn.prec": (0.913, 0.0006),
    "tableII.cnn.rec": (0.877, 0.0006), "tableII.cnn.f1": (0.853, 0.0006),
    "tableII.cnn_lstm.acc": (89.75, 0.006), "tableII.cnn_lstm.prec": (0.896, 0.0006),
    "tableII.cnn_lstm.rec": (0.892, 0.0006), "tableII.cnn_lstm.f1": (0.850, 0.0006),
    "tableIII.acc": (0.9538, 0.00006), "tableIII.tn": (601, 0), "tableIII.fp": (12, 0),
    "tableIII.fn": (33, 0), "tableIII.tp": (329, 0),
    "tableIII.stress.prec": (0.9648, 0.00006), "tableIII.stress.rec": (0.9088, 0.00006),
    "tableIII.stress.f1": (0.9360, 0.00006), "tableIII.macro_f1": (0.9500, 0.00006),
    "tableIII.weighted_f1": (0.9535, 0.00006),
    "fused.f1_std": (0.1618, 0.00006), "fused.S2_f1": (0.357, 0.0006),
    "cnn_ensemble.acc": (95.35, 0.006), "cnn_ensemble.f1": (0.937, 0.0006),
    "cnn_ensemble.f1_std": (0.0745, 0.00006), "cnn_ensemble.rec": (0.930, 0.0006), "fused.rec": (0.907, 0.0006),
    "cnn_ensemble.fp": (20, 0), "cnn_ensemble.fn": (25, 0),
    "wilcoxon.acc_p": (0.72, 0.006), "wilcoxon.f1_p": (0.77, 0.006),
    "nine_model.acc": (95.35, 0.006), "nine_model.f1": (0.93, 0.006),
}


# ── Loading ──────────────────────────────────────────────────────────────────
def load_predictions(model: str) -> pd.DataFrame:
    d = pd.read_csv(RESULTS / f"dl_{model}_predictions.csv")
    assert not d.duplicated(["experiment", "subject", "window_idx"]).any(), "duplicate rows"
    return d


def load_fold_metrics(model: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"dl_{model}_results_detailed.csv")


# ── Fusion: Eq. (1), probability averaging ───────────────────────────────────
def fuse(pred: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    """Average y_prob over `members` for every (subject, window). All members
    were trained inside the same LOSO fold, so no test subject leaks."""
    sub = pred[pred.experiment.isin(members)]
    wide = sub.pivot_table(index=["subject", "window_idx"], columns="experiment", values="y_prob")
    missing = set(members) - set(wide.columns)
    assert not missing, f"missing experiments: {missing}"
    y = sub.drop_duplicates(["subject", "window_idx"]).set_index(["subject", "window_idx"]).y_true
    out = pd.DataFrame({"y_prob": wide[members].mean(axis=1)}).join(y)
    assert out.notna().all().all(), "members cover different windows"
    out["y_pred"] = (out.y_prob >= THRESHOLD).astype(int)
    return out.reset_index()


def per_fold(f: pd.DataFrame) -> pd.DataFrame:
    rows = [dict(subject=s,
                 acc=accuracy_score(g.y_true, g.y_pred),
                 f1=f1_score(g.y_true, g.y_pred, zero_division=0),
                 rec=recall_score(g.y_true, g.y_pred, zero_division=0))
            for s, g in f.groupby("subject")]
    return pd.DataFrame(rows).set_index("subject").sort_index(key=lambda i: i.str[1:].astype(int))


def summary(f: pd.DataFrame) -> dict:
    pf = per_fold(f)
    tn, fp, fn, tp = confusion_matrix(f.y_true, f.y_pred, labels=[0, 1]).ravel()
    return dict(acc=pf.acc.mean() * 100, std=pf.acc.std() * 100, f1=pf.f1.mean(),
                f1_std=pf.f1.std(), worst=pf.acc.min() * 100, worst_subject=pf.acc.idxmin(),
                pooled_acc=accuracy_score(f.y_true, f.y_pred),
                rec=pf.rec.mean(), pooled_rec=recall_score(f.y_true, f.y_pred), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                per_fold=pf)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", action="store_true", help="also write Figs. 2-5 as PNG")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    P = {m: load_predictions(m) for m in ("cnn_lstm", "cnn")}
    got: dict[str, float] = {}
    md: list[str] = ["# Reproduced paper results\n",
                     "Generated by `reproduce/paper_results.py` from `results/`.\n"]

    # Table I — sensor-location ablation (CNN-LSTM)
    cfg = {"chest": ["chest_all"], "wrist": ["wrist_all"], "fused": ["chest_all", "wrist_all"]}
    T1 = {k: summary(fuse(P["cnn_lstm"], v)) for k, v in cfg.items()}
    md += ["## Table I — LOSO sensor-location ablation (CNN-LSTM)\n",
           "| Configuration | Mean Acc. | Std. | Mean F1 | Worst-fold Acc. |", "|---|---|---|---|---|"]
    for k, name in [("chest", "Chest-only"), ("wrist", "Wrist-only"), ("fused", "Multimodal (fused)")]:
        s = T1[k]
        md.append(f"| {name} | {s['acc']:.2f}% | {s['std']:.2f}% | {s['f1']:.4f} | {s['worst']:.2f}% ({s['worst_subject']}) |")
        got.update({f"tableI.{k}.acc": s["acc"], f"tableI.{k}.std": s["std"],
                    f"tableI.{k}.f1": s["f1"], f"tableI.{k}.worst": s["worst"]})

    # Table II — architectures on the single all-sensor ('multimodal') model
    md += ["\n## Table II — architectures, single model on all-sensor features\n",
           "| Model | Acc. | Prec. | Recall | F1 |", "|---|---|---|---|---|"]
    for m, name in [("cnn", "CNN"), ("cnn_lstm", "CNN-LSTM")]:
        r = load_fold_metrics(m)
        r = r[r.experiment == "multimodal"][["accuracy", "precision", "recall", "f1"]].mean()
        md.append(f"| {name} | {r.accuracy*100:.2f}% | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} |")
        got.update({f"tableII.{m}.acc": r.accuracy * 100, f"tableII.{m}.prec": r.precision,
                    f"tableII.{m}.rec": r.recall, f"tableII.{m}.f1": r.f1})
    md.append("| LSTM | — | — | — | — |\n\n*LSTM row: its LOSO outputs are not in `results/`, "
              "so it is not regenerated here (see README, Known gaps).*")

    # Table III — pooled report, fused CNN-LSTM
    fused = fuse(P["cnn_lstm"], cfg["fused"])
    rep = classification_report(fused.y_true, fused.y_pred, output_dict=True, digits=4)
    s = T1["fused"]
    md += ["\n## Table III — pooled classification report, fused CNN-LSTM\n",
           "| Class | Prec. | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for k, name in [("0", "Non-stress (0)"), ("1", "Stress (1)"), ("macro avg", "Macro avg."),
                    ("weighted avg", "Weighted avg.")]:
        r = rep[k]
        md.append(f"| {name} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |")
    md.append(f"| Accuracy | | | {rep['accuracy']:.4f} | {len(fused)} |")
    md.append(f"\nConfusion matrix: TN={s['tn']} FP={s['fp']} FN={s['fn']} TP={s['tp']}")
    got.update({"tableIII.acc": rep["accuracy"], "tableIII.tn": s["tn"], "tableIII.fp": s["fp"],
                "tableIII.fn": s["fn"], "tableIII.tp": s["tp"],
                "tableIII.stress.prec": rep["1"]["precision"], "tableIII.stress.rec": rep["1"]["recall"],
                "tableIII.stress.f1": rep["1"]["f1-score"], "tableIII.macro_f1": rep["macro avg"]["f1-score"],
                "tableIII.weighted_f1": rep["weighted avg"]["f1-score"],
                "fused.f1_std": s["f1_std"], "fused.S2_f1": s["per_fold"].loc["S2", "f1"]})

    # Sec. V-C — CNN ensemble and Wilcoxon test
    c = summary(fuse(P["cnn"], cfg["fused"]))
    a, b = c["per_fold"].align(s["per_fold"])
    p_acc, p_f1 = wilcoxon(a.acc, b.acc).pvalue, wilcoxon(a.f1, b.f1).pvalue
    got.update({"cnn_ensemble.acc": c["acc"], "cnn_ensemble.f1": c["f1"], "cnn_ensemble.f1_std": c["f1_std"],
                "cnn_ensemble.rec": c["rec"], "fused.rec": s["rec"], "cnn_ensemble.fp": c["fp"], "cnn_ensemble.fn": c["fn"],
                "wilcoxon.acc_p": p_acc, "wilcoxon.f1_p": p_f1})
    md += ["\n## Sec. V-C — CNN-LSTM vs CNN (fused chest+wrist ensembles)\n",
           "| | CNN-LSTM | CNN |", "|---|---|---|",
           f"| Mean acc. | {s['acc']:.2f}% | {c['acc']:.2f}% |",
           f"| Mean F1 (std) | {s['f1']:.3f} ({s['f1_std']:.4f}) | {c['f1']:.3f} ({c['f1_std']:.4f}) |",
           f"| Stress recall, mean per fold (pooled) | {s['rec']:.3f} ({s['pooled_rec']:.3f}) | {c['rec']:.3f} ({c['pooled_rec']:.3f}) |",
           f"| FP / FN | {s['fp']} / {s['fn']} | {c['fp']} / {c['fn']} |",
           f"\nPaired Wilcoxon signed-rank over 15 folds: accuracy p = {p_acc:.2f}, F1 p = {p_f1:.2f}"]

    # Sec. III-D — deployed API configuration: average of the 9 modality models
    n = summary(fuse(P["cnn_lstm"], CHEST_BUNDLES + WRIST_BUNDLES))
    got.update({"nine_model.acc": n["acc"], "nine_model.f1": n["f1"]})
    md += ["\n## Sec. III-D — deployed API (average of 9 modality CNN-LSTMs)\n",
           f"Mean acc. {n['acc']:.2f}% (std {n['std']:.2f}%), mean F1 {n['f1']:.4f}, "
           f"worst fold {n['worst']:.2f}% ({n['worst_subject']})"]

    # Per-subject table (Fig. 5)
    md += ["\n## Per-subject F1, fused CNN-LSTM (Fig. 5)\n", "| Subject | Acc. | F1 |", "|---|---|---|"]
    md += [f"| {sid} | {r.acc:.4f} | {r.f1:.4f} |" for sid, r in s["per_fold"].iterrows()]

    # Check against the paper
    md += ["\n## Check against paper\n", "| Quantity | Paper | Reproduced | OK |", "|---|---|---|---|"]
    failures = []
    for k, (want, tol) in PAPER.items():
        ok = abs(got[k] - want) <= tol + 1e-9
        failures += [] if ok else [k]
        md.append(f"| {k} | {want} | {got[k]:.4f} | {'✓' if ok else '✗'} |")

    (OUT / "paper_results.md").write_text("\n".join(md) + "\n")
    (OUT / "paper_results.json").write_text(json.dumps({k: float(v) for k, v in got.items()}, indent=2))

    if args.figures:
        make_figures(T1, fused, P)

    print("\n".join(md[:md.index("\n## Check against paper\n")]))
    print(f"\n{len(PAPER) - len(failures)}/{len(PAPER)} values match the paper.")
    if failures:
        print("MISMATCH:", ", ".join(failures))
    print(f"Written to {OUT.relative_to(ROOT)}/")
    return 1 if failures else 0


def make_figures(T1, fused, P) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig. 2 — per-fold accuracy by sensor configuration
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Chest-only", "Wrist-only", "Multimodal\n(fused)"]
    data = [T1[k]["per_fold"].acc * 100 for k in ("chest", "wrist", "fused")]
    ax.boxplot(data, widths=0.5, showfliers=False)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-.08, .08, len(d)), d, s=14, alpha=.7)
        ax.scatter(i, d.mean(), marker="D", color="black", zorder=3)
    ax.set_xticks([1, 2, 3], labels); ax.set_ylabel("LOSO accuracy (%)")
    ax.set_title("Per-fold accuracy (CNN-LSTM)"); fig.tight_layout(); fig.savefig(OUT / "fig2_sensor_ablation.png", dpi=200)

    # Fig. 3 — architecture comparison (Table II, CNN and CNN-LSTM rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics = ["accuracy", "precision", "recall", "f1"]
    for j, m in enumerate(["cnn", "cnn_lstm"]):
        r = load_fold_metrics(m); r = r[r.experiment == "multimodal"][metrics].mean()
        ax.bar(np.arange(4) + (j - .5) * .35, r.values, .35, label={"cnn": "CNN", "cnn_lstm": "CNN-LSTM"}[m])
    ax.set_xticks(range(4), ["Accuracy", "Precision", "Recall", "F1"]); ax.set_ylim(.8, 1)
    ax.legend(); ax.set_title("All-sensor single model, LOSO mean"); fig.tight_layout()
    fig.savefig(OUT / "fig3_architectures.png", dpi=200)

    # Fig. 4 — pooled confusion matrix
    cm = confusion_matrix(fused.y_true, fused.y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["Non-stress", "Stress"]); ax.set_yticks([0, 1], ["Non-stress", "Stress"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Fused CNN-LSTM, pooled LOSO")
    fig.tight_layout(); fig.savefig(OUT / "fig4_confusion_matrix.png", dpi=200)

    # Fig. 5 — subject-wise F1
    pf = T1["fused"]["per_fold"]
    colors = ["tab:green" if v == 1 else "tab:red" if v == pf.f1.min() else "tab:blue" for v in pf.f1]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(pf.index, pf.f1, color=colors); ax.axhline(pf.f1.mean(), ls="--", color="black", lw=1)
    ax.set_ylabel("F1-score"); ax.set_ylim(0, 1.05); ax.set_title("Subject-wise F1, fused CNN-LSTM")
    fig.tight_layout(); fig.savefig(OUT / "fig5_subject_f1.png", dpi=200)
    plt.close("all")


if __name__ == "__main__":
    sys.exit(main())
