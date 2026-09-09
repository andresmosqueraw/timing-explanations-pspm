"""The figures the reference code bases produce, for one log.

SHAP: beeswarm, bar, waterfall, force, decision, dependence (scatter).
LIME: per-instance bar and HTML. Permutation importance: box plot over the
repetitions and bar chart of the means. ALE: one line per feature.
Correlation heat map and mutual-information bar chart of the state
features. Cross-method agreement heat map of the global importances.
All matplotlib (``Agg``), written as PDF and PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, out: Path, stem: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        paths.append(str(p))
    plt.close(fig)
    plt.close("all")  # shap's force/decision plots open figures of their own
    return paths


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def shap_global_plots(explanation, out: Path, prefix: str) -> list[str]:
    import shap

    files = []
    plt.figure(figsize=(6, 3.2))
    shap.plots.beeswarm(explanation, show=False)
    files += _save(plt.gcf(), out, f"{prefix}_beeswarm")

    plt.figure(figsize=(6, 3.0))
    shap.plots.bar(explanation, show=False)
    files += _save(plt.gcf(), out, f"{prefix}_bar")

    for j, name in enumerate(explanation.feature_names):
        plt.figure(figsize=(5, 3.2))
        shap.plots.scatter(explanation[:, j], color=explanation, show=False)
        files += _save(plt.gcf(), out, f"{prefix}_dependence_{name}")
    return files


def shap_local_plots(explanation, i: int, out: Path, prefix: str) -> list[str]:
    import shap

    files = []
    plt.figure(figsize=(6, 3.2))
    shap.plots.waterfall(explanation[i], show=False)
    files += _save(plt.gcf(), out, f"{prefix}_waterfall_{i}")

    shap.force_plot(
        float(explanation.base_values[i]), np.asarray(explanation.values[i]), np.round(np.asarray(explanation.data[i]), 3),
        feature_names=list(explanation.feature_names), matplotlib=True, show=False, figsize=(12, 2.4), text_rotation=15,
    )
    files += _save(plt.gcf(), out, f"{prefix}_force_{i}")

    plt.figure(figsize=(5, 3.2))
    shap.decision_plot(
        float(explanation.base_values[i]), np.asarray(explanation.values[i]), np.asarray(explanation.data[i]),
        feature_names=list(explanation.feature_names), show=False,
    )
    files += _save(plt.gcf(), out, f"{prefix}_decision_{i}")
    return files


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------


def lime_local_plot(lime_exp, i: int, out: Path, prefix: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    fig = lime_exp.as_pyplot_figure()
    fig.set_size_inches(5.5, 2.6)
    files = _save(fig, out, f"{prefix}_lime_{i}")
    html = out / f"{prefix}_lime_{i}.html"
    lime_exp.save_to_file(str(html))
    files.append(str(html))
    return files


# ---------------------------------------------------------------------------
# Permutation importance, ALE, correlations, mutual information
# ---------------------------------------------------------------------------


def permutation_plots(perm: dict, feature_names: list[str], out: Path, prefix: str) -> list[str]:
    reps = np.asarray(perm["importances"])  # (D, R)
    order = np.argsort(-reps.mean(axis=1))
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.boxplot([reps[j] for j in order], vert=False, tick_labels=[feature_names[j] for j in order])
    ax.set_xlabel("RMS displacement of the target (per repetition)")
    files = _save(fig, out, f"{prefix}_permutation_box")

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.barh([feature_names[j] for j in order][::-1], reps.mean(axis=1)[order][::-1],
            xerr=reps.std(axis=1)[order][::-1], color="#4c72b0")
    ax.set_xlabel("Permutation importance (mean over repetitions)")
    files += _save(fig, out, f"{prefix}_permutation_bar")
    return files


def ale_plots(ale: dict, feature_names: list[str], out: Path, prefix: str) -> list[str]:
    fig, axes = plt.subplots(1, len(feature_names), figsize=(3.0 * len(feature_names), 2.8))
    for ax, name in zip(np.atleast_1d(axes), feature_names):
        a = ale[name]
        ax.plot(a["grid"], a["ale"], marker="o", ms=3, color="#c44e52")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(name)
    np.atleast_1d(axes)[0].set_ylabel("ALE of the target")
    fig.tight_layout()
    return _save(fig, out, f"{prefix}_ale")


def correlation_heatmap(states: np.ndarray, feature_names: list[str], out: Path, prefix: str) -> list[str]:
    with np.errstate(invalid="ignore", divide="ignore"):  # constant columns
        corr = np.nan_to_num(np.corrcoef(states.astype(np.float64), rowvar=False))
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")
    ax.set_yticks(range(len(feature_names)), feature_names)
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            ax.text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, out, f"{prefix}_correlations")


def mutual_info_plot(mi: np.ndarray, feature_names: list[str], out: Path, prefix: str) -> list[str]:
    order = np.argsort(-np.asarray(mi))
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.bar([feature_names[j] for j in order], np.asarray(mi)[order], color="#2e8b57")
    ax.set_ylabel("Mutual information with the binary target")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, out, f"{prefix}_mutual_information")


# ---------------------------------------------------------------------------
# Comparisons across methods
# ---------------------------------------------------------------------------


def importance_comparison(importances: dict[str, np.ndarray], feature_names: list[str], out: Path, prefix: str) -> list[str]:
    """Global importance of every method, each scaled to share of its own
    total, side by side per feature."""
    names = list(importances)
    M = np.column_stack([np.abs(np.asarray(importances[n], dtype=np.float64)) for n in names])
    M = M / np.where(M.sum(axis=0) > 0, M.sum(axis=0), 1.0)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    width = 0.8 / len(names)
    x = np.arange(len(feature_names))
    for c, n in enumerate(names):
        ax.bar(x + c * width - 0.4 + width / 2, M[:, c], width, label=n)
    ax.set_xticks(x, feature_names, rotation=15, ha="right")
    ax.set_ylabel("Share of the method's total importance")
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, out, f"{prefix}_method_comparison")


def agreement_heatmap(agree: dict, out: Path, prefix: str, key: str = "spearman") -> list[str]:
    names = list(agree["ranking"])
    n = len(names)
    M = np.eye(n)
    for pair, v in agree["pairs"].items():
        a, b = pair.split("|")
        val = v[key]
        M[names.index(a), names.index(b)] = M[names.index(b), names.index(a)] = np.nan if val is None else val
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(M, cmap="viridis", vmin=-1 if key != "jaccard_top2" else 0, vmax=1)
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, "n/a" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] < 0.5 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{key} between methods' global importances", fontsize=9)
    return _save(fig, out, f"{prefix}_agreement_{key}")
