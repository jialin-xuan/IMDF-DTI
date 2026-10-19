import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


def _set_paper_style():
    """设置论文投稿风格。"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none"
    })


PAIR_PALETTE = {
    "drug_1d": "#4C78A8",
    "drug_2d": "#F58518",
    "protein_1d": "#72B7B2",
    "protein_2d": "#54A24B",
}


def _safe_1d_array(x):
    """将输入统一转换为一维 numpy 数组。"""
    x = np.asarray(x, dtype=float).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return x


def _normalize_for_display(x, mode="minmax"):
    """
    mode='raw'：保留原始 mask 数值，更严谨。
    mode='minmax'：分支内部归一化，更适合展示 Top-k 局部差异。
    """
    x = _safe_1d_array(x)

    if mode == "raw":
        return np.clip(x, 0, 1)

    if mode == "minmax":
        x_min, x_max = np.min(x), np.max(x)
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)

    raise ValueError("mode must be 'raw' or 'minmax'")


def _blend_with_white(hex_color, strength):
    """
    生成颜色深浅梯度。
    strength 越大，颜色越深。
    """
    base = np.array(to_rgb(hex_color))
    white = np.array([1.0, 1.0, 1.0])
    color = white * (1 - strength) + base * strength
    return color


def _rank_gradient_colors(base_color, n, min_strength=0.35, max_strength=0.95):
    """
    Top-1 最深，Top-n 最浅。
    """
    if n <= 1:
        return [_blend_with_white(base_color, max_strength)]

    strengths = np.linspace(max_strength, min_strength, n)
    return [_blend_with_white(base_color, s) for s in strengths]


def load_ensemble_weight_file(weight_path):
    """
    读取 ensemble_weights.npy。
    你的 ensemble_weights.npy 通常是 np.save(dict) 得到的 object array。
    """
    obj = np.load(weight_path, allow_pickle=True)

    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        obj = obj.item()

    if not isinstance(obj, dict):
        raise TypeError(
            "ensemble_weights.npy 应该是一个 dict，且包含 "
            "mask_1d_drug、mask_1d_prot、mask_2d_drug、mask_2d_prot 等键。"
        )

    return obj


def _get_topk(mask, top_k=15, display_mode="minmax"):
    """
    返回 Top-k 的 index 和 score。
    """
    x = _normalize_for_display(mask, mode=display_mode)
    top_k = min(top_k, len(x))

    top_idx = np.argsort(x)[-top_k:][::-1]
    top_score = x[top_idx]

    return top_idx, top_score


def plot_mirrored_topk_pair(
    left_mask,
    right_mask,
    left_name,
    right_name,
    title,
    save_base,
    top_k=15,
    display_mode="minmax",
    left_color="#4C78A8",
    right_color="#F58518",
    xlabel="Normalized mask score",
):
    """
    绘制左右镜像 Top-k 条形图（最终对称版）。

    左侧：1D 分支，条形向左；
    右侧：2D 分支，条形向右；
    中间：Top 排名；
    左右两侧：index + score，使用固定文字列，保证版式完全对称。

    适合论文最终版，重点优化：
    1. 左右数字完全对称；
    2. 避免 BioSNAP 左侧 Top-1 重合问题；
    3. 隐藏 y 轴短刻度线；
    4. 适当增大左右边距，使版面更整洁。
    """
    _set_paper_style()

    left_idx, left_score = _get_topk(left_mask, top_k=top_k, display_mode=display_mode)
    right_idx, right_score = _get_topk(right_mask, top_k=top_k, display_mode=display_mode)

    k = min(len(left_idx), len(right_idx))
    left_idx, left_score = left_idx[:k], left_score[:k]
    right_idx, right_score = right_idx[:k], right_score[:k]

    y = np.arange(k)

    left_colors = _rank_gradient_colors(left_color, k)
    right_colors = _rank_gradient_colors(right_color, k)

    fig_height = max(4.8, 0.34 * k + 1.3)
    fig, ax = plt.subplots(figsize=(9.0, fig_height))


    ax.barh(
        y,
        -left_score,
        color=left_colors,
        edgecolor="white",
        linewidth=0.6,
        height=0.72,
        label=left_name
    )
    ax.barh(
        y,
        right_score,
        color=right_colors,
        edgecolor="white",
        linewidth=0.6,
        height=0.72,
        label=right_name
    )


    ax.axvline(0, color="black", linewidth=0.9)


    ax.set_yticks(y)
    ax.set_yticklabels([f"Top-{i+1}" for i in range(k)])
    ax.invert_yaxis()


    ax.tick_params(axis="y", length=0, pad=18)


    lim = 1.45
    ax.set_xlim(-lim, lim)


    ticks = np.linspace(-1.0, 1.0, 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(t):.1f}" for t in ticks])


    text_color = "#222222"
    left_text_x = -1.20
    right_text_x = 1.20

    for yi, idx, score in zip(y, left_idx, left_score):
        ax.text(
            left_text_x,
            yi,
            f"{idx}  {score:.2f}",
            va="center",
            ha="right",
            fontsize=8.3,
            color=text_color,
            clip_on=False
        )

    for yi, idx, score in zip(y, right_idx, right_score):
        ax.text(
            right_text_x,
            yi,
            f"{idx}  {score:.2f}",
            va="center",
            ha="left",
            fontsize=8.3,
            color=text_color,
            clip_on=False
        )

    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=14)


    ax.text(
        -0.58,
        -0.95,
        left_name,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=left_color
    )
    ax.text(
        0.58,
        -0.95,
        right_name,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=right_color
    )


    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    note = (
        "Numbers indicate feature index and normalized score. "
        "Scores are normalized within each branch."
        if display_mode == "minmax"
        else
        "Numbers indicate feature index and raw mask score."
    )
    fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=8)


    fig.subplots_adjust(left=0.18, right=0.96, top=0.90, bottom=0.13)

    save_base = Path(save_base)
    save_base.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        f"{save_base}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white"
    )
    fig.savefig(
        f"{save_base}.pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white"
    )
    plt.close(fig)


def generate_mirrored_topk_figures_from_weights(
    weight_path,
    save_dir,
    dataset_name="Dataset",
    top_k=15,
    display_mode="minmax",
):
    """
    从单个数据集的 ensemble_weights.npy 生成两张镜像 Top-k 图：

    1) Protein branches: 1D Target vs 2D Target
    2) Drug branches: 1D Drug vs 2D Drug

    输出文件：
    - {dataset_name}_protein_1D_vs_2D_topK_mirrored.png/pdf
    - {dataset_name}_drug_1D_vs_2D_topK_mirrored.png/pdf
    """
    _set_paper_style()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    weights = load_ensemble_weight_file(weight_path)

    required_keys = [
        "mask_1d_drug",
        "mask_2d_drug",
        "mask_1d_prot",
        "mask_2d_prot",
    ]
    for key in required_keys:
        if key not in weights:
            raise KeyError(f"{weight_path} 中缺少必要 key: {key}")


    plot_mirrored_topk_pair(
        left_mask=weights["mask_1d_prot"],
        right_mask=weights["mask_2d_prot"],
        left_name="1D Target",
        right_name="2D Target",
        title=f"{dataset_name}: Top-{top_k} Mask-weighted Features in Target Branches",
        save_base=save_dir / f"{dataset_name}_target_1D_vs_2D_top{top_k}_mirrored",
        top_k=top_k,
        display_mode=display_mode,
        left_color=PAIR_PALETTE["protein_1d"],
        right_color=PAIR_PALETTE["protein_2d"],
    )


    plot_mirrored_topk_pair(
        left_mask=weights["mask_1d_drug"],
        right_mask=weights["mask_2d_drug"],
        left_name="1D Drug",
        right_name="2D Drug",
        title=f"{dataset_name}: Top-{top_k} Mask-weighted Features in Drug Branches",
        save_base=save_dir / f"{dataset_name}_drug_1D_vs_2D_top{top_k}_mirrored",
        top_k=top_k,
        display_mode=display_mode,
        left_color=PAIR_PALETTE["drug_1d"],
        right_color=PAIR_PALETTE["drug_2d"],
    )

    print(f"[Done] {dataset_name} mirrored Top-{top_k} figures saved to: {save_dir.resolve()}")
