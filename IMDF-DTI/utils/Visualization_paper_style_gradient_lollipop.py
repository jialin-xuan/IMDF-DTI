import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path


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


PALETTE = {
    "1D Drug": "#4C78A8",
    "1D Target": "#72B7B2",
    "2D Drug": "#F58518",
    "2D Target": "#54A24B"
}


MASK_NAME_MAP = {
    "mask_1d_drug": ("1D Drug X-Mask Scores", "1D Drug"),
    "mask_1d_prot": ("1D Target X-Mask Scores", "1D Target"),
    "mask_2d_drug": ("2D Drug X-Mask Scores", "2D Drug"),
    "mask_2d_prot": ("2D Target X-Mask Scores", "2D Target"),
}


def _blend_with_white(color, blend_factor):
    """
    将颜色与白色混合。blend_factor 越大，颜色越浅。
    blend_factor=0 表示原色，blend_factor=1 表示白色。
    """
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    mixed = rgb * (1 - blend_factor) + white * blend_factor
    return mixed


def _make_rank_gradient(base_color, n, light_blend=0.55, dark_blend=0.0):
    """
    生成从深到浅的渐变色列表。排名越靠前，颜色越深。
    """
    if n <= 1:
        return [base_color]
    blend_values = np.linspace(dark_blend, light_blend, n)
    return [_blend_with_white(base_color, b) for b in blend_values]


def _safe_1d_array(x):
    """将输入统一转换为一维 numpy 数组。"""
    x = np.asarray(x, dtype=float).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return x


def _normalize_for_display(x, mode="raw"):
    """
    mode='raw'：保留原始 mask 数值，更严谨。
    mode='minmax'：单独归一化，更容易看出局部差异。
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


def load_ensemble_weight_file(weight_path):
    """
    读取 ensemble_weights.npy。
    该文件一般是 np.save(dict) 保存的 object array。
    """
    obj = np.load(weight_path, allow_pickle=True)

    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        obj = obj.item()

    if not isinstance(obj, dict):
        raise TypeError(
            "ensemble_weights.npy 应该是一个 dict，里面包含 mask_1d_drug、mask_1d_prot、mask_2d_drug、mask_2d_prot 等键。"
        )

    return obj


def plot_single_mask_heatmap(mask, title, save_base, display_mode="raw", cmap="YlGnBu"):
    """生成单个 X-Mask 热力图。"""
    x = _normalize_for_display(mask, mode=display_mode)
    length = len(x)

    fig_width = max(8, min(16, length / 14))
    fig, ax = plt.subplots(figsize=(fig_width, 2.2))

    im = ax.imshow(
        x.reshape(1, -1),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax.set_title(title, pad=10)
    ax.set_xlabel("Feature index")
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Mask score" if display_mode == "raw" else "Normalized mask score")

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.savefig(f"{save_base}.png", dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(f"{save_base}.pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def plot_single_mask_combined(mask, title, save_base, display_mode="raw", cmap="YlGnBu"):
    """
    生成组合图：上方热图，下方曲线。
    适合图 3–6。
    """
    x = _normalize_for_display(mask, mode=display_mode)
    length = len(x)
    fig_width = max(8, min(16, length / 14))

    fig, axes = plt.subplots(
        2, 1,
        figsize=(fig_width, 4.0),
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.18}
    )

    ax_hm, ax_line = axes

    im = ax_hm.imshow(
        x.reshape(1, -1),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax_hm.set_title(title, pad=8)
    ax_hm.set_yticks([])
    ax_hm.set_xticks([])

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.015)
    cbar.set_label("Mask score" if display_mode == "raw" else "Normalized mask score")

    idx = np.arange(length)
    ax_line.plot(idx, x, linewidth=1.3, color="#2C7A7B")
    ax_line.fill_between(idx, x, alpha=0.20, color="#BFE3DC")
    ax_line.set_xlabel("Feature index")
    ax_line.set_ylabel("Mask score" if display_mode == "raw" else "Normalized mask score")
    ax_line.set_ylim(0, 1.02)
    ax_line.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(f"{save_base}.png", dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(f"{save_base}.pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def plot_topk_mask_lollipop(mask, title, save_base, top_k=20, display_mode="minmax", branch_label=None):
    """
    生成 Top-k 高敏感位置横向 lollipop 图。
    每个分支使用各自基色，并在图内做“越重要颜色越深”的渐变。
    适合图 7 / 图 8。
    """
    x = _normalize_for_display(mask, mode=display_mode)
    top_k = min(top_k, len(x))

    top_idx = np.argsort(x)[-top_k:][::-1]
    top_score = x[top_idx]
    y_pos = np.arange(top_k)

    if branch_label is None:
        branch_label = "1D Drug"
    base_color = PALETTE.get(branch_label, "#4C78A8")
    gradient_colors = _make_rank_gradient(base_color, top_k, light_blend=0.60, dark_blend=0.0)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))


    for y, score, c in zip(y_pos, top_score, gradient_colors):
        ax.hlines(y=y, xmin=0, xmax=score, color=c, alpha=0.95, linewidth=2.0)
        ax.scatter(score, y, s=58, color=c, edgecolor="white", linewidth=0.9, zorder=3)
        ax.text(score + 0.015, y, f"{score:.2f}", va="center", ha="left", fontsize=8, color="#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(i) for i in top_idx])
    ax.invert_yaxis()

    ax.set_xlim(0, min(1.18, max(top_score) + 0.16))
    ax.set_xlabel("Mask score" if display_mode == "raw" else "Normalized mask score")
    ax.set_ylabel("Feature index")
    ax.set_title(title, pad=10)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(f"{save_base}.png", dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(f"{save_base}.pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def plot_dfgu_weights_single(weights_dict, save_base):
    """
    生成单数据集 DFGU 权重点图（可选）。
    """
    labels, values = _extract_dfgu_values(weights_dict)
    colors = [PALETTE.get(label, "#4C78A8") for label in labels]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    y = np.arange(len(labels))

    ax.hlines(y, xmin=0, xmax=values, color=colors, linewidth=1.6, alpha=0.50)
    ax.scatter(values, y, s=65, color=colors, edgecolor="black", linewidth=0.4, zorder=3)

    for yi, v in zip(y, values):
        ax.text(v + 0.015, yi, f"{v:.3f}", va="center", ha="left", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, values.max() + 0.15))
    ax.set_xlabel("Mean Weight")
    ax.set_title("DFGU Dynamic Fusion Weights", pad=10)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(f"{save_base}.png", dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(f"{save_base}.pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def _extract_dfgu_values(weights_dict):
    """提取 DFGU 1D/2D 的四个平均权重。"""
    labels = []
    values = []

    has_1d = "dfgu_1d_mean" in weights_dict
    has_2d = "dfgu_2d_mean" in weights_dict

    if not has_1d and not has_2d:
        raise KeyError("ensemble_weights.npy 中没有找到 dfgu_1d_mean 或 dfgu_2d_mean")

    if has_1d:
        w1d = _safe_1d_array(weights_dict["dfgu_1d_mean"])
        labels.extend(["1D Drug", "1D Target"])
        values.extend([w1d[0], w1d[1]])

    if has_2d:
        w2d = _safe_1d_array(weights_dict["dfgu_2d_mean"])
        labels.extend(["2D Drug", "2D Target"])
        values.extend([w2d[0], w2d[1]])

    return labels, np.asarray(values, dtype=float)


def plot_dfgu_weights_slope(drugbank_weights, biosnap_weights, save_base):
    """
    生成 DFGU 的 slope chart。
    适合图 9：比较 DrugBank 与 BioSNAP 的权重变化。
    """
    labels_db, values_db = _extract_dfgu_values(drugbank_weights)
    labels_bs, values_bs = _extract_dfgu_values(biosnap_weights)

    if labels_db != labels_bs:
        raise ValueError("DrugBank 与 BioSNAP 的 DFGU 分支标签不一致，无法绘制 slope chart。")

    labels = labels_db

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    x_left, x_right = 0, 1
    for i, label in enumerate(labels):
        color = PALETTE.get(label, "#4C78A8")
        y_left = values_db[i]
        y_right = values_bs[i]

        ax.plot([x_left, x_right], [y_left, y_right], color=color, linewidth=2.0, alpha=0.9)
        ax.scatter([x_left, x_right], [y_left, y_right], s=70, color=color,
                   edgecolor="white", linewidth=0.9, zorder=3)

        ax.text(x_left - 0.05, y_left, f"{label}\n{y_left:.3f}", ha="right", va="center", fontsize=9)
        ax.text(x_right + 0.05, y_right, f"{label}\n{y_right:.3f}", ha="left", va="center", fontsize=9)

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(["DrugBank", "BioSNAP"], fontsize=10)
    ax.set_ylabel("Mean Weight")
    ax.set_title("DFGU Dynamic Fusion Weights Across Datasets", pad=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(f"{save_base}.png", dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(f"{save_base}.pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def generate_interpretability_figures_from_weights(
    weight_path,
    save_dir="./IMDF_interpretability_from_weights",
    display_mode="raw",
    cmap="YlGnBu",
    top_k=20
):
    """
    直接从 ensemble_weights.npy 生成 IMDF-DTI 可解释性图。

    图 3–6：combined 图
    图 7/8：Top-k 横向 lollipop 图
    """
    _set_paper_style()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    weights = load_ensemble_weight_file(weight_path)

    print("[Info] ensemble_weights.npy keys:")
    for k in weights.keys():
        try:
            print(f"  {k}: shape={np.asarray(weights[k]).shape}")
        except Exception:
            print(f"  {k}")

    found_masks = 0

    for key, (title, branch_label) in MASK_NAME_MAP.items():
        if key not in weights:
            print(f"[Warning] {key} not found in ensemble_weights.npy")
            continue

        found_masks += 1
        mask = weights[key]
        base_name = f"{key}_{display_mode}"


        plot_single_mask_combined(
            mask=mask,
            title=title,
            save_base=save_dir / f"{base_name}_combined",
            display_mode=display_mode,
            cmap=cmap
        )


        plot_single_mask_heatmap(
            mask=mask,
            title=title,
            save_base=save_dir / f"{base_name}_heatmap",
            display_mode=display_mode,
            cmap=cmap
        )


        plot_topk_mask_lollipop(
            mask=mask,
            title=f"Top-{top_k} Mask-weighted Features: {title}",
            save_base=save_dir / f"{base_name}_top{top_k}_lollipop",
            top_k=top_k,
            display_mode=display_mode,
            branch_label=branch_label
        )


    try:
        plot_dfgu_weights_single(weights, save_dir / "DFGU_dynamic_fusion_weights_single")
    except Exception as e:
        print(f"[Warning] 单数据集 DFGU 图未生成：{e}")

    if found_masks == 0:
        print("[Error] 没有找到 X-Mask 权重，请检查 ensemble_weights.npy 中的 key 名称。")
    else:
        print(f"[Done] 可解释性图已保存至: {save_dir.resolve()}")
        print(f"[Info] display_mode = {display_mode}")


def generate_dfgu_compare_figure(drugbank_weight_path, biosnap_weight_path, save_dir):
    """
    图 9：生成 DrugBank vs BioSNAP 的 DFGU slope chart。
    """
    _set_paper_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    drugbank_weights = load_ensemble_weight_file(drugbank_weight_path)
    biosnap_weights = load_ensemble_weight_file(biosnap_weight_path)

    plot_dfgu_weights_slope(
        drugbank_weights=drugbank_weights,
        biosnap_weights=biosnap_weights,
        save_base=save_dir / "DFGU_dynamic_fusion_weights_compare"
    )

    print(f"[Done] DFGU 对比图已保存至: {save_dir.resolve()}")

