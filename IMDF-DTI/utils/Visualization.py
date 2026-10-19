import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def plot_feature_importance(weights_dict, save_path):
    """
    IMDF-DTI 专用可视化工具
    功能：将 1D/2D 掩码权重输出为高分辨率热力图，并自动分类归档
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)


    plt.rcParams['font.family'] = 'serif'

    for name, weight in weights_dict.items():

        plt.figure(figsize=(15, 3))


        im = plt.imshow(weight.reshape(1, -1), aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)


        cbar = plt.colorbar(im, pad=0.02)
        cbar.set_label('Mask score', fontsize=10)

        plt.title(f"IMDF-DTI Interpretability: {name.replace('_', ' ').title()}", fontsize=12, pad=15)
        plt.xlabel("Feature index", fontsize=10)
        plt.yticks([])


        save_filename = os.path.join(save_path, f"{name}.png")
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"📊 解释性报告已成功导出至: {os.path.abspath(save_path)}")

def _set_paper_style():
    """
    设置论文投稿风格。
    """
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


def _safe_1d_array(x):
    """
    将输入统一转换为一维 numpy 数组。
    """
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


def plot_single_mask_heatmap(mask, title, save_base, display_mode="raw", cmap="YlOrRd"):
    """
    生成单个 X-Mask 热力图。
    """
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
    if display_mode == "raw":
        cbar.set_label("Mask score")
    else:
        cbar.set_label("Normalized mask score")

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.savefig(f"{save_base}.png", dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(f"{save_base}.pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def plot_single_mask_combined(mask, title, save_base, display_mode="raw", cmap="YlOrRd"):
    """
    生成组合图：上方热图，下方曲线。
    这个最适合论文 Figure 3。
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
    if display_mode == "raw":
        cbar.set_label("Mask score")
    else:
        cbar.set_label("Normalized mask score")

    idx = np.arange(length)


    ax_line.plot(idx, x, linewidth=1.3, color="#2C7A7B")
    ax_line.fill_between(idx, x, alpha=0.20, color="#BFE3DC")

    ax_line.set_xlabel("Feature index")
    if display_mode == "raw":
        ax_line.set_ylabel("Mask score")
    else:
        ax_line.set_ylabel("Normalized mask score")

    ax_line.set_ylim(0, 1.02)
    ax_line.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(f"{save_base}.png", dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(f"{save_base}.pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def plot_topk_mask(mask, title, save_base, top_k=20, display_mode="raw"):
    """
    生成 Top-k 高敏感位置图（论文推荐版：lollipop / 点线图）。

    相比普通柱状图，lollipop 图减少了大面积色块，保留排序信息，
    更适合 Figure 7 / Figure 8 这类四个子图组合展示。
    """
    x = _normalize_for_display(mask, mode=display_mode)

    top_k = min(top_k, len(x))
    top_idx = np.argsort(x)[-top_k:][::-1]
    top_score = x[top_idx]
    ranks = np.arange(top_k)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))


    stem_color = "#9ECAE1"
    point_color = "#2B8CBE"
    edge_color = "#1B4F72"

    ax.vlines(
        ranks,
        ymin=0,
        ymax=top_score,
        color=stem_color,
        linewidth=2.0,
        alpha=0.95
    )

    ax.scatter(
        ranks,
        top_score,
        s=42,
        color=point_color,
        edgecolor=edge_color,
        linewidth=0.55,
        zorder=3
    )


    for i in range(min(3, top_k)):
        ax.text(
            ranks[i],
            top_score[i] + 0.025,
            str(top_idx[i]),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=edge_color
        )

    ax.set_title(title, pad=9)
    ax.set_xlabel("Top-ranked feature index")
    if display_mode == "raw":
        ax.set_ylabel("Mask score")
    else:
        ax.set_ylabel("Normalized mask score")

    ax.set_xticks(ranks)
    ax.set_xticklabels([str(i) for i in top_idx], rotation=45, ha="right")
    ax.set_ylim(0, min(1.08, max(1.0, float(top_score.max()) + 0.08)))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    fig.subplots_adjust(bottom=0.22)
    fig.savefig(f"{save_base}.png", dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(f"{save_base}.pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def plot_dfgu_weights(weights_dict, save_dir):
    """
    生成 DFGU 动态融合权重图（论文推荐版：横向点图）。

    横向点图比普通柱状图更简洁，数值标签更清楚，
    适合单个数据集的 DFGU 权重展示。
    """
    save_dir = Path(save_dir)

    has_1d = "dfgu_1d_mean" in weights_dict
    has_2d = "dfgu_2d_mean" in weights_dict

    if not has_1d and not has_2d:
        print("[Warning] ensemble_weights.npy 中没有找到 dfgu_1d_mean 或 dfgu_2d_mean")
        return

    labels = []
    values = []

    if has_1d:
        w1d = _safe_1d_array(weights_dict["dfgu_1d_mean"])
        labels.extend(["1D Drug", "1D Target"])
        values.extend([w1d[0], w1d[1]])

    if has_2d:
        w2d = _safe_1d_array(weights_dict["dfgu_2d_mean"])
        labels.extend(["2D Drug", "2D Target"])
        values.extend([w2d[0], w2d[1]])

    values = np.asarray(values, dtype=float)
    y = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(5.8, 3.2))

    colors = ["#2B8CBE", "#74C476", "#08519C", "#238B45"][:len(values)]

    ax.hlines(
        y,
        xmin=0,
        xmax=values,
        color="#D9D9D9",
        linewidth=4.0,
        zorder=1
    )
    ax.scatter(
        values,
        y,
        s=90,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=3
    )

    for yi, v in zip(y, values):
        ax.text(
            v + 0.025,
            yi,
            f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=9
        )

    ax.set_title("DFGU Dynamic Fusion Weights", pad=9)
    ax.set_xlabel("Mean weight")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(1.0, float(values.max()) + 0.15))
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    out_base = save_dir / "DFGU_dynamic_fusion_weights"
    fig.savefig(f"{out_base}.png", dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def plot_dfgu_weights_two_datasets(weight_paths, save_dir, out_name="DFGU_dynamic_fusion_weights_compare"):
    """
    生成 Figure 9 推荐版：DrugBank 与 BioSNAP 的 DFGU 权重并排比较图。

    参数：
    weight_paths:
        dict，例如：
        {
            "DrugBank": "./DrugBank-430/ensemble_weights.npy",
            "BioSNAP": "./BIOSNAP-428/ensemble_weights.npy"
        }

    save_dir:
        图片保存目录。
    """
    _set_paper_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    panel_labels = ["1D Drug", "1D Target", "2D Drug", "2D Target"]
    rows = []

    for dataset_name, weight_path in weight_paths.items():
        weights = load_ensemble_weight_file(weight_path)

        if "dfgu_1d_mean" not in weights or "dfgu_2d_mean" not in weights:
            print(f"[Warning] {dataset_name} 缺少 dfgu_1d_mean 或 dfgu_2d_mean，跳过。")
            continue

        w1d = _safe_1d_array(weights["dfgu_1d_mean"])
        w2d = _safe_1d_array(weights["dfgu_2d_mean"])
        values = np.asarray([w1d[0], w1d[1], w2d[0], w2d[1]], dtype=float)
        rows.append((dataset_name, values))

    if len(rows) == 0:
        print("[Error] 没有可用于绘制 DFGU 对比图的数据。")
        return

    fig, axes = plt.subplots(
        1,
        len(rows),
        figsize=(5.2 * len(rows), 3.6),
        sharex=True
    )

    if len(rows) == 1:
        axes = [axes]

    colors = ["#2B8CBE", "#74C476", "#08519C", "#238B45"]

    for ax, (dataset_name, values) in zip(axes, rows):
        y = np.arange(len(panel_labels))

        ax.hlines(
            y,
            xmin=0,
            xmax=values,
            color="#D9D9D9",
            linewidth=4.0,
            zorder=1
        )
        ax.scatter(
            values,
            y,
            s=95,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=3
        )

        for yi, v in zip(y, values):
            ax.text(
                v + 0.025,
                yi,
                f"{v:.3f}",
                va="center",
                ha="left",
                fontsize=9
            )

        ax.set_title(dataset_name, pad=8)
        ax.set_yticks(y)
        ax.set_yticklabels(panel_labels)
        ax.set_xlim(0, max(1.0, float(values.max()) + 0.15))
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.45)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    fig.supxlabel("Mean DFGU weight", y=0.02)
    fig.suptitle("DFGU Dynamic Fusion Weights", y=1.02, fontsize=12)
    fig.tight_layout()

    out_base = save_dir / out_name
    fig.savefig(f"{out_base}.png", dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)

    print(f"[Done] DFGU 对比图已保存至: {out_base}.png / .pdf")


def generate_interpretability_figures_from_weights(
    weight_path,
    save_dir="./IMDF_interpretability_from_weights",
    display_mode="raw",
    cmap="YlOrRd",
    top_k=20
):
    """
    直接从 ensemble_weights.npy 生成 IMDF-DTI 可解释性图。

    参数：
    weight_path:
        ensemble_weights.npy 路径。

    save_dir:
        图片输出目录。

    display_mode:
        raw：保留原始 mask 数值。
        minmax：每个分支单独归一化，视觉差异更明显。

    cmap:
        推荐 YlOrRd、viridis、magma。

    top_k:
        Top-k 敏感位置数量。
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

    mask_name_map = {
        "mask_1d_drug": "1D Drug X-Mask Scores",
        "mask_1d_prot": "1D Target X-Mask Scores",
        "mask_2d_drug": "2D Drug X-Mask Scores",
        "mask_2d_prot": "2D Target X-Mask Scores",
    }

    found_masks = 0

    for key, title in mask_name_map.items():
        if key not in weights:
            print(f"[Warning] {key} not found in ensemble_weights.npy")
            continue

        found_masks += 1
        mask = weights[key]
        base_name = f"{key}_{display_mode}"

        plot_single_mask_heatmap(
            mask=mask,
            title=title,
            save_base=save_dir / f"{base_name}_heatmap",
            display_mode=display_mode,
            cmap=cmap
        )

        plot_single_mask_combined(
            mask=mask,
            title=title,
            save_base=save_dir / f"{base_name}_combined",
            display_mode=display_mode,
            cmap=cmap
        )

        plot_topk_mask(
            mask=mask,
            title=f"Top-{top_k} Mask-weighted Features: {title}",
            save_base=save_dir / f"{base_name}_top{top_k}",
            top_k=top_k,
            display_mode=display_mode
        )

    plot_dfgu_weights(weights, save_dir)

    if found_masks == 0:
        print("[Error] 没有找到 X-Mask 权重，请检查 ensemble_weights.npy 中的 key 名称。")
    else:
        print(f"[Done] 可解释性图已保存至: {save_dir.resolve()}")
        print(f"[Info] display_mode = {display_mode}")
