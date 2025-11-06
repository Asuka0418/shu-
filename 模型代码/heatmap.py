#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GABMIL Attention Visualization — A/B/C三联图 + 批量模式
- A: 缩略图
- B: 注意力热力图 (离散Turbo色阶 + 色条)
- C: Top-K高注意力patch拼图
- 支持 --csv_path + --fold 批量生成
"""

import os, sys, glob, argparse
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# -------- Aslide (.kfb) --------
ASLIDE_DIR_DEFAULT = "/data/wl2p/Aslide"
if ASLIDE_DIR_DEFAULT not in sys.path:
    sys.path.append(ASLIDE_DIR_DEFAULT)
from aslide import Slide

# ========== Args ==========
def parse_args():
    ap = argparse.ArgumentParser("GABMIL Attention Visualization — A/B/C三联图 + 批量模式")
    ap.add_argument("--slide_id", help="单个slide_id")
    ap.add_argument("--csv_path", help="cv_test_predictions.csv路径（用于批量模式）")
    ap.add_argument("--fold", type=int, help="选择fold编号（用于csv批量模式）")

    ap.add_argument("--wsi_dir", required=True)
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--att", required=True)
    ap.add_argument("--out", default="./heatmaps")

    # 可调参数
    ap.add_argument("--patch_size", type=int, default=512)
    ap.add_argument("--scale", type=int, default=32)
    ap.add_argument("--pclip", type=float, default=97.0)
    ap.add_argument("--gamma", type=float, default=0.4)
    ap.add_argument("--mask_thresh", type=int, default=220)
    ap.add_argument("--prefix_len", type=int, default=9)
    ap.add_argument("--zoom_topk", type=int, default=5)
    ap.add_argument("--zoom_size", type=int, default=512)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--min_att", type=float, default=0.2)
    return ap.parse_args()

# --- safe colormap getter ---
def _get_cmap(name: str):
    try:
        return matplotlib.colormaps.get_cmap(name)
    except AttributeError:
        from matplotlib import cm
        return cm.get_cmap(name)

# ========== Utils ==========
def find_kfb_by_prefix(wsi_dir: str, slide_id: str, prefix_len: int = 9):
    exact = glob.glob(os.path.join(wsi_dir, "**", f"{slide_id}.kfb"), recursive=True)
    if exact:
        return exact[0]
    pref = slide_id[:prefix_len]
    cand = glob.glob(os.path.join(wsi_dir, "**", f"{pref}*.kfb"), recursive=True)
    if cand:
        cand = sorted(cand, key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
        return cand[0]
    return None

def read_wsi_thumb(kfb_path, scale):
    with Slide(kfb_path) as wsi:
        W0, H0 = wsi.dimensions
        lowest = wsi.level_count - 1
        lw, lh = wsi.level_dimensions[lowest]
        img_low = wsi.read_region((0,0), lowest, (lw, lh)).convert("RGB")
        Tw, Th = max(1, W0//scale), max(1, H0//scale)
        img_thumb = img_low.resize((Tw, Th), Image.LANCZOS) if (lw,lh)!=(Tw,Th) else img_low
    return W0, H0, np.array(img_thumb)

def build_tissue_mask(thumb_rgb, thresh=220):
    gray = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray < thresh).astype(np.uint8)
    mask = cv2.medianBlur(mask, 7)
    return mask

def _try_read_xy(pt_path):
    try:
        if pt_path.endswith((".pt",".pth")):
            d = torch.load(pt_path, map_location="cpu")
            if isinstance(d, dict):
                if "xy" in d: v = d["xy"]
                elif "xy_topleft" in d: v = d["xy_topleft"]
                else: return None
                v = v.numpy() if torch.is_tensor(v) else np.asarray(v)
                return v.astype(np.float32)
        elif pt_path.endswith(".npz"):
            d = np.load(pt_path)
            if "xy" in d.files: return d["xy"].astype(np.float32)
            if "xy_topleft" in d.files: return d["xy_topleft"].astype(np.float32)
    except Exception as e:
        print(f"[warn] fail load xy {pt_path}: {e}")
    return None

def load_xy(slide_id, feature_dir, prefix_len=9):
    prefix = slide_id[:prefix_len]
    cand = []
    for ext in (".pt",".pth",".npz"):
        cand += glob.glob(os.path.join(feature_dir, f"{prefix}*{ext}"))
    if not cand:
        raise FileNotFoundError(f"xy not found for prefix {prefix} in {feature_dir}")
    cand = sorted(cand, key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
    for p in cand:
        xy = _try_read_xy(p)
        if xy is not None:
            print(f"[xy] {os.path.basename(p)}  N={len(xy)}")
            return xy
    raise RuntimeError("matched files found but none contained xy.")

def load_att(slide_id, att_npz_path, prefix_len=9):
    d = np.load(att_npz_path)
    keys = list(d.files)
    if slide_id in d:
        return d[slide_id]
    pref = slide_id[:prefix_len]
    matches = [k for k in keys if k.startswith(pref)]
    if matches:
        key = sorted(matches, key=len)[0]
        print(f"[att] use {key} (prefix match {pref})")
        return d[key]
    loose = [k for k in keys if slide_id in k or pref in k]
    if loose:
        key = sorted(loose, key=len)[0]
        print(f"[att] use {key} (loose match {pref})")
        return d[key]
    raise KeyError(f"{slide_id} not found in {att_npz_path} (no loose match).")

# ========== Core ==========
def visualize_slide(slide_id, args):
    try:
        kfb = find_kfb_by_prefix(args.wsi_dir, slide_id, args.prefix_len)
        assert kfb and os.path.exists(kfb), f"未找到 {slide_id}.kfb"
        print(f"[INFO] === Processing {slide_id} ===")

        W0, H0, thumb = read_wsi_thumb(kfb, args.scale)
        Th, Tw = thumb.shape[:2]
        mask = build_tissue_mask(thumb, args.mask_thresh)

        xy = load_xy(slide_id, args.feature_dir, args.prefix_len)
        att = load_att(slide_id, args.att, args.prefix_len).astype(np.float32)
        if len(att) != len(xy):
            n = min(len(att), len(xy))
            att, xy = att[:n], xy[:n]

        xy_center = xy + args.patch_size/2.0
        scale_x, scale_y = Tw/float(W0), Th/float(H0)
        xy_thumb = np.stack([xy_center[:,0]*scale_x, xy_center[:,1]*scale_y], 1).astype(np.int32)
        h_idx = np.clip(xy_thumb[:,1], 0, Th-1)
        w_idx = np.clip(xy_thumb[:,0], 0, Tw-1)
        keep = (mask[h_idx, w_idx] > 0)
        xy_thumb, xy_center, att = xy_thumb[keep], xy_center[keep], att[keep]

        # === 注意力归一化 ===
        hi = np.percentile(att, args.pclip)
        att_norm = np.clip(att / (hi + 1e-8), 0, 1)
        att_norm = np.power(att_norm, args.gamma)

        strong = att_norm >= args.min_att
        xy_keep, att_keep = xy_thumb[strong], att_norm[strong]

        # === 构建离散色阶热力图 ===
        bins = args.bins
        levels = np.linspace(0, 1, bins + 1)
        idx_bin = np.digitize(att_keep, levels, right=True)
        idx_bin = np.clip(idx_bin, 1, bins)
        turbo = _get_cmap('turbo')
        palette = (turbo((np.arange(1, bins + 1) - 0.5) / bins)[:, :3] * 255).astype(np.uint8)
        overlayA = thumb.copy()
        half = max(2, int(round((args.patch_size/2) / args.scale)))
        for (x, y), b in zip(xy_keep, idx_bin):
            color = tuple(int(c) for c in palette[b - 1])
            x0, y0, x1, y1 = int(x - half), int(y - half), int(x + half), int(y + half)
            cv2.rectangle(overlayA, (x0, y0), (x1, y1), color, thickness=-1)
            if b >= bins - 1:
                cv2.rectangle(overlayA, (x0, y0), (x1, y1), (25, 25, 25), 1, cv2.LINE_AA)

        # === Top-K patch 拼图 ===
        idx_top = np.argsort(-att)[:args.zoom_topk]
        zoom_patches = []
        tile = 384
        with Slide(kfb) as wsi:
            for rank, i in enumerate(idx_top, start=1):
                x, y = xy_center[i]
                x0 = max(int(x - args.zoom_size/2), 0)
                y0 = max(int(y - args.zoom_size/2), 0)
                patch = wsi.read_region((x0, y0), 0, (args.zoom_size, args.zoom_size)).convert("RGB")
                patch = np.array(patch)
                patch = cv2.resize(patch, (tile, tile))
                cv2.putText(patch, f"#{rank}  att={att[i]:.4f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,64,64), 2)
                zoom_patches.append(patch)
        concat_for_plot = None
        if zoom_patches:
            cols = min(3, len(zoom_patches))
            rows = int(np.ceil(len(zoom_patches)/cols))
            while len(zoom_patches) < rows*cols:
                zoom_patches.append(np.ones((tile, tile, 3), dtype=np.uint8)*255)
            rows_list = [cv2.hconcat(zoom_patches[r*cols:(r+1)*cols]) for r in range(rows)]
            grid = cv2.vconcat(rows_list)
            concat_for_plot = grid[..., ::-1]

        # === 三联图输出 ===
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"{slide_id} — GABMIL Attention Visualization", fontsize=16)
        axes[0,0].imshow(thumb); axes[0,0].set_title("A. Thumbnail"); axes[0,0].axis("off")
        axes[0,1].imshow(overlayA); axes[0,1].set_title("B. Attention Heatmap (Turbo)"); axes[0,1].axis("off")
        cmap = matplotlib.colors.ListedColormap(palette/255.0)
        norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
                            ax=axes[0,1], fraction=0.046, pad=0.02, ticks=levels)
        cbar.ax.set_ylabel("Attention Intensity", rotation=90, fontsize=12)
        cbar.ax.set_yticklabels([f"{v:.1f}" for v in levels])
        if concat_for_plot is not None:
            axes[1,0].imshow(concat_for_plot)
            axes[1,0].set_title(f"C. Top-{args.zoom_topk} patches ({args.zoom_size}×{args.zoom_size})")
        else:
            axes[1,0].set_title("C. (no zoom)")
        axes[1,0].axis("off"); axes[1,1].axis("off")
        plt.tight_layout(rect=[0,0,1,0.96])
        outD = os.path.join(args.out, f"{slide_id}_paper_figure.png")
        plt.savefig(outD, dpi=300); plt.close(fig)
        print(f"[SAVE] {outD}")

    except Exception as e:
        print(f"[ERROR] Failed {slide_id}: {e}")

# ========== Main ==========
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.csv_path and args.fold is not None:
        df = pd.read_csv(args.csv_path)
        slides = df[df["fold"] == args.fold]["slide_id"].unique().tolist()
        print(f"[INFO] 批量模式：fold={args.fold}, 共 {len(slides)} 个样本")
        for sid in slides:
            visualize_slide(sid, args)
        print("[DONE] 所有样本已完成可视化")
    elif args.slide_id:
        visualize_slide(args.slide_id, args)
    else:
        raise ValueError("请指定 --slide_id 或 (--csv_path + --fold)")

if __name__ == "__main__":
    main()
