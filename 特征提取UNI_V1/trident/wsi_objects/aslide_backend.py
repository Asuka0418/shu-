# trident/wsi_objects/aslide_backend.py
# -*- coding: utf-8 -*-
"""
ASlide 后端适配（支持 .kfb 等），与 Processor.py 完整对齐的实现。

功能要点：
- 屏蔽 ASlide 的 tmap 依赖（避免 libopencv_core.so.3.4）
- 提供 AslideWSI，包含：
  - 基本字段：path/name/ext/level_dimensions/dimensions/level_downsamples/mpp_x/mpp_y/properties
  - 分割：segment_tissue(...) -> 保存 contours/*.jpg + contours_geojson/*.geojson，返回 geojson 路径
  - 取坐标：extract_tissue_coords(...) -> 在组织多边形内生成 patch 坐标，按 min_tissue_proportion 过滤
  - 可视化：visualize_coords(...) -> 在缩略图上画网格框
  - Patch 特征：extract_patch_features(...) -> 读图块 -> 预处理 -> patch_encoder 前向 -> 写 .h5
  - Slide 聚合：extract_slide_features(...) -> 读 patch 特征 -> slide_encoder 前向 -> 写 .h5
"""

import os
import sys
import types
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

# ---- 1) 确保能 import Aslide -------------------------------------------------
# 优先读环境变量；否则尝试常见路径（你的机器是 /data/wl2p/Aslide）
_ASLIDE_BASES: List[str] = []
if "ASLIDE_HOME" in os.environ:
    _ASLIDE_BASES += [os.environ["ASLIDE_HOME"], os.path.join(os.environ["ASLIDE_HOME"], "Aslide")]
_ASLIDE_BASES += ["/data/wl2p", "/data/wl2p/Aslide"]

for base in _ASLIDE_BASES:
    if os.path.isdir(base):
        parent = base if os.path.basename(base) != "Aslide" else os.path.dirname(base)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        break

# ---- 2) 屏蔽 Aslide.tmap（避免 OpenCV 3.4 动态库依赖） --------------------------
if "Aslide.tmap.tmap_slide" not in sys.modules:
    tmap_pkg = types.ModuleType("Aslide.tmap")
    tmap_slide = types.ModuleType("Aslide.tmap.tmap_slide")
    class _DummyTmapSlide:  # 只占位，KFB 不会用到
        pass
    tmap_slide.TmapSlide = _DummyTmapSlide
    sys.modules["Aslide.tmap"] = tmap_pkg
    sys.modules["Aslide.tmap.tmap_slide"] = tmap_slide

# ---- 3) 导入 ASlide -----------------------------------------------------------
try:
    from Aslide.aslide import Slide as _ASlide
except Exception as e:
    raise ImportError(
        f"Cannot import Aslide. Make sure /data/wl2p is in PYTHONPATH and ASlide is at /data/wl2p/Aslide. Original error: {e}"
    )

# ------------------------------------------------------------------------------
#                             AslideWSI
# ------------------------------------------------------------------------------
class AslideWSI:
    """
    ASlide -> TRIDENT 适配。
    """

    # ---------------- 基础构造 ----------------
    def __init__(self, path: str,
                 name: Optional[str] = None,
                 tissue_seg_path: Optional[str] = None,
                 **kwargs) -> None:
        # 路径与命名
        self.path: str = os.path.abspath(path)
        self.name: str = name or os.path.splitext(os.path.basename(path))[0]
        self.ext: str = os.path.splitext(path)[1].lower()
        # 记录已有的分割矢量路径（Processor.__init__ 可能会传进来）
        self.tissue_seg_path: Optional[str] = tissue_seg_path
        # 其它记录
        self.coords_dir: Optional[str] = None
        self.coords_path: Optional[str] = None
        self.properties: dict = {}

        # 打开 ASlide
        self._s = _ASlide(path)

        # 金字塔各层尺寸
        dims = None
        for attr in ("level_dimensions", "levels", "resolutions", "dims"):
            if hasattr(self._s, attr):
                dims = getattr(self._s, attr)
                break
        if isinstance(dims, dict) and "level_dimensions" in dims:
            dims = dims["level_dimensions"]
        if dims is None:
            raise RuntimeError("ASlide does not expose level_dimensions / levels / resolutions / dims.")
        self.level_dimensions: List[Tuple[int, int]] = [tuple(map(int, d)) for d in dims]

        # 主分辨率尺寸
        self.dimensions: Tuple[int, int] = self.level_dimensions[0]

        # 各层 downsample（第 0 层=1.0）
        W0, H0 = self.dimensions
        self.level_downsamples: List[float] = [1.0] + [
            (W0 / max(w, 1.0)) for (w, h) in self.level_dimensions[1:]
        ]

        # MPP（如 ASlide 有就挂上）
        mpp = getattr(self._s, "mpp", (None, None))
        if isinstance(mpp, (list, tuple)) and len(mpp) >= 2:
            self.mpp_x, self.mpp_y = mpp[0], mpp[1]
        else:
            self.mpp_x = self.mpp_y = None

    # ---------------- 必要属性/方法 ----------------
    @property
    def level_count(self) -> int:
        return len(self.level_dimensions)

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        """返回 numpy 数组（RGB）。"""
        pil_im = self._s.read_region(location=location, level=level, size=size)
        if isinstance(pil_im, Image.Image):
            pil_im = pil_im.convert("RGB")
        arr = np.asarray(pil_im)
        if arr.ndim == 3 and arr.shape[-1] == 4:  # 丢弃 alpha
            arr = arr[..., :3]
        return arr

    def get_best_level_for_downsample(self, downsample: float) -> int:
        diffs = [abs(d - float(downsample)) for d in self.level_downsamples]
        return int(np.argmin(diffs))

    # ---------------- 工具：目标倍率选层 ----------------
    def _choose_level_for_mag(self, target_mag: int) -> int:
        mag2mpp = {5: 2.0, 10: 1.0, 20: 0.5, 40: 0.25, 80: 0.125}
        tgt_mpp = mag2mpp.get(int(target_mag), 0.5)
        base_mpp = self.mpp_x if getattr(self, "mpp_x", None) else 0.25
        best_l, best_diff = 0, 1e9
        for l, ds in enumerate(self.level_downsamples):
            mpp_l = base_mpp * float(ds)
            diff = abs(mpp_l - tgt_mpp)
            if diff < best_diff:
                best_l, best_diff = l, diff
        return best_l

    # ---------------- 1) 真实分割：写 JPG + GeoJSON，返回 GeoJSON 路径 ----------------
    def segment_tissue(self,
                       segmentation_model=None,
                       target_mag: int = 10,
                       holes_are_tissue: bool = True,
                       job_dir: Optional[str] = None,
                       batch_size: int = 64,
                       device: str = "cpu",
                       **kwargs) -> str:
        """
        产物：
          - {job_dir}/contours/{name}.jpg
          - {job_dir}/contours_geojson/{name}.geojson
        返回：geojson 路径（绝对），并设置 self.tissue_seg_path
        """
        import geopandas as gpd
        from shapely.geometry import Polygon, box
        from shapely.ops import unary_union

        # I/O 目录
        if job_dir is None:
            base = os.environ.get("TRIDENT_JOB_DIR")
            if not base:
                trident_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                base = os.path.dirname(trident_dir)
            job_dir = base
        contours_dir = os.path.join(job_dir, "contours")
        contours_geojson_dir = os.path.join(job_dir, "contours_geojson")
        os.makedirs(contours_dir, exist_ok=True)
        os.makedirs(contours_geojson_dir, exist_ok=True)

        # 选择合适层并读取整幅缩略图
        lvl = self._choose_level_for_mag(target_mag)
        Wl, Hl = map(int, self.level_dimensions[lvl])
        img = self._s.read_region(location=(0, 0), level=lvl, size=(Wl, Hl)).convert("RGB")
        arr = np.asarray(img)

        # --- 简单且鲁棒的组织分割：亮度阈值 + 两次 3x3 邻域筛选（无 opencv/scipy 依赖） ---
        gray = (0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]).astype(np.float32)
        t = np.percentile(gray, 85)  # 85% 分位做阈值：背景更亮
        mask = (gray < t).astype(np.uint8)

        # 邻域统计 >=5 作为保留（近似开闭操作）
        def _filter9(x):
            pad = np.pad(x, 1, mode="constant")
            neigh = (pad[:-2,:-2]+pad[:-2,1:-1]+pad[:-2,2:]+
                     pad[1:-1,:-2]+pad[1:-1,1:-1]+pad[1:-1,2:]+
                     pad[2:,:-2]+pad[2:,1:-1]+pad[2:,2:])
            return (neigh >= 5).astype(np.uint8)
        mask = _filter9(mask)
        mask = _filter9(mask)

        # 保存可视化 JPG（供 Processor 锁文件/日志）
        vis = Image.fromarray((mask*255).astype(np.uint8)).convert("L")
        vis_path = os.path.join(contours_dir, f"{self.name}.jpg")
        vis.save(vis_path, "JPEG", quality=92)

        # --- 矢量化（优先 skimage，降级为矩形/合并） ---
        polys = []
        scaled = float(self.level_downsamples[lvl])
        try:
            from skimage.measure import find_contours
            contours = find_contours(mask, level=0.5)
            for cnt in contours:
                yy, xx = cnt[:,0], cnt[:,1]  # row(y), col(x)
                poly = Polygon([(x*scaled, y*scaled) for x, y in zip(xx, yy)])
                if poly.is_valid and poly.area > 0:
                    polys.append(poly)
        except Exception:
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                xmin, ymin = xs.min()*scaled, ys.min()*scaled
                xmax, ymax = (xs.max()+1)*scaled, (ys.max()+1)*scaled
                polys.append(box(xmin, ymin, xmax, ymax))

        if not polys:
            W0, H0 = map(int, self.level_dimensions[0])
            polys = [box(0, 0, W0, H0)]

        geom = unary_union(polys)
        if not holes_are_tissue:
            geom = geom.buffer(0)

        gdf = gpd.GeoDataFrame({"slide_id":[self.name], "level":[0]}, geometry=[geom], crs=None)
        geojson_path = os.path.join(contours_geojson_dir, f"{self.name}.geojson")
        gdf.to_file(geojson_path, driver="GeoJSON")

        self.tissue_seg_path = geojson_path
        return geojson_path

    # ---------------- 2) 真实坐标提取：组织内网格 & 面积占比过滤 ----------------
    def extract_tissue_coords(self,
                              target_mag: int,
                              patch_size: int,
                              save_coords: str,
                              overlap: int = 0,
                              min_tissue_proportion: float = 0.0):
        """
        从 self.tissue_seg_path 读取组织面域；按网格生成 patch，
        计算与组织相交面积占比 >= 阈值则保留。
        保存到 {save_coords}/patches/{name}_patches.h5
        """
        import h5py
        import geopandas as gpd
        from shapely.ops import unary_union
        from shapely.geometry import box
        from shapely.prepared import prep

        os.makedirs(os.path.join(save_coords, "patches"), exist_ok=True)
        out_path = os.path.join(save_coords, "patches", f"{self.name}_patches.h5")

        gdf = gpd.read_file(self.tissue_seg_path)
        if gdf.empty:
            raise RuntimeError(f"Empty tissue polygons: {self.tissue_seg_path}")
        geom = unary_union(gdf.geometry.values)
        prep_geom = prep(geom)

        level = self._choose_level_for_mag(target_mag)
        W, H = map(int, self.level_dimensions[level])
        stride = max(1, int(patch_size) - int(overlap))
        xs = list(range(0, max(1, W - patch_size + 1), stride))
        ys = list(range(0, max(1, H - patch_size + 1), stride))

        coords = []
        scaled = float(self.level_downsamples[level])
        patch_area_l0 = float(patch_size * patch_size) * scaled * scaled

        for y in ys:
            yc = y + patch_size*0.5
            for x in xs:
                xc = x + patch_size*0.5
                # 用中心点快速预筛
                if not prep_geom.contains(box((xc-0.5)*scaled, (yc-0.5)*scaled, (xc+0.5)*scaled, (yc+0.5)*scaled).centroid):
                    if min_tissue_proportion <= 0:
                        continue
                if min_tissue_proportion > 0.0:
                    patch_poly = box(x*scaled, y*scaled, (x+patch_size)*scaled, (y+patch_size)*scaled)
                    inter = geom.intersection(patch_poly).area
                    if inter / patch_area_l0 < float(min_tissue_proportion):
                        continue
                coords.append((x, y, level))

        coords = np.asarray(coords, dtype=np.int32)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("coords", data=coords, compression="gzip")
            f.attrs["patch_size"] = int(patch_size)
            f.attrs["target_mag"] = int(target_mag)
            f.attrs["overlap"] = int(overlap)
            f.attrs["level"] = int(level)

        self.coords_dir = save_coords
        self.coords_path = out_path
        return out_path

    # ---------------- 3) 可视化：在缩略图上画出部分 patch ----------------
    def visualize_coords(self, coords_path: str, save_patch_viz: str):
        import h5py
        os.makedirs(save_patch_viz, exist_ok=True)
        with h5py.File(coords_path, "r") as f:
            coords = np.array(f["coords"])
            patch_size = int(f.attrs.get("patch_size", 256))
            level = int(f.attrs.get("level", max(0, self.level_count - 1)))

        W, H = map(int, self.level_dimensions[level])
        thumb = self._s.read_region((0, 0), level, (W, H)).convert("RGB")
        draw = ImageDraw.Draw(thumb)

        # 随机/均匀采样最多 1500 个框，避免绘制过慢
        if len(coords) > 1500:
            idxs = np.linspace(0, len(coords) - 1, 1500).astype(int)
            coords_draw = coords[idxs]
        else:
            coords_draw = coords

        for x, y, lvl in coords_draw:
            if lvl != level:
                continue
            draw.rectangle([x, y, x + patch_size, y + patch_size], outline=(255, 0, 0), width=1)

        out_png = os.path.join(save_patch_viz, f"{self.name}.png")
        thumb.save(out_png, "PNG")
        return out_png

    # ---------------- 4) Patch 特征提取：真实前向 ----------------
    def extract_patch_features(self,
                               patch_encoder,
                               coords_path: str,
                               save_features: str,
                               device: str = "cuda:0",
                               saveas: str = "h5",
                               batch_limit: int = 512):
        """
        从 coords 读取网格，批量 read_region -> 归一化 -> patch_encoder 前向 -> 写 .h5
        """
        import h5py
        import torch

        os.makedirs(save_features, exist_ok=True)
        with h5py.File(coords_path, "r") as f:
            coords = np.array(f["coords"], dtype=np.int32)
            patch_size = int(f.attrs["patch_size"])

        # 预处理：ImageNet 标准化（若 encoder 提供自定义 normalize，则优先用）
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(device)
        has_custom_norm = hasattr(patch_encoder, "normalize")

        patch_encoder = patch_encoder.to(device).eval()
        feats_list: List[torch.Tensor] = []
        B = max(1, int(batch_limit))

        with torch.no_grad():
            for i in range(0, len(coords), B):
                batch = coords[i:i + B]
                if len(batch) == 0:
                    continue
                ims = []
                for x, y, lvl in batch:
                    tile = self._s.read_region((int(x), int(y)), int(lvl), (patch_size, patch_size)).convert("RGB")
                    arr = np.asarray(tile, dtype=np.float32) / 255.0  # HWC
                    arr = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
                    ims.append(arr)

                im = torch.cat(ims, dim=0).to(device)
                if has_custom_norm:
                    im = patch_encoder.normalize(im)
                else:
                    im = (im - imagenet_mean) / imagenet_std

                out = patch_encoder(im)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if isinstance(out, dict):
                    out = out.get("features", out.get("feats", list(out.values())[0]))
                if isinstance(out, torch.Tensor) and out.ndim >= 2:
                    out = out.squeeze()
                feats_list.append(out.detach().float().cpu())

        if len(feats_list) == 0:
            import numpy as _np
            D = int(getattr(patch_encoder, "enc_dim", 768))
            feats = _np.zeros((0, D), dtype=_np.float32)
        else:
            import numpy as _np
            feats = _np.concatenate([t.numpy() for t in feats_list], axis=0)

        out_path = os.path.join(save_features, f"{self.name}.{saveas}")
        with h5py.File(out_path, "w") as f:
            f.create_dataset("features", data=feats, compression="gzip")
            f.create_dataset("coords", data=coords, compression="gzip")
            f.attrs["encoder"] = getattr(patch_encoder, "enc_name", "unknown")
            f.attrs["feat_dim"] = int(feats.shape[1]) if feats.size else int(getattr(patch_encoder, "enc_dim", 768))
        return out_path

    # ---------------- 5) Slide 聚合（ABMIL 等） ----------------
    def extract_slide_features(self,
                               patch_features_path: str,
                               slide_encoder,
                               device: str,
                               save_features: str):
        """
        读取 patch 特征 (N,D)，送入 slide_encoder（如 ABMIL）得到 [1,Ds] 并保存。
        """
        import h5py
        import torch

        os.makedirs(save_features, exist_ok=True)
        with h5py.File(patch_features_path, "r") as f:
            feats = np.array(f["features"])
        if feats.ndim != 2 or feats.shape[0] == 0:
            # 空特征时写全零向量
            import numpy as _np
            D = int(getattr(slide_encoder, "enc_dim", feats.shape[1] if feats.ndim == 2 else 256))
            slide_vec = _np.zeros((1, D), dtype=_np.float32)
        else:
            x = torch.from_numpy(feats).to(device)  # [N,D]
            slide_encoder = slide_encoder.to(device).eval()
            with torch.no_grad():
                try:
                    out = slide_encoder(x)            # 期望 [1,D] 或 [D]
                except Exception:
                    out = slide_encoder(x.unsqueeze(0))  # 兼容某些 (1,N,D)
            if isinstance(out, (list, tuple)):
                out = out[0]
            if isinstance(out, dict):
                out = out.get("slide_features", out.get("embedding", list(out.values())[0]))
            if isinstance(out, torch.Tensor):
                vec = out.detach().float().view(1, -1).cpu().numpy()
            else:
                import numpy as _np
                vec = _np.asarray(out, dtype=_np.float32).reshape(1, -1)
            slide_vec = vec

        out_path = os.path.join(save_features, f"{self.name}.h5")
        with h5py.File(out_path, "w") as f:
            f.create_dataset("slide_features", data=slide_vec, compression="gzip")
            f.attrs["encoder"] = getattr(slide_encoder, "enc_name", "abmil")
            f.attrs["feat_dim"] = int(slide_vec.shape[1])
        return out_path

    # ---------------- 资源释放（可选） ----------------
    def release(self) -> None:
        try:
            if hasattr(self._s, "close"):
                self._s.close()
        except Exception:
            pass


__all__ = ["AslideWSI"]
