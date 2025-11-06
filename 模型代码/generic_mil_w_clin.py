import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Tuple

try:
    import h5py
except Exception:
    h5py = None

SUPPORTED_EXTS = ('.h5', '.hdf5', '.pt', '.pth', '.npz')

class Generic_MIL_Dataset_Clin(Dataset):
    """
    每个样本 = 一个 slide (bag)。
    支持 HDF5/PT/NPZ；label 若缺失，从 CSV 指定列读取。
    clin 向量支持可选扩展列，并支持外部传入的标准化变换（避免数据泄漏）。
    """

    # 基础临床列与可选列（说明性；当前通过 CSV 自动解析，不强依赖）
    BASE_CLIN_COLS = ["NIH评估", "年龄", "部位", "大小", "肿瘤破裂", "肝脏转移", "腹腔播散", "坏死", "基因检测"]
    OPTIONAL_CLIN_COLS = ["核异型性编码", "形态学评估编码"]

    def __init__(self,
                 data_dir: str,
                 clinical_csv: str,
                 slide_ids: Optional[List[str]] = None,
                 max_patches: Optional[int] = None,
                 label_col: str = 'label',
                 num_classes: int = 2,
                 clin_transform: Optional[callable] = None):
        self.data_dir = data_dir
        self.df = pd.read_csv(clinical_csv)
        self.df.set_index('slide_id', inplace=True)

        self.label_col = label_col
        self.num_classes = num_classes

        # 这些列视为类别/指示变量，用 one-hot，不做标准化
        self.CAT_COLS = ["部位", "肿瘤破裂", "肝脏转移", "腹腔播散", "坏死", "基因检测", "核异型性编码", "形态学评估编码"]

        # one-hot（仅对存在的列）
        self.df = pd.get_dummies(
            self.df,
            columns=[c for c in self.CAT_COLS if c in self.df.columns],
            prefix=[c for c in self.CAT_COLS if c in self.df.columns],
            dtype=float
        )

        # clin 列：除标签列外的所有特征列（包含 one-hot 后的列）
        self.clin_cols = [c for c in self.df.columns if c != self.label_col]

        # 记录分类列前缀；连续列 = 非这些前缀的列
        self.cat_prefixes = tuple(f"{c}_" for c in self.CAT_COLS)
        self.cont_cols = [c for c in self.clin_cols if not c.startswith(self.cat_prefixes)]

        # 标签是否需要从 1..C 平移到 0..C-1
        if self.label_col in self.df.columns:
            try:
                vals = self.df[self.label_col].dropna().astype(int).values
                if len(vals) > 0 and vals.min() == 1 and vals.max() <= self.num_classes:
                    self._label_shift = -1
                else:
                    self._label_shift = 0
            except Exception:
                self._label_shift = 0
        else:
            self._label_shift = 0

        # 收集 slide 列表（以 data_dir 下的文件名为准，或使用传入的 slide_ids）
        if slide_ids is None:
            names = []
            for fn in os.listdir(data_dir):
                for ext in SUPPORTED_EXTS:
                    if fn.endswith(ext):
                        names.append(fn[:-len(ext)])
                        break
            slide_ids = sorted(list(set(names)))

        # 仅保留：CSV 中存在且能解析到文件路径的 slide
        self.slide_ids = [sid for sid in slide_ids if sid in self.df.index and self._resolve_path(sid) is not None]
        self.max_patches = max_patches

        # 可选：临床向量的变换（如折内 StandardScaler.transform）
        self.clin_transform = clin_transform

    def __len__(self):
        return len(self.slide_ids)

    def _resolve_path(self, sid: str) -> Optional[str]:
        """根据 slide_id 在 data_dir 下推断真实文件路径"""
        for ext in SUPPORTED_EXTS:
            p = os.path.join(self.data_dir, sid + ext)
            if os.path.exists(p):
                return p
        return None

    @staticmethod
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
        if isinstance(x, np.ndarray):
            return x.astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _first_key(d: Dict, keys: List[str]):
        """安全地在 dict 里按顺序取第一个存在且非 None 的键；不做布尔运算，避免 Tensor 触发 bool 错误。"""
        if not isinstance(d, dict):
            return None
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    def _load_any(self, path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
        """读取任意支持格式，返回 (feat[N,D], xy[N,2]|None, label|None)"""
        ext = os.path.splitext(path)[1].lower()
        feat = None
        xy = None
        label = None

        if ext in ('.h5', '.hdf5'):
            if h5py is None:
                raise ImportError('需要安装 h5py 以读取 HDF5')
            with h5py.File(path, 'r') as f:
                key_feat = 'feat' if 'feat' in f else ('features' if 'features' in f else None)
                if key_feat is None:
                    raise KeyError(f"{path} 缺少 'feat' 或 'features'")
                feat = np.array(f[key_feat])
                # 坐标键兜底
                for k in ('xy', 'coords', 'coord', 'positions', 'xy_topleft'):
                    if k in f:
                        xy = np.array(f[k])
                        break
                # 标签（可选）
                if 'label' in f:
                    label = int(np.array(f['label']).reshape(()))

        elif ext in ('.pt', '.pth'):
            obj = torch.load(path, map_location='cpu')

            # 顶层就是 Tensor：仅有特征
            if torch.is_tensor(obj):
                feat = obj
                xy = None
                label = None

            elif isinstance(obj, dict):
                # 用 _first_key 避免 "Tensor or" 布尔错误
                feat = self._first_key(obj, ['feat', 'features', 'feats', 'x', 'data'])
                xy = self._first_key(obj, ['xy', 'coords', 'coord', 'positions', 'xy_topleft'])
                lab = self._first_key(obj, ['label', 'y', 'target'])
                if lab is not None:
                    try:
                        label = int(lab)
                    except Exception:
                        label = None

            elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
                feat, xy = obj[0], obj[1]
                if len(obj) >= 3:
                    try:
                        label = int(obj[2])
                    except Exception:
                        label = None
            else:
                raise ValueError(f"Unrecognized PT format: {path}")

            # 统一到 numpy.float32
            feat = self._to_np(feat)
            if xy is not None:
                xy = self._to_np(xy)
                if xy.ndim == 1 and xy.size == 2:
                    xy = xy[None, :]  # [2] -> [1,2]

        elif ext == '.npz':
            d = np.load(path)
            if 'feat' in d.files:
                feat = d['feat']
            elif 'features' in d.files:
                feat = d['features']
            else:
                raise KeyError(f"{path} 缺少 'feat/features'")

            xy = None
            for k in ('xy', 'coords', 'coord', 'positions', 'xy_topleft'):
                if k in d.files:
                    xy = d[k]
                    break

            label = int(d['label']) if 'label' in d.files else None

        else:
            raise ValueError(f"Unsupported ext: {ext}")

        # 类型与形状整理
        feat = np.asarray(feat, dtype=np.float32)
        if xy is not None:
            xy = np.asarray(xy, dtype=np.float32)
            if xy.ndim == 1 and xy.size == 2:
                xy = xy[None, :]
        return feat, xy, label

    def _clin_vec(self, row: pd.Series) -> np.ndarray:
        """按列顺序取临床特征向量；如配置了 transform，则只对连续列或整体应用（由外部保证）。"""
        vec = row[self.clin_cols].astype(float).values.astype(np.float32)
        if self.clin_transform is not None:
            v = self.clin_transform(vec)  # 期望输入输出均为 1D np.ndarray
            vec = np.asarray(v, dtype=np.float32)
        return vec

    def __getitem__(self, idx: int) -> Dict:
        sid = self.slide_ids[idx]
        path = self._resolve_path(sid)
        feat, xy, label = self._load_any(path)

        # 若坐标缺失，填零占位（后续可视化/模型若必需坐标，你可以在 forward 前判断）
        if xy is None:
            xy = np.zeros((feat.shape[0], 2), dtype=np.float32)

        # 子采样（如设置了 max_patches）
        if self.max_patches is not None and feat.shape[0] > self.max_patches:
            sel = np.random.choice(feat.shape[0], self.max_patches, replace=False)
            feat = feat[sel]
            xy = xy[sel]

        # 临床向量
        clin = self._clin_vec(self.df.loc[sid])

        # 标签兜底：优先文件内；否则从 CSV 取并做 0/1 平移
        if label is None:
            if self.label_col in self.df.columns:
                label = int(self.df.loc[sid, self.label_col]) + self._label_shift
            else:
                raise ValueError(f"{sid} 缺少 label：请在 H5/PT/NPZ 或 clinical.csv 中提供（或指定 --label-col）")

        return {
            'slide_id': sid,
            'feat': torch.from_numpy(feat),
            'xy': torch.from_numpy(xy),                  # 始终有形状 [N,2]
            'clin': torch.from_numpy(clin),
            'label': torch.tensor(label, dtype=torch.long),
        }
