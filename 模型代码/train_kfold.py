import os
import json
import math
import argparse
from datetime import datetime
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    confusion_matrix, balanced_accuracy_score, classification_report
)

from generic_mil_w_clin import Generic_MIL_Dataset_Clin
from gabmil_fusion import GABMILFusion
# === 缺失样本预检查（保存到结果目录） ===
from generic_mil_w_clin import SUPPORTED_EXTS  # 复用数据集里定义的后缀
import math
import torch
import torch.nn as nn
from typing import Dict

def attention_entropy_loss(alpha: torch.Tensor, mode: str = "max") -> torch.Tensor:
    """
    alpha: [N]  实例维的注意力分布，sum(alpha)=1
    mode:
      - "max":  鼓励“更平滑/更均匀”的注意力（避免塌缩）。实现为 KL(alpha || Uniform)。
                该损失 >= 0，alpha=均匀时为 0；值越大表示越尖锐。
      - "min":  鼓励“更稀疏/更尖锐”的注意力（更专注于少数实例）。实现为 归一化的熵 H(alpha)/log N。
                该损失 ∈ [0,1]，越小越尖锐（delta 分布为 0），越大越平滑（均匀为 1）。
    """
    eps = 1e-8
    N = alpha.numel()
    if mode == "max":
        # KL(alpha || U) = sum alpha * log(alpha / (1/N)) = sum alpha*log(alpha) + log N
        kl = (alpha * (alpha + eps).log()).sum() + math.log(N)
        return kl / math.log(N)  # 归一化到 ~[0, +∞)
    elif mode == "min":
        # 归一化熵：H(alpha)/log N，范围 [0,1]（均匀=1，delta=0）
        H = -(alpha * (alpha + eps).log()).sum()
        return H / math.log(N)
    else:
        raise ValueError("mode must be 'max' or 'min'")
    
def soft_dice_multiclass_with_logits(
    logits: torch.Tensor, target: torch.Tensor,
    class_weight: torch.Tensor = None, eps: float = 1e-6,
    present_only: bool = True
) -> torch.Tensor:
    """
    多分类 Soft Dice（macro；可选只对本 batch 出现的类求平均）。
    logits: [C] 或 [B,C]
    target: [ ] 或 [B]，取值 0..C-1
    class_weight: 可选类权（建议均值归一）
    present_only: True => 仅对本 batch 里出现过的类别做平均（强烈建议在 batch 很小时启用）
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)  # [1,C]
        target = target.view(1)       # [1]
    B, C = logits.shape
    p = torch.softmax(logits, dim=1)             # [B,C]
    onehot = torch.zeros_like(p).scatter_(1, target.view(-1,1), 1.0)  # [B,C]

    inter = (p * onehot).sum(dim=0)              # [C]
    denom = (p + onehot).sum(dim=0)              # [C]
    dice_c = (2*inter + eps) / (denom + eps)     # [C]
    loss_c = 1.0 - dice_c                        # [C]

    # 仅对“本 batch 出现过的类别”求平均
    if present_only:
        present = (onehot.sum(dim=0) > 0)        # [C] bool
    else:
        present = torch.ones(C, dtype=torch.bool, device=logits.device)

    if class_weight is not None:
        w = class_weight.to(logits.device)
        w = w / (w[present].mean() + 1e-12)      # 只对参与平均的类做均值归一
        return (loss_c[present] * w[present]).mean()

    return loss_c[present].mean()

def _list_file_ids(data_root_dir):
    ids = []
    for fn in os.listdir(data_root_dir):
        for ext in SUPPORTED_EXTS:
            if fn.endswith(ext):
                ids.append(fn[:-len(ext)])
                break
    return sorted(set(ids))

def _list_csv_ids(clinical_csv):
    import pandas as pd
    df = pd.read_csv(clinical_csv)
    # 注意 strip 空白，避免 "20S04320 " 这种带空格的情况
    s = df['slide_id'].astype(str).str.strip()
    return sorted(set(s.tolist()))

def write_missing_report(args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    file_ids = set(_list_file_ids(args.data_root_dir))
    csv_ids  = set(_list_csv_ids(args.clinical_csv))

    only_in_csv   = sorted(csv_ids - file_ids)
    only_in_files = sorted(file_ids - csv_ids)
    matched       = sorted(csv_ids & file_ids)

    # 文本版
    txt_path = os.path.join(out_dir, "missing_ids.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"[Dataset] files={len(file_ids)}, csv_ids={len(csv_ids)}, matched={len(matched)}\n")
        f.write("\n# CSV有但没有特征文件（请补特征或从CSV移除）:\n")
        for sid in only_in_csv:
            f.write(sid + "\n")
        f.write("\n# 目录里有特征但不在CSV（请补CSV或移走特征）:\n")
        for sid in only_in_files:
            f.write(sid + "\n")

    # JSON 版（便于程序化查看）
    json_path = os.path.join(out_dir, "missing_ids.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "files": len(file_ids),
            "csv_ids": len(csv_ids),
            "matched": len(matched),
            "csv_without_features": only_in_csv,
            "features_not_in_csv": only_in_files
        }, f, ensure_ascii=False, indent=2)

    print(f"[Precheck] saved missing-id report -> {txt_path}")

# --------------------------
# 1) 命令行参数
# --------------------------
def parse_args():
    ap = argparse.ArgumentParser("GABMIL Fusion - 5-Fold Trainer")
    ap.add_argument('--data_root_dir', type=str, required=True, help='特征目录（h5/pt/npz）')
    ap.add_argument('--clinical_csv', type=str, required=True, help='包含 slide_id 与临床特征（含标签列）')
    ap.add_argument('--results_dir', type=str, default='./results_kfold')
    ap.add_argument('--label-col', type=str, default='label', help='CSV 中用作标签的列名（如 NIH评估）')
    ap.add_argument('--num_classes', type=int, default=4, help='类别数（四分类=4）')
    ap.add_argument('--n_splits', type=int, default=5, help='折数（默认为5）')
    ap.add_argument('--test_ids', type=str, default=None,
                help='可选：测试集 slide_id 文本文件（每行一个 slide_id）。若提供，将用各折最优模型做概率集成评估。')
    # 模型超参
    ap.add_argument('--embed_dim', type=int, default=512)
    ap.add_argument('--att_dim', type=int, default=128)
    ap.add_argument('--fusion_hidden', type=int, default=256)
    ap.add_argument('--gated_att', action='store_true', default=True)
    ap.add_argument('--use_simm', action='store_true', help='未来有坐标时启用 SIMM（需在模型里注入）')

    # 训练超参
    ap.add_argument('--batch_size', type=int, default=1)  # MIL 通常 1
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--max_patches', type=int, default=None)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--patience', type=int, default=5, help='早停容忍轮数')

    # 记录/保存
    ap.add_argument('--save_att', action='store_true', help='保存注意力权重（bag 内 patch 级），用于可视化')
    # —— 损失配置 —— #
    ap.add_argument('--loss', choices=['ce', 'dice_ce'], default='ce',help='ce=仅交叉熵；dice_ce=交叉熵+Soft Dice（推荐长尾多分类）')
    ap.add_argument('--lambda_dice', type=float, default=0.5,help='Dice 的权重，常用 0.3~1.0')
    ap.add_argument('--device', type=str, default='cuda:0',help="目标设备，如：'cuda:0' / 'cuda:1' / 'cuda' / 'cpu'")
    ap.add_argument('--monitor', choices=['AUC', 'F1', 'ACC', 'bACC'], default='AUC',help='验证集早停/调度监控指标（最大化）')

    return ap.parse_args()

# --------------------------
# 2) 工具函数
# --------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fit_clin_scaler(train_ids: List[str], df: pd.DataFrame, cols: List[str]) -> StandardScaler:
    X = df.loc[train_ids, cols].astype(float).to_numpy()
    scaler = StandardScaler()
    scaler.fit(X)
    # 统一处理零方差列：训练/验证/测试都用同一套数值
    eps = 1e-8
    scale = scaler.scale_.copy()
    scale[scale < eps] = eps
    scaler.scale_ = scale
    scaler.var_ = scaler.scale_ ** 2  # sklearn 需要 var_ 与 scale_ 一致
    return scaler

def make_np_transform(scaler: StandardScaler):
    """
    把 sklearn 的 transform 包一层，适配我们 Dataset 里 1D np.ndarray -> 1D np.ndarray 的调用。
    """
    def _f(v: np.ndarray) -> np.ndarray:
        return scaler.transform(v.reshape(1, -1)).reshape(-1)
    return _f
def make_np_transform_partial(scaler: StandardScaler, cont_idx, log1p: bool = True):
    def _f(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32).copy()
        if len(cont_idx):
            sub = v[cont_idx].astype(np.float32)
            if log1p: sub = np.log1p(sub)
            v[cont_idx] = scaler.transform(sub.reshape(1, -1)).reshape(-1).astype(np.float32)
        return v
    return _f

def make_results_root(args) -> str:
    tag = f"{os.path.basename(args.data_root_dir)}_{args.label_col}_C{args.num_classes}_K{args.n_splits}"
    ts  = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.results_dir, f"{tag}_{ts}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    return out


def build_model(in_dim: int, clin_dim: int, args):
    """
    构建 GABMIL Fusion 模型。
    当 args.use_simm=True 时，自动加载 SIMM。
    """
    simm_module = None
    if args.use_simm:
        try:
            from models.simm import SIMM
            print("[GABMIL] use_simm=True，已成功加载 SIMM 模块。")
            simm_module = SIMM(
                in_dim=args.embed_dim,
                hidden_dim=args.embed_dim,
                num_heads=4,
                block_kernel=3,
                dropout=0.1,
                use_block=True,
                use_grid=True,
            )
        except Exception as e:
            print(f"[警告] 加载 SIMM 失败，改为直通模式：{e}")

    model = GABMILFusion(
        in_dim=in_dim,
        embed_dim=args.embed_dim,
        att_dim=args.att_dim,
        clin_dim=clin_dim,
        num_classes=args.num_classes,
        gated_att=args.gated_att,
        use_simm=args.use_simm,
        simm_module=simm_module,
        fusion_hidden=args.fusion_hidden,
    )
    return model

def get_class_weights(train_ids: List[str], df: pd.DataFrame, label_col: str, num_classes: int, label_shift: int):
    ys = [int(df.loc[sid, label_col]) + label_shift for sid in train_ids]
    binc = np.bincount(ys, minlength=num_classes).astype(np.float32)
    w = 1.0 / (binc + 1e-6)
    w = w * (num_classes / w.sum())
    return torch.tensor(w, dtype=torch.float)


def train_one_epoch(model, loader, opt, device, num_classes,
                    cls_w: torch.Tensor = None, lambda_att: float = 0.0,
                    loss_mode: str = 'ce', lambda_dice: float = 0.0,
                    amp: bool = True, max_grad_norm: float = 5.0):
    model.train()
    enabled = bool(amp and device.type == 'cuda')
    try:
        # PyTorch 2.x 推荐写法：设备类型作为第一个位置参数
        scaler = torch.amp.GradScaler('cuda', enabled=enabled)
    except TypeError:
        # 兼容旧版（只有 torch.cuda.amp.GradScaler）
        scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    tot = 0.0
    for batch in loader:
        feat = batch['feat'][0].to(device, non_blocking=True)
        xy = None
        if 'xy' in batch and (batch['xy'][0] is not None):
            xy = batch['xy'][0].to(device, non_blocking=True)
        clin = batch['clin'][0].to(device, non_blocking=True)
        y    = batch['label'].to(device, non_blocking=True).view(-1).long()

        # 仅包前向和loss
        with torch.amp.autocast(device_type='cuda', enabled=(amp and device.type == 'cuda')):
            logits, aux = model(feat, xy, clin)
            ce = F.cross_entropy(logits.unsqueeze(0), y, weight=cls_w)
            if loss_mode == 'dice_ce':
                dice = soft_dice_multiclass_with_logits(logits, y, class_weight=cls_w, present_only=True)
                loss = ce + lambda_dice * dice
            else:
                loss = ce
            if lambda_att > 0 and 'att' in aux:
                A = aux['att']; eps = 1e-8
                att_entropy = -(A * (A + eps).log()).sum() / A.numel()
                loss = loss + lambda_att * att_entropy

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(opt)
        scaler.update()
        tot += float(loss.detach().item())
    return tot / max(len(loader), 1)

@torch.no_grad()
def evaluate(model, loader, device, num_classes) -> Dict:
    model.eval()
    ys, preds, probs, sids = [], [], [], []
    atts = []  # 图像patch的注意力
    clin_atts = []  # 临床特征的注意力
    for batch in loader:
        feat = batch['feat'][0].to(device,non_blocking=True)
        xy = None
        if 'xy' in batch and (batch['xy'][0] is not None):
            xy = batch['xy'][0].to(device, non_blocking=True)

        clin = batch['clin'][0].to(device,non_blocking=True)
        y    = int(batch['label'].item())
        sid  = batch['slide_id'][0] if isinstance(batch['slide_id'], list) else batch['slide_id']

        logits, aux = model(feat, xy, clin)
        if num_classes >= 2:
            p = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # [C]
            pred = int(np.argmax(p))
        else:
            s = torch.sigmoid(logits.view(()))
            p = np.array([1.0 - s.item(), s.item()])
            pred = int(s.item() >= 0.5)

        ys.append(y); preds.append(pred); probs.append(p); sids.append(sid)
        if 'att' in aux:
            atts.append(aux['att'].detach().cpu().numpy())  # [N]
        if 'clin_att' in aux:
            clin_atts.append(aux['clin_att'].detach().cpu().numpy())  # [D] 临床特征维度

    ys_arr = np.array(ys)
    preds_arr = np.array(preds)
    P = np.stack(probs) if len(probs) else np.zeros((0, num_classes))

    metrics = {}
    metrics['ACC']  = float(accuracy_score(ys_arr, preds_arr)) if len(ys_arr) else float('nan')
    metrics['bACC'] = float(balanced_accuracy_score(ys_arr, preds_arr)) if len(ys_arr) else float('nan')
    metrics['F1']   = float(f1_score(ys_arr, preds_arr, average='macro')) if len(ys_arr) else float('nan')
    try:
        if num_classes == 2:
            auc = roc_auc_score(ys_arr, P[:, 1])
        else:
            auc = roc_auc_score(ys_arr, P, multi_class='ovr', average='macro')
        metrics['AUC'] = float(auc)
    except Exception:
        metrics['AUC'] = float('nan')

    cm = confusion_matrix(ys_arr, preds_arr, labels=list(range(num_classes))) if len(ys_arr) else np.zeros((num_classes, num_classes))
    report = classification_report(ys_arr, preds_arr, labels=list(range(num_classes)), output_dict=True, zero_division=0) if len(ys_arr) else {}

    return {
        "metrics": metrics,
        "y_true": ys_arr,
        "y_pred": preds_arr,
        "probs": P,
        "slide_ids": sids,
        "conf_mat": cm,
        "report": report,
        "atts": atts,
        "clin_atts": clin_atts  # 临床特征注意力
    }

# --------------------------
# 4) 主流程（5折）
# --------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    # 优先按用户指定
    if args.device.startswith('cuda'):
        assert torch.cuda.is_available(), "未检测到 CUDA，请改用 --device cpu"
        device = torch.device(args.device)          # 例如 cuda:1
        # 可选：显式把当前进程默认 GPU 设为该卡
        if device.index is not None:
            torch.cuda.set_device(device.index)
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        # 回退逻辑：自动选择
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] using {device}")

    # 先创建本次结果根目录（时间戳）
    root = make_results_root(args)

    # 写入缺失样本清单
    write_missing_report(args, out_dir=root)

    # 再构建全量数据集（用于拿到 id 列表 & 维度）

    # 先构建全量数据集（用于拿到 id 列表 & 维度）
    ds_all = Generic_MIL_Dataset_Clin(
        args.data_root_dir, args.clinical_csv,
        slide_ids=None, max_patches=args.max_patches,
        label_col=args.label_col, num_classes=args.num_classes
    )
    # 把标签转成 0..C-1（与 Dataset 一致：用 _label_shift）
    df = ds_all.df
    label_shift = getattr(ds_all, "_label_shift", 0)
    all_ids = ds_all.slide_ids
    y_all = np.array([int(df.loc[sid, args.label_col]) + label_shift for sid in all_ids], dtype=int)

    # 用一个样本推断维度
    sample = ds_all[0]
    in_dim = sample['feat'].shape[1]
    clin_dim = sample['clin'].numel()

    # 分层K折（外层：做“测试折”）
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    # 用于跨折汇总
    all_oof_val_rows = []   # 折内验证集（val）的逐样本结果（内层）
    all_cvtest_rows  = []   # 折内测试集（test）的逐样本结果（外层）
    fold_summaries   = []   # 每折 best 的验证指标汇总
    fold_test_metrics = []  # 每折测试指标（方便打印/汇总）

    for fold, (trval_idx, te_idx) in enumerate(skf.split(all_ids, y_all), start=1):
        fold_dir = os.path.join(root, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # 外层：当前折的“测试集”
        te_ids = [all_ids[i] for i in te_idx]

        # 内层：在剩余 80% 上再分 stratified train/val（按 75/25 -> 全局 60/20）
        trval_ids = [all_ids[i] for i in trval_idx]
        y_trval   = np.array([int(df.loc[sid, args.label_col]) + label_shift for sid in trval_ids], dtype=int)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed + fold)
        inner_tr_idx, inner_va_idx = next(sss.split(trval_ids, y_trval))
        tr_ids = [trval_ids[i] for i in inner_tr_idx]
        va_ids = [trval_ids[i] for i in inner_va_idx]

        # —— 拟合本折的临床标准化（仅用训练集） —— #
        # —— 拟合本折的临床标准化（仅用训练集） —— #
        # 使用 Dataset 里记录的列；只挑“连续列”做标准化
        # （Dataset 在 one-hot 后提供了 clin_cols / cont_cols / cat_prefixes）
        if hasattr(ds_all, "cont_cols"):
            cont_cols = [c for c in ds_all.cont_cols if c in ds_all.df.columns]
        else:
            # 兼容旧逻辑：若没有 cont_cols，就仅把“大小”当连续列
            cont_cols = [c for c in ["大小"] if c in ds_all.df.columns]

        if len(cont_cols) == 0:
            scaler = None
            clin_tf = None
        else:
            # 1) 仅用训练集在连续列上拟合（log1p 后再标准化）
            X = ds_all.df.loc[tr_ids, cont_cols].astype(float).to_numpy()
            X_fit = np.log1p(X)
            scaler = StandardScaler().fit(X_fit)

            # 防零方差
            eps = 1e-8
            scale = scaler.scale_.copy()
            scale[scale < eps] = eps
            scaler.scale_ = scale
            scaler.var_   = scaler.scale_ ** 2

            # 2) 构造“只动连续列”的 transform
            full_cols = ds_all.clin_cols
            cont_idx = [full_cols.index(c) for c in cont_cols]
            clin_tf = make_np_transform_partial(scaler, cont_idx, log1p=True)

            # 3) 保存本折 scaler
            with open(os.path.join(fold_dir, "scaler.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "cont_cols": cont_cols,
                    "full_clin_cols": full_cols,
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "log1p": True
                }, f, ensure_ascii=False, indent=2)


        # —— 本折数据集/加载器：train/val/test —— #
        ds_tr = Generic_MIL_Dataset_Clin(args.data_root_dir, args.clinical_csv, tr_ids,
                                        max_patches=args.max_patches,
                                        label_col=args.label_col, num_classes=args.num_classes,
                                        clin_transform=clin_tf)
        ds_va = Generic_MIL_Dataset_Clin(args.data_root_dir, args.clinical_csv, va_ids,
                                        max_patches=None,
                                        label_col=args.label_col, num_classes=args.num_classes,
                                        clin_transform=clin_tf)
        ds_te = Generic_MIL_Dataset_Clin(args.data_root_dir, args.clinical_csv, te_ids,
                                        max_patches=None,
                                        label_col=args.label_col, num_classes=args.num_classes,
                                        clin_transform=clin_tf)
        pin = (device.type == 'cuda')
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
        dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

        # —— 模型、优化器、调度器、类权重 —— #
        model = build_model(in_dim, clin_dim, args).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sche = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)
        last_lr = opt.param_groups[0]["lr"]
        cls_w = get_class_weights(tr_ids, ds_tr.df, args.label_col, args.num_classes,
                                getattr(ds_tr, "_label_shift", 0)).to(device)

        # —— 训练（以验证 F1 早停） —— #
        log_path = os.path.join(fold_dir, "metrics_per_epoch.csv")
        log_rows = []
        best_score, best_epoch, best_loss= -1, -1 ,-1
        best_val_blob = None
        bad = 0

        for ep in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, dl_tr, opt, device, args.num_classes, cls_w=cls_w, lambda_att=1e-3, loss_mode=args.loss, lambda_dice=args.lambda_dice)

            val_blob = evaluate(model, dl_va, device, args.num_classes)
            val = val_blob["metrics"]

            # 用选择的指标
            score = float(val[args.monitor])   # <= 这里不再写死 AUC

            # 记录日志
            log_rows.append({
                "epoch": ep, "train_loss": tr_loss, "val_AUC": val["AUC"],
                "val_ACC": val["ACC"], "val_bACC": val["bACC"], "val_F1": val["F1"],
                "lr": opt.param_groups[0]["lr"]
            })
            pd.DataFrame(log_rows).to_csv(log_path, index=False)
            print(f"[Fold {fold}] Epoch {ep:02d} | loss={tr_loss:.4f} | val={val}")

            sche.step(score)
            new_lr = opt.param_groups[0]["lr"]
            if new_lr != last_lr:
                print(f"[Fold {fold}] LR reduced: {last_lr:.2e} -> {new_lr:.2e} (val_score={score:.4f})")
                last_lr = new_lr
            improve_thresh = 1e-4
            if score > best_score :
                best_score, best_epoch, best_loss = score, ep, tr_loss
                best_val_blob = val_blob
                torch.save({
                    "model": model.state_dict(),
                    "0": in_dim, "clin_dim": clin_dim,
                    "args": vars(args), "fold": fold, "epoch": ep
                }, os.path.join(fold_dir, "best_model.pth"))
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"[Fold {fold}] Early stopped at epoch {ep}. Best epoch = {best_epoch}")
                    break

        # —— 保存“验证集”的最佳结果（OOF-val） —— #
        bv = best_val_blob
        probs_cols = [f"prob_c{i}" for i in range(args.num_classes)]
        rows_val = []
        for sid, y, yhat, p in zip(bv["slide_ids"], bv["y_true"], bv["y_pred"], bv["probs"]):
            r = {"fold": fold, "split": "val", "slide_id": sid, "y_true": int(y), "y_pred": int(yhat)}
            r.update({c: float(v) for c, v in zip(probs_cols, p)})
            rows_val.append(r)
        pd.DataFrame(rows_val).to_csv(os.path.join(fold_dir, "val_preds.csv"), index=False)
        cm_df = pd.DataFrame(bv["conf_mat"], index=[f"true_{i}" for i in range(args.num_classes)],
                            columns=[f"pred_{i}" for i in range(args.num_classes)])
        cm_df.to_csv(os.path.join(fold_dir, "confusion_matrix_val.csv"))
        pd.DataFrame(bv["report"]).to_csv(os.path.join(fold_dir, "classification_report_val.csv"))

        fold_summaries.append({"fold": fold, "best_epoch": best_epoch, **bv["metrics"]})
        all_oof_val_rows.extend(rows_val)

        # —— 用“本折最优模型”在本折测试集评估 —— #
        # （此处 model 已是最佳参数；如你更严谨，亦可重新 load best_model.pth 再 eval）
        # —— 用“本折最优模型”在本折测试集评估 —— #
        ckpt = torch.load(os.path.join(fold_dir, "best_model.pth"), map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        te_blob = evaluate(model, dl_te, device, args.num_classes)

        rows_te = []
        for sid, y, yhat, p in zip(te_blob["slide_ids"], te_blob["y_true"], te_blob["y_pred"], te_blob["probs"]):
            r = {"fold": fold, "split": "test", "slide_id": sid, "y_true": int(y), "y_pred": int(yhat)}
            r.update({c: float(v) for c, v in zip(probs_cols, p)})
            rows_te.append(r)
        pd.DataFrame(rows_te).to_csv(os.path.join(fold_dir, "test_preds.csv"), index=False)

        cm_te = pd.DataFrame(te_blob["conf_mat"], index=[f"true_{i}" for i in range(args.num_classes)],
                            columns=[f"pred_{i}" for i in range(args.num_classes)])
        cm_te.to_csv(os.path.join(fold_dir, "confusion_matrix_test.csv"))
        pd.DataFrame(te_blob["report"]).to_csv(os.path.join(fold_dir, "classification_report_test.csv"))

        # 记录本折测试指标（便于打印/汇总）
        fold_test_metrics.append({"fold": fold, **te_blob["metrics"]})
        all_cvtest_rows.extend(rows_te)

        # 注意力可选保存
        if args.save_att and 'atts' in bv and len(bv["atts"]) == len(bv["slide_ids"]):
            np.savez_compressed(os.path.join(fold_dir, "att_weights_val.npz"),
                                **{sid: att for sid, att in zip(bv["slide_ids"], bv["atts"])})
        # 保存临床特征注意力
        if args.save_att and 'clin_atts' in bv and len(bv["clin_atts"]) == len(bv["slide_ids"]):
            np.savez_compressed(os.path.join(fold_dir, "clin_att_weights_val.npz"),
                                **{sid: att for sid, att in zip(bv["slide_ids"], bv["clin_atts"])})

    # —— 跨折汇总 —— #
    # OOF 逐样本表
    oof_df = pd.DataFrame(all_oof_val_rows)
    oof_path = os.path.join(root, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)

    # OOF 指标
    y_true = oof_df["y_true"].to_numpy()
    y_pred = oof_df["y_pred"].to_numpy()
    P = oof_df[[c for c in oof_df.columns if c.startswith("prob_c")]].to_numpy()

    summary = {}
    summary["OOF_ACC"]  = float(accuracy_score(y_true, y_pred)) if len(y_true) else float('nan')
    summary["OOF_bACC"] = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else float('nan')
    summary["OOF_F1"]   = float(f1_score(y_true, y_pred, average='macro')) if len(y_true) else float('nan')
    try:
        if args.num_classes == 2:
            summary["OOF_AUC"] = float(roc_auc_score(y_true, P[:, 1]))
        else:
            summary["OOF_AUC"] = float(roc_auc_score(y_true, P, multi_class='ovr', average='macro'))
    except Exception:
        summary["OOF_AUC"] = float('nan')

    # 各折 best 概览
    folds_df = pd.DataFrame(fold_summaries)
    folds_df.to_csv(os.path.join(root, "fold_best_summary.csv"), index=False)

    # OOF 汇总表
    with open(os.path.join(root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"overall": summary}, f, ensure_ascii=False, indent=2)
    print("\n=== K-Fold Summary ===")
    print(summary)
    print(f"Files saved under: {root}")
    # ====== 追加：跨折“测试集（外层20%）”汇总 ======
    cvtest_df = pd.DataFrame(all_cvtest_rows)
    cvtest_path = os.path.join(root, "cv_test_predictions.csv")
    cvtest_df.to_csv(cvtest_path, index=False)

    if len(cvtest_df):
        y_true_te = cvtest_df["y_true"].to_numpy()
        y_pred_te = cvtest_df["y_pred"].to_numpy()
        P_te = cvtest_df[[c for c in cvtest_df.columns if c.startswith("prob_c")]].to_numpy()

        cvtest_summary = {
            "CVTEST_ACC":  float(accuracy_score(y_true_te, y_pred_te)),
            "CVTEST_bACC": float(balanced_accuracy_score(y_true_te, y_pred_te)),
            "CVTEST_F1":   float(f1_score(y_true_te, y_pred_te, average='macro'))
        }
        try:
            if args.num_classes == 2:
                cvtest_summary["CVTEST_AUC"] = float(roc_auc_score(y_true_te, P_te[:, 1]))
            else:
                cvtest_summary["CVTEST_AUC"] = float(roc_auc_score(y_true_te, P_te, multi_class='ovr', average='macro'))
        except Exception:
            cvtest_summary["CVTEST_AUC"] = float('nan')

        with open(os.path.join(root, "cv_test_summary.json"), "w", encoding="utf-8") as f:
            json.dump(cvtest_summary, f, ensure_ascii=False, indent=2)

        # 也把每折的测试指标表存一下
        pd.DataFrame(fold_test_metrics).to_csv(os.path.join(root, "fold_test_metrics.csv"), index=False)

        print("\n=== 5-Fold CV (per-fold TEST) Summary ===")
        print(cvtest_summary)

        # ========== 可选：独立测试集 + 多折集成 ==========
    if args.test_ids is not None and os.path.isfile(args.test_ids):
        with open(args.test_ids, 'r', encoding='utf-8') as f:
            test_ids = [ln.strip() for ln in f if ln.strip()]
        # 仅保留既在 CSV 又有特征文件的 id
        test_ids = [sid for sid in test_ids if sid in ds_all.slide_ids]
        if len(test_ids) == 0:
            print("[Test] 提供的 test_ids 均不在可用样本中，跳过测试阶段。")
        else:
            # 累加每折预测用于集成
            agg_probs: Dict[str, np.ndarray] = {}
            y_true_map: Dict[str, int] = {}
            for fold in range(1, args.n_splits + 1):
                fold_dir = os.path.join(root, f"fold_{fold}")
                # 1) 载入该折最优模型
                ckpt = torch.load(os.path.join(fold_dir, "best_model.pth"), map_location='cpu')
                model = build_model(in_dim, clin_dim, args).to(device)
                model.load_state_dict(ckpt["model"]); model.eval()

                # 2) 加载该折的 scaler
                with open(os.path.join(fold_dir, "scaler.json"), "r", encoding="utf-8") as f:
                    sdict = json.load(f)
                scaler = StandardScaler()
                scaler.mean_ = np.array(sdict["mean"], dtype=float)
                scale = np.array(sdict["scale"], dtype=float)
                scale[scale < 1e-8] = 1e-8
                scaler.scale_ = scale
                scaler.var_   = scaler.scale_ ** 2

                full_cols = sdict["full_clin_cols"]
                cont_cols = sdict["cont_cols"]
                cont_idx  = [full_cols.index(c) for c in cont_cols]
                log1p_flag = bool(sdict.get("log1p", True))
                clin_tf = make_np_transform_partial(scaler, cont_idx, log1p=log1p_flag)

                # 3) 构建测试集（用该折的 scaler）
                ds_te = Generic_MIL_Dataset_Clin(args.data_root_dir, args.clinical_csv, test_ids,
                                                 max_patches=None, label_col=args.label_col,
                                                 num_classes=args.num_classes,
                                                 clin_transform=clin_tf)
                dl_te = DataLoader(ds_te, batch_size=1, shuffle=False, num_workers=args.num_workers)

                # 4) 评估（拿到顺序对齐的 probs 与 slide_id）
                blob = evaluate(model, dl_te, device, args.num_classes)
                sids = blob["slide_ids"]
                P    = blob["probs"]
                ys   = blob["y_true"]
                # 5) 累加概率
                for sid, p, y in zip(sids, P, ys):
                    if sid not in agg_probs:
                        agg_probs[sid] = np.zeros_like(p, dtype=float)
                        y_true_map[sid] = int(y)
                    agg_probs[sid] += p

            # —— 集成完成：做最终预测与指标 —— #
            rows = []
            for sid in test_ids:
                if sid not in agg_probs:
                    continue
                p = agg_probs[sid] / float(args.n_splits)
                y = y_true_map[sid]
                yhat = int(np.argmax(p)) if args.num_classes >= 2 else int(p[1] >= 0.5)
                r = {"slide_id": sid, "y_true": y, "y_pred": yhat}
                for i, v in enumerate(p):
                    r[f"prob_c{i}"] = float(v)
                rows.append(r)
            test_df = pd.DataFrame(rows)
            test_df.to_csv(os.path.join(root, "test_preds.csv"), index=False)

            # 汇总测试指标
            if len(test_df):
                y_true = test_df["y_true"].to_numpy()
                y_pred = test_df["y_pred"].to_numpy()
                P = test_df[[c for c in test_df.columns if c.startswith("prob_c")]].to_numpy()

                test_metrics = {
                    "ACC":  float(accuracy_score(y_true, y_pred)),
                    "bACC": float(balanced_accuracy_score(y_true, y_pred)),
                    "F1":   float(f1_score(y_true, y_pred, average='macro'))
                }
                try:
                    if args.num_classes == 2:
                        test_metrics["AUC"] = float(roc_auc_score(y_true, P[:, 1]))
                    else:
                        test_metrics["AUC"] = float(roc_auc_score(y_true, P, multi_class='ovr', average='macro'))
                except Exception:
                    test_metrics["AUC"] = float('nan')

                with open(os.path.join(root, "test_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(test_metrics, f, ensure_ascii=False, indent=2)
                print("\n=== TEST (Ensemble over folds) ===")
                print(test_metrics)
            else:
                print("[Test] 无可评估样本，跳过。")

if __name__ == "__main__":
    main()
