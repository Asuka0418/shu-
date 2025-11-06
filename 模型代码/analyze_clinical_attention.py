#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临床特征注意力分析脚本
用于分析训练好的模型对各个临床特征的关注程度

使用方法:
python analyze_clinical_attention.py --result_dir /path/to/results --fold 1
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体 - 自动检测并使用可用字体
import matplotlib
import matplotlib.font_manager as fm

# 查找系统中可用的中文字体
def find_chinese_font():
    """查找系统中可用的中文字体"""
    # 优先级列表
    preferred_fonts = [
        'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Microsoft YaHei', 'SimHei', 'Arial Unicode MS'
    ]
    
    # 获取系统所有字体
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    # 找到第一个可用的中文字体
    for font in preferred_fonts:
        if font in available_fonts:
            print(f"[字体] 使用中文字体: {font}")
            return font
    
    # 如果都没有，尝试找任何包含CJK的字体
    for font_name in available_fonts:
        if 'CJK' in font_name or 'Chinese' in font_name:
            print(f"[字体] 使用中文字体: {font_name}")
            return font_name
    
    print("[警告] 未找到合适的中文字体，中文可能显示为方框")
    return 'DejaVu Sans'

# 配置字体
chinese_font = find_chinese_font()
matplotlib.rcParams['font.sans-serif'] = [chinese_font]
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'

def load_clinical_columns(csv_path):
    """
    加载并解析临床特征列名（经过one-hot编码后的）
    """
    df = pd.read_csv(csv_path)
    df.set_index('slide_id', inplace=True)
    
    # 复现数据集的one-hot编码过程
    CAT_COLS = ["部位", "肿瘤破裂", "肝脏转移", "腹腔播散", "坏死", "基因检测", "核异型性编码", "形态学评估编码"]
    df_encoded = pd.get_dummies(
        df,
        columns=[c for c in CAT_COLS if c in df.columns],
        prefix=[c for c in CAT_COLS if c in df.columns],
        dtype=float
    )
    
    # 排除标签列
    label_cols = ['NIH评估', 'label']
    clin_cols = [c for c in df_encoded.columns if c not in label_cols]
    
    return clin_cols, df_encoded

def analyze_fold_attention(fold_dir, clin_cols, top_k=10):
    """
    分析单个fold的临床特征注意力
    """
    clin_att_path = os.path.join(fold_dir, "clin_att_weights_val.npz")
    
    if not os.path.exists(clin_att_path):
        print(f"[警告] {clin_att_path} 不存在，跳过")
        return None
    
    # 加载注意力权重
    att_data = np.load(clin_att_path)
    slide_ids = list(att_data.keys())
    
    # 汇总所有样本的注意力权重
    all_atts = []
    for sid in slide_ids:
        att = att_data[sid]
        if att.shape[0] == len(clin_cols):
            all_atts.append(att)
    
    if len(all_atts) == 0:
        print(f"[警告] {fold_dir} 中没有有效的临床注意力数据")
        return None
    
    # 计算平均注意力和标准差
    atts_matrix = np.stack(all_atts)  # [N_samples, N_features]
    mean_att = atts_matrix.mean(axis=0)
    std_att = atts_matrix.std(axis=0)
    
    # 创建DataFrame - 保持clin_cols的原始顺序！
    att_df = pd.DataFrame({
        'feature': clin_cols,
        'mean_attention': mean_att,
        'std_attention': std_att
    })
    
    # 为了显示Top K，创建排序后的副本
    att_df_sorted = att_df.sort_values('mean_attention', ascending=False)
    
    print(f"\n{'='*60}")
    print(f"Top {top_k} 最重要的临床特征:")
    print(f"{'='*60}")
    for idx, row in att_df_sorted.head(top_k).iterrows():
        print(f"{row['feature']:30s} | 注意力: {row['mean_attention']:.4f} ± {row['std_attention']:.4f}")
    
    # 返回未排序的DataFrame，保持特征顺序一致！
    return att_df, atts_matrix

def visualize_attention(att_df, atts_matrix, output_dir, fold_name, top_k=15):
    """
    可视化临床特征注意力
    注意：att_df应该是未排序的，保持原始特征顺序
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 按注意力排序以获取Top K
    att_df_sorted = att_df.sort_values('mean_attention', ascending=False)
    
    # 1. 条形图：Top K 特征的平均注意力
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = att_df_sorted.head(top_k)
    
    bars = ax.barh(range(len(top_features)), top_features['mean_attention'].values, 
                   xerr=top_features['std_attention'].values, capsize=3)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('平均注意力权重', fontsize=12)
    ax.set_title(f'{fold_name} - Top {top_k} 临床特征注意力', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (val, std) in enumerate(zip(top_features['mean_attention'].values, 
                                       top_features['std_attention'].values)):
        ax.text(val + std + 0.001, i, f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fold_name}_top_features.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 热力图：所有特征的注意力分布（样本 x 特征）
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 只显示top_k特征（使用排序后的）
    top_indices = att_df_sorted.head(top_k).index.tolist()
    top_feature_names = [att_df_sorted.loc[i, 'feature'] for i in top_indices]
    
    # 获取对应的注意力矩阵列
    feature_to_idx = {feat: i for i, feat in enumerate(att_df['feature'].values)}
    top_col_indices = [feature_to_idx[feat] for feat in top_feature_names]
    
    heatmap_data = atts_matrix[:, top_col_indices].T  # [features, samples]
    
    sns.heatmap(heatmap_data, 
                yticklabels=top_feature_names,
                cmap='YlOrRd', 
                cbar_kws={'label': '注意力权重'},
                ax=ax)
    ax.set_xlabel('样本 ID', fontsize=12)
    ax.set_ylabel('临床特征', fontsize=12)
    ax.set_title(f'{fold_name} - Top {top_k} 特征在各样本上的注意力分布', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fold_name}_attention_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 箱线图：展示注意力分布
    fig, ax = plt.subplots(figsize=(14, 8))
    
    box_data = [atts_matrix[:, feature_to_idx[feat]] for feat in top_feature_names]
    bp = ax.boxplot(box_data, labels=top_feature_names, vert=True, patch_artist=True)
    
    # 美化箱线图
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_ylabel('注意力权重', fontsize=12)
    ax.set_title(f'{fold_name} - Top {top_k} 特征注意力分布（箱线图）', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fold_name}_attention_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[可视化] 已保存到 {output_dir}")

def analyze_all_folds(result_dir, clin_cols, n_splits=5, top_k=10):
    """
    分析所有fold的注意力并汇总
    """
    all_fold_dfs = []
    
    for fold in range(1, n_splits + 1):
        fold_dir = os.path.join(result_dir, f"fold_{fold}")
        if not os.path.exists(fold_dir):
            continue
        
        print(f"\n{'='*60}")
        print(f"分析 Fold {fold}")
        print(f"{'='*60}")
        
        result = analyze_fold_attention(fold_dir, clin_cols, top_k=top_k)
        if result is None:
            continue
        
        att_df, atts_matrix = result
        all_fold_dfs.append(att_df)
        
        # 可视化
        visualize_attention(att_df, atts_matrix, 
                           output_dir=os.path.join(fold_dir, "clinical_attention_viz"),
                           fold_name=f"Fold_{fold}",
                           top_k=top_k)
    
    if len(all_fold_dfs) == 0:
        print("[错误] 没有找到任何有效的临床注意力数据")
        return
    
    # 汇总所有fold的注意力
    print(f"\n{'='*80}")
    print("跨fold汇总分析")
    print(f"{'='*80}")
    
    # 计算各fold平均注意力的均值和标准差
    all_means = np.stack([df['mean_attention'].values for df in all_fold_dfs])
    grand_mean = all_means.mean(axis=0)
    grand_std = all_means.std(axis=0)
    
    summary_df = pd.DataFrame({
        'feature': clin_cols,
        'cross_fold_mean': grand_mean,
        'cross_fold_std': grand_std
    })
    summary_df = summary_df.sort_values('cross_fold_mean', ascending=False)
    
    # 保存汇总结果
    summary_path = os.path.join(result_dir, "clinical_attention_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n[保存] 汇总结果已保存到: {summary_path}")
    
    print(f"\n跨fold Top {top_k} 最重要的临床特征:")
    print(f"{'='*80}")
    for idx, row in summary_df.head(top_k).iterrows():
        print(f"{row['feature']:30s} | 注意力: {row['cross_fold_mean']:.4f} ± {row['cross_fold_std']:.4f}")
    
    # 可视化汇总结果
    fig, ax = plt.subplots(figsize=(12, 8))
    top_summary = summary_df.head(top_k)
    
    bars = ax.barh(range(len(top_summary)), top_summary['cross_fold_mean'].values,
                   xerr=top_summary['cross_fold_std'].values, capsize=3,
                   color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(top_summary)))
    ax.set_yticklabels(top_summary['feature'].values)
    ax.set_xlabel('平均注意力权重（跨fold）', fontsize=12)
    ax.set_title(f'跨{n_splits}折 - Top {top_k} 临床特征注意力汇总', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (val, std) in enumerate(zip(top_summary['cross_fold_mean'].values,
                                       top_summary['cross_fold_std'].values)):
        ax.text(val + std + 0.001, i, f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    summary_fig_path = os.path.join(result_dir, "clinical_attention_summary.png")
    plt.savefig(summary_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 汇总图已保存到: {summary_fig_path}")

def main():
    parser = argparse.ArgumentParser(description="分析临床特征注意力")
    parser.add_argument('--result_dir', type=str, required=True, 
                       help='训练结果目录（包含fold_1, fold_2等子目录）')
    parser.add_argument('--clinical_csv', type=str, required=True,
                       help='临床数据CSV文件路径')
    parser.add_argument('--fold', type=int, default=None,
                       help='指定分析某个fold（不指定则分析所有fold）')
    parser.add_argument('--top_k', type=int, default=15,
                       help='显示Top K个重要特征（默认15）')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='总fold数（默认5）')
    
    args = parser.parse_args()
    
    # 智能查找实际结果目录（处理带时间戳的子目录）
    result_dir = args.result_dir
    args_path = os.path.join(result_dir, "args.json")
    
    if not os.path.exists(args_path):
        # 尝试查找子目录
        subdirs = []
        if os.path.exists(result_dir):
            for item in os.listdir(result_dir):
                item_path = os.path.join(result_dir, item)
                if os.path.isdir(item_path):
                    args_in_subdir = os.path.join(item_path, "args.json")
                    if os.path.exists(args_in_subdir):
                        subdirs.append((os.path.getmtime(item_path), item_path))
        
        if subdirs:
            subdirs.sort(reverse=True)
            result_dir = subdirs[0][1]
            print(f"[自动检测] 找到结果目录: {os.path.basename(result_dir)}")
            args.result_dir = result_dir  # 更新result_dir
        else:
            print(f"[错误] 在 {result_dir} 及其子目录中都找不到 args.json")
            return
    
    # 加载临床特征列名
    print(f"[加载] 临床数据: {args.clinical_csv}")
    clin_cols, df_encoded = load_clinical_columns(args.clinical_csv)
    print(f"[信息] 共 {len(clin_cols)} 个临床特征（经one-hot编码后）")
    
    if args.fold is not None:
        # 分析单个fold
        fold_dir = os.path.join(args.result_dir, f"fold_{args.fold}")
        if not os.path.exists(fold_dir):
            print(f"[错误] {fold_dir} 不存在")
            return
        
        result = analyze_fold_attention(fold_dir, clin_cols, top_k=args.top_k)
        if result is not None:
            att_df, atts_matrix = result
            visualize_attention(att_df, atts_matrix,
                              output_dir=os.path.join(fold_dir, "clinical_attention_viz"),
                              fold_name=f"Fold_{args.fold}",
                              top_k=args.top_k)
    else:
        # 分析所有fold
        analyze_all_folds(args.result_dir, clin_cols, n_splits=args.n_splits, top_k=args.top_k)

if __name__ == "__main__":
    main()

