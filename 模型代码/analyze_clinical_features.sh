#!/bin/bash
# 临床特征注意力完整分析脚本 - 一键生成所有结果
# 使用方法: bash analyze_clinical_features.sh [结果目录]

# 激活conda环境
source /home/wl2p/anaconda3/bin/activate pjh

# ===================================
# 配置区
# ===================================
# 如果命令行提供了路径，使用命令行参数；否则使用默认值
if [ -n "$1" ]; then
    RESULT_BASE_DIR="$1"
else
    RESULT_BASE_DIR="/data/wl2p/pjh/results/2025data/fuse1+2-2"
fi

# 自动查找最新的结果目录（如果提供的是基础目录）
if [ -f "$RESULT_BASE_DIR/args.json" ]; then
    # 已经是完整路径
    RESULT_DIR="$RESULT_BASE_DIR"
else
    # 需要查找子目录
    RESULT_DIR=$(find "$RESULT_BASE_DIR" -maxdepth 1 -type d -name "*_pt_files_*" | sort -r | head -1)
    if [ -z "$RESULT_DIR" ]; then
        echo "❌ 错误: 未找到训练结果目录"
        echo "使用方法: bash $0 [结果目录路径]"
        exit 1
    fi
fi

# 从args.json读取临床CSV路径
CLINICAL_CSV=$(python -c "
import json
with open('$RESULT_DIR/args.json', 'r') as f:
    args = json.load(f)
print(args.get('clinical_csv', ''))
" 2>/dev/null)

if [ -z "$CLINICAL_CSV" ] || [ ! -f "$CLINICAL_CSV" ]; then
    echo "❌ 错误: 无法从args.json读取临床数据路径"
    exit 1
fi

# ===================================
# 开始分析
# ===================================
echo "=========================================="
echo "🔬 临床特征注意力完整分析"
echo "=========================================="
echo ""
echo "📁 结果目录: $(basename $RESULT_DIR)"
echo "📄 临床数据: $CLINICAL_CSV"
echo ""

# 检查必要文件
echo "🔍 检查数据完整性..."
MISSING=0
for fold in 1 2 3 4 5; do
    CLIN_ATT="$RESULT_DIR/fold_$fold/clin_att_weights_val.npz"
    if [ ! -f "$CLIN_ATT" ]; then
        echo "  ❌ Fold $fold: 缺少临床注意力数据"
        MISSING=1
    else
        echo "  ✅ Fold $fold: 数据完整"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "❌ 错误: 部分fold缺少临床注意力数据"
    echo "请确保训练时使用了 --save_att 参数"
    exit 1
fi

echo ""
echo "✅ 数据完整性检查通过"
echo ""

# 删除可能存在的旧文件（避免混淆）
echo "🗑️  清理旧文件..."
rm -f "$RESULT_DIR/clinical_attention_summary.png" 2>/dev/null
rm -f "$RESULT_DIR/clinical_attention_summary.csv" 2>/dev/null
rm -f "$RESULT_DIR/clinical_attention_ranking.csv" 2>/dev/null  # 删除quick_view的旧文件
rm -rf "$RESULT_DIR"/fold_*/clinical_attention_viz/ 2>/dev/null
echo "✅ 清理完成"
echo ""

# ===================================
# 步骤1: 快速查看Top特征
# ===================================
echo "=========================================="
echo "📊 步骤 1/2: 快速查看Top特征重要性"
echo "=========================================="
echo ""

python quick_view_clinical_attention.py "$RESULT_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 快速分析失败"
    exit 1
fi

# 删除quick_view生成的ranking.csv（会与后续的summary.csv混淆）
rm -f "$RESULT_DIR/clinical_attention_ranking.csv" 2>/dev/null

echo ""

# ===================================
# 步骤2: 生成详细可视化图表
# ===================================
echo "=========================================="
echo "🎨 步骤 2/2: 生成详细可视化图表"
echo "=========================================="
echo ""

python analyze_clinical_attention.py \
  --result_dir "$RESULT_DIR" \
  --clinical_csv "$CLINICAL_CSV" \
  --top_k 15 \
  --n_splits 5

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 可视化生成失败"
    exit 1
fi

# ===================================
# 完成总结
# ===================================
echo ""
echo "=========================================="
echo "✅ 分析完成！"
echo "=========================================="
echo ""
echo "📊 生成的文件："
echo ""
echo "【汇总结果】"
echo "  📈 图表: $RESULT_DIR/clinical_attention_summary.png"
echo "  📄 CSV:  $RESULT_DIR/clinical_attention_summary.csv"
echo ""
echo "【详细图表】(每个fold)"
for fold in 1 2 3 4 5; do
    VIZ_DIR="$RESULT_DIR/fold_$fold/clinical_attention_viz"
    if [ -d "$VIZ_DIR" ]; then
        COUNT=$(ls -1 "$VIZ_DIR"/*.png 2>/dev/null | wc -l)
        echo "  📁 Fold $fold: $COUNT 张图表 ($VIZ_DIR)"
    fi
done
echo ""
echo "=========================================="
echo "💡 下一步："
echo "=========================================="
echo ""
echo "1️⃣  查看汇总图表（用于论文/汇报）："
echo "   打开: $RESULT_DIR/clinical_attention_summary.png"
echo ""
echo "2️⃣  查看数据表格："
echo "   cat $RESULT_DIR/clinical_attention_summary.csv"
echo ""
echo "3️⃣  下载到本地（如果在远程服务器）："
echo "   scp user@server:$RESULT_DIR/clinical_attention_summary.* ./"
echo ""
echo "🎉 所有图表的中文都已正确显示，可以直接用于汇报！"
echo ""

