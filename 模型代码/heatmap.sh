#!/bin/bash
# ==========================================================
# GABMIL Attention Visualization 批量热力图生成脚本
#  - 支持按fold批量生成所有slide_id热力图
#  - 输出：原图 + 注意力热力图 + Top-K patch 拼图
# ==========================================================

# === 用户配置区 ===
CSV_PATH="..........."
WSI_DIR="..........."
FEATURE_DIR="..........."
ATT_BASE="..........."
OUT_ROOT="..........."

# === 可调参数 ===
PATCH_SIZE=1
SCALE=1
ZOOM_TOPK=1
ZOOM_SIZE=1
PCLIP=1          # 提高到98，更保守的裁剪
GAMMA=1       # 提高到0.45，稍微压暗以突出最高值
MASK_THRESH=1
BINS=1            # 只保留2档（高/极高），简化视觉
MIN_ATT=1
# ==========================================================
# 循环多个fold（若只需单个可保留一项）
# ==========================================================
for FOLD in 1 2 3 4 5; do
  ATT_PATH="${ATT_BASE}/fold_${FOLD}/att_weights_val.npz"
  OUT_DIR="${OUT_ROOT}/fold_${FOLD}"

  echo "============================================"
  echo ">>> 开始处理 FOLD=${FOLD}"
  echo "    CSV: ${CSV_PATH}"
  echo "    ATT: ${ATT_PATH}"
  echo "    OUT: ${OUT_DIR}"
  echo "============================================"

  python heatmap.py \
    --csv_path "${CSV_PATH}" \
    --fold ${FOLD} \
    --wsi_dir "${WSI_DIR}" \
    --feature_dir "${FEATURE_DIR}" \
    --att "${ATT_PATH}" \
    --out "${OUT_DIR}" \
    --scale ${SCALE} \
    --patch_size ${PATCH_SIZE} \
    --zoom_topk ${ZOOM_TOPK} \
    --zoom_size ${ZOOM_SIZE} \
    --pclip ${PCLIP} \
    --gamma ${GAMMA} \
    --mask_thresh ${MASK_THRESH} \
    --bins ${BINS} \
    --min_att ${MIN_ATT}

  echo ">>> Fold ${FOLD} 处理完成 ✅"
  echo
done

echo "🎯 所有fold已完成热力图生成！"

