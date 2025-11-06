#!/bin/bash
# ==========================================================
# GABMIL Attention Visualization 批量热力图生成脚本
#  - 支持按fold批量生成所有slide_id热力图
#  - 输出：原图 + 注意力热力图 + Top-K patch 拼图
# ==========================================================

# === 用户配置区 ===
CSV_PATH="/data/wl2p/pjh/results/2025data/fuse1/2025_pt_files_NIH评估_C4_K5_20251030-132647/oof_predictions.csv"
WSI_DIR="/data/Dataset_all/GIST多模态数据/2025"
FEATURE_DIR="/data/wl2p/pjh/2025_pt_files"
ATT_BASE="/data/wl2p/pjh/results/2025data/fuse1/2025_pt_files_NIH评估_C4_K5_20251030-132647"
OUT_ROOT="./heatmaps_25data_fuse_1_0.7"

# === 可调参数 ===
PATCH_SIZE=512
SCALE=32
ZOOM_TOPK=5
ZOOM_SIZE=512
PCLIP=98          # 提高到98，更保守的裁剪
GAMMA=0.42        # 提高到0.45，稍微压暗以突出最高值
MASK_THRESH=220
BINS=5            # 只保留2档（高/极高），简化视觉
MIN_ATT=0.7
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

