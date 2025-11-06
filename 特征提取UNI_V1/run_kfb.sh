export PYTHONPATH="........."
export LD_LIBRARY_PATH="..........."


#!/bin/bash
# 按顺序执行：分割(seg) → 坐标提取(coords) → 彩色图像生成(img) → 嵌入向量提取(feat)

# ============== 核心参数配置 ==============
WSI_DIR="......"  # .kfb 文件所在目录（含子目录）
JOB_DIR=".........."                # 结果输出目录（避免覆盖，建议用新目录）
GPU_INDEX=0                                # 使用的 GPU 编号（0、1、2 等）
READER_TYPE="aslide"                          # 读取器类型（必须用 aslide 适配 .kfb）
WSI_EXT=".kfb"                                # WSI 文件扩展名（仅处理 .kfb）
PATCH_ENCODER="uni_v1"                      # 特征提取编码器（resnet50 等可选）
MAG=20                                        # 特征提取倍率（20×）
PATCH_SIZE=512                                # Patch 大小（像素）
# ==========================================
CKPT="........."
# 确保输出目录存在
mkdir -p "${JOB_DIR}"

# 步骤1：生成分割掩码（seg 任务）
echo "===== 1/4 执行 seg 任务（生成分割掩码） ====="
if ! python run_kfb.py \
  --task seg \
  --gpu "${GPU_INDEX}" \
  --wsi_dir "${WSI_DIR}" \
  --job_dir "${JOB_DIR}" \
  --reader_type "${READER_TYPE}" \
  --wsi_ext "${WSI_EXT}" \
  --seg_batch_size 64 \
  --search_nested  \
  --patch_encoder_ckpt_path "${CKPT}"; then
  echo "===== seg 任务失败，终止流程 ====="
  exit 1
fi

# 步骤2：生成组织坐标（coords 任务）→ 修复错误提示为“coords 失败”
echo "===== 2/4 执行 coords 任务（生成坐标文件） ====="
if ! python run_kfb.py \
  --task coords \
  --gpu "${GPU_INDEX}" \
  --wsi_dir "${WSI_DIR}" \
  --job_dir "${JOB_DIR}" \
  --reader_type "${READER_TYPE}" \
  --wsi_ext "${WSI_EXT}" \
  --mag "${MAG}" \
  --patch_size "${PATCH_SIZE}" \
  --search_nested \
  --overlap 0  \
  --patch_encoder_ckpt_path "${CKPT}"; then
  echo "===== coords 任务失败，终止流程 ====="
  exit 1
fi

# 步骤3：生成彩色图像（img 任务）→ 修复错误提示为“img 失败”
echo "===== 3/4 执行 img 任务（生成彩色图像） ====="
if ! python run_kfb.py \
  --task img \
  --gpu "${GPU_INDEX}" \
  --wsi_dir "${WSI_DIR}" \
  --job_dir "${JOB_DIR}" \
  --reader_type "${READER_TYPE}" \
  --search_nested \
  --wsi_ext "${WSI_EXT}" \
  --patch_encoder_ckpt_path "${CKPT}"; then
  echo "===== img 任务失败，终止流程 ====="
  exit 1
fi

# 步骤4：提取嵌入向量（feat 任务）
echo "===== 4/4 执行 feat 任务（提取嵌入向量） ====="
if ! python run_kfb.py \
  --task feat \
  --gpu "${GPU_INDEX}" \
  --wsi_dir "${WSI_DIR}" \
  --job_dir "${JOB_DIR}" \
  --reader_type "${READER_TYPE}" \
  --wsi_ext "${WSI_EXT}" \
  --patch_encoder "${PATCH_ENCODER}" \
  --mag "${MAG}" \
  --search_nested \
  --patch_size "${PATCH_SIZE}" \
  --feat_batch_size 64  \
  --patch_encoder_ckpt_path "${CKPT}"; then
  echo "===== feat 任务失败，终止流程 ====="
  exit 1
fi

echo "===== 所有任务完成！结果路径： ====="
echo "1. 分割掩码：${JOB_DIR}/segmentations"
echo "2. 坐标文件：${JOB_DIR}/${MAG}x_${PATCH_SIZE}px_0px_overlap"
echo "3. 彩色图像：${JOB_DIR}/visualization"
echo "4. 嵌入向量：${JOB_DIR}/features"
