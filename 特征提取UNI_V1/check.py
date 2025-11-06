import h5py

def check_fused_h5(fused_file_path):
    with h5py.File(fused_file_path, "r") as f:
        print(f"文件中的键：{list(f.keys())}")  # 应该包含'fused_features'
        if "fused_features" in f:
            fused_feats = f["fused_features"][:]
            print(f"融合特征的形状：{fused_feats.shape}")
            print(f"解释：共{fused_feats.shape[0]}个patch，每个patch的特征维度为{fused_feats.shape[1]}（WSI维度 + CSV维度）")

# 替换为你的融合后文件路径
check_fused_h5(".........")
