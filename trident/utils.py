# TRIDENT/trident/utils.py
import torch
import pandas as pd

def encode_clinical_features(clinical_data):
    """
    将原始临床数据（包含分类特征和数值特征）转换为嵌入向量
    
    参数:
        clinical_data (dict): 包含临床特征的字典，键为特征名（需与CSV列名一致），值为特征值
                             示例: {'年龄': 1, '大小': 7.0, '部位': 1, '基因检测': 1, ...}
    
    返回:
        torch.Tensor: 编码后的特征向量；若输入为None则返回None
    """
    if clinical_data is None:
        return None

    features = []
    
    # 1. 处理数值特征（仅保留连续值特征“大小”，根据CSV数据调整）
    numeric_features = ['大小']  # “大小”为连续数值，单位可能是cm等
    for feat in numeric_features:
        # 缺失值填充为0.0，实际场景建议用均值/中位数填充
        features.append(clinical_data.get(feat, 0.0))

    # 2. 处理分类特征（根据CSV中实际取值定义类别，采用独热编码）
    categorical_features = {
        # 二元分类特征（0/1）
        '年龄': [0, 1],               # 推测0/1代表某种二元属性（如年龄分组）
        '肿瘤破裂': [0, 1],           # 0=未破裂，1=破裂
        '肝脏转移': [0, 1],           # 0=无，1=有
        '腹腔播散': [0, 1],           # 0=无，1=有
        '坏死': [0, 1],               # 0=无坏死，1=有坏死
        
        # 多分类特征
        '部位': [1, 2, 3],            # 1/2/3分别代表不同部位（需结合业务确认具体含义）
        '基因检测': [0, 1, 2, 3],     # 0/1/2/3代表不同基因检测结果
        '核异型性': [0, 1, 2, 3, 4, 5],# 核异型性分级（0-5级）
        'NIH评估': [1, 2, 3, 4]       # NIH评估等级（1-4级）
    }
    
    for feat, categories in categorical_features.items():
        val = clinical_data.get(feat, -1)  # 缺失值用-1表示（不匹配任何类别）
        # 生成独热向量，匹配的类别位置为1.0，其余为0.0
        one_hot = [1.0 if c == val else 0.0 for c in categories]
        features.extend(one_hot)

    # 转换为PyTorch张量，便于后续与影像特征融合
    return torch.tensor(features, dtype=torch.float32)