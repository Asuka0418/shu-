import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

class ClinicalEncoder(nn.Module):
    def __init__(self, num_numerical, num_categorical, embed_dim=128):
        super().__init__()
        # 数值特征处理：标准化后通过线性层
        self.scaler = StandardScaler()
        self.numerical_encoder = nn.Linear(num_numerical, embed_dim//2)
        
        # 类别特征处理：每个类别通过嵌入层
        self.cat_embedders = nn.ModuleList()
        for num_classes in num_categorical:
            self.cat_embedders.append(nn.Embedding(num_classes, embed_dim//(2*len(num_categorical))))
        
        # 融合层
        self.fusion = nn.Linear(embed_dim//2 + sum(embed_dim//(2*len(num_categorical)) for _ in num_categorical), embed_dim)
        self.relu = nn.ReLU()

    def fit(self, df, numerical_cols, categorical_cols):
        # 拟合标准化器和编码器（仅训练时调用）
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scaler.fit(df[numerical_cols])
        # 记录类别特征的映射关系（如部位：肺→0，肝→1...）
        self.cat_maps = {}
        for col in categorical_cols:
            self.cat_maps[col] = {v: i for i, v in enumerate(df[col].unique())}

    def forward(self, df):
        # 处理数值特征
        numerical = self.scaler.transform(df[self.numerical_cols])
        numerical = torch.tensor(numerical, dtype=torch.float32)
        num_embed = self.relu(self.numerical_encoder(numerical))
        
        # 处理类别特征
        cat_embeds = []
        for i, col in enumerate(self.categorical_cols):
            # 将类别映射为整数
            cat_ids = torch.tensor([self.cat_maps[col][v] for v in df[col]], dtype=torch.long)
            cat_embed = self.cat_embedders[i](cat_ids)
            cat_embeds.append(cat_embed)
        cat_embed = torch.cat(cat_embeds, dim=1)
        
        # 融合特征
        clinical_embed = self.relu(self.fusion(torch.cat([num_embed, cat_embed], dim=1)))
        return clinical_embed