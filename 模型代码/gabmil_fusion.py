import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

# ====== ABMIL 注意力池化（Ilse et al.） ======
class AttentionPooling(nn.Module):
    def __init__(self, in_dim: int, att_dim: int, gated: bool = True):
        super().__init__()
        self.gated = gated
        self.att_a = nn.Linear(in_dim, att_dim)
        self.att_b = nn.Linear(in_dim, att_dim) if gated else None
        self.att_c = nn.Linear(att_dim, 1)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A = torch.tanh(self.att_a(H))
        if self.gated:
            B = torch.sigmoid(self.att_b(H))
            A = A * B
        A = self.att_c(A)
        A = torch.softmax(A, dim=0)
        M = torch.sum(A * H, dim=0)
        return M, A.squeeze(1)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

class ClinCrossAttentionPooling(nn.Module):
    """
    Cross-Attention 池化：临床向量作为 Query，patch 表示作为 Key/Value
    - 多头：更稳定；每个头产生一组权重 α^h，再对 H 加权求和得到 M^h，最后聚合为 M
    - 模态 dropout：训练时以 p 概率屏蔽临床对注意力的影响（q=0 => 近似均匀/图像主导）
    """
    def __init__(self,
                 in_dim: int,          # = embed_dim (来自你的 patch_embed 输出维)
                 clin_dim: int,        # ClinEncoder 输出维
                 att_dim: int,         # 每个头的 key/query 维度
                 num_heads: int = 4,
                 temperature: float = 1.0,
                 modality_dropout_p: float = 0.0,   # 训练时屏蔽临床 p
                 combine: str = "mean"              # "mean" or "concat_proj"
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.clin_dim = clin_dim
        self.att_dim = att_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.modality_dropout_p = modality_dropout_p
        self.combine = combine

        # 线性投影
        self.Wq = nn.Linear(clin_dim, att_dim * num_heads, bias=True)  # [B, clin_dim] -> [B, H*att_dim]
        self.Wk = nn.Linear(in_dim, att_dim * num_heads, bias=True)    # [N, in_dim]  -> [N, H*att_dim]
        # 可选：对 V 再做投影（这里直接用 H 作 V，稳妥且少参）
        if combine == "concat_proj":
            self.out = nn.Linear(in_dim * num_heads, in_dim)
        elif combine == "mean":
            self.out = nn.Identity()
        else:
            raise ValueError("combine must be 'mean' or 'concat_proj'")

    def forward(self, H: torch.Tensor, clin_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H:      [N, in_dim]   — bag 内 N 个 patch 的嵌入
        clin_z: [clin_dim] 或 [1, clin_dim]
        返回：
          M: [in_dim]    — 条件注意力后的袋级表示
          alpha: [N]     — 条件注意力权重（对 N 个实例）
        """
        if clin_z.dim() == 1:
            clin_z = clin_z.unsqueeze(0)            # [1, clin_dim]

        # --- 模态 dropout：训练阶段以 p 概率屏蔽临床对注意力的影响 ---
        cz = clin_z
        if self.training and self.modality_dropout_p > 0.0:
            if torch.rand(1).item() < self.modality_dropout_p:
                cz = torch.zeros_like(clin_z)       # q≈0 → 分数趋同（更看图像）

        # 计算 Q, K
        # Q: [1, H*att_dim] -> [H, att_dim]
        # K: [N, H*att_dim] -> [N, H, att_dim]
        Q = self.Wq(cz).view(1, self.num_heads, self.att_dim).squeeze(0)   # [H, att_dim]
        K = self.Wk(H).view(H.size(0), self.num_heads, self.att_dim)       # [N, H, att_dim]

        # 打分：s_i^h = <q^h, k_i^h>/sqrt(d)
        # s: [N, H]
        s = torch.einsum('hd,nhd->nh', Q, K) / (self.att_dim ** 0.5)
        s = s / self.temperature
        s = s - s.max(dim=0, keepdim=True).values                           # 稳定（按头）

        # 注意力：对实例维 N softmax，得到 α: [N, H]
        alpha = torch.softmax(s, dim=0)

        # 对 H 做按头加权：每个头一组权重
        # 先扩展 H -> [N, H, D]，逐头加权求和 -> [H, D]
        # ClinCrossAttentionPooling.forward 中，替换加权汇聚部分：
        # alpha: [N,H], H: [N,D]
        M_heads = alpha.t().matmul(H)  # [H, D]  ← 避免 H_expand 和逐元素乘

        if self.combine == "mean":
            M = M_heads.mean(dim=0)
        else:
            M = self.out(M_heads.flatten(0,1))
        alpha_mean = alpha.mean(dim=1)
        return M, alpha_mean

# ====== SIMM 适配器（无坐标时自动跳过） ======
class SIMMAdapter(nn.Module):
    """
    将官方 SIMM 作为可选子模块挂载：
    - use_simm=False 或 xy is None -> 直通
    - use_simm=True 且传入 xy -> 调用 SIMM 模块
    这样就满足你“先无坐标，后启用空间”的流程。
    """
    def __init__(self, in_dim: int, use_simm: bool = False, simm_module: Optional[nn.Module] = None):
        super().__init__()
        self.use_simm = use_simm
        self.simm = simm_module  # 未来：from models.simm import SIMM(...)
        self.in_dim = in_dim

    def forward(self, H: torch.Tensor, xy: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.use_simm or (xy is None):
            return H
        if self.simm is None:
            # 兼容占位：未注入官方 SIMM 时，保持直通
            return H
        # 官方 SIMM 期望的调用接口（请按仓库实际实现调整）
        return self.simm(H, xy)

# ====== 临床特征编码器 ======
class ClinEncoder(nn.Module):
    """
    - 可选缺失 mask：miss_mask [B, clin_in_dim]，True=有值, False=缺失
    - 缺失处替换为可学习 miss_token（每列一个）
    """
    def __init__(self, in_dim: int, hidden: int = 128, act: str = "gelu"):
        super().__init__()
        Act = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        self.miss_token = nn.Parameter(torch.zeros(in_dim))
        self.fc = nn.Linear(in_dim, hidden)
        self.act = Act()
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(0.15)

    def forward(self, x: torch.Tensor, miss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [in_dim] 或 [B, in_dim]
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)
        if miss_mask is not None:
            x = torch.where(miss_mask, x, self.miss_token)  # 缺失→learnable token
        h = self.fc(x)
        h = self.act(h)
        h = self.norm(h)
        h = self.drop(h)
        return h.squeeze(0) if single else h
class ClinFeatureAttention(nn.Module):
    """
    对临床向量 x ∈ R^D 做“维度级注意力池化”：
      - 为每个维度 i 学一个嵌入 e_i ∈ R^{token_dim}
      - 令该维 token t_i = x_i * e_i
      - 用 ABMIL 风格（可 gated）在 D 个 token 上做注意力，得到临床表征 C ∈ R^{out_dim}
    """
    def __init__(self, clin_dim: int, token_dim: int, att_dim: int, out_dim: int,
                 gated: bool = True, dropout: float = 0.1):
        super().__init__()
        self.clin_dim = clin_dim
        self.emb = nn.Parameter(torch.empty(clin_dim, token_dim))
        nn.init.xavier_uniform_(self.emb)

        self.gated = gated
        self.att_a = nn.Linear(token_dim, att_dim, bias=True)
        self.att_b = nn.Linear(token_dim, att_dim, bias=True) if gated else None
        self.att_c = nn.Linear(att_dim, 1, bias=True)

        self.proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Dropout(dropout),
            nn.Linear(token_dim, out_dim)
        )

    def forward(self, x: torch.Tensor, miss_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [D] 或 [B, D] —— 临床向量（含 one-hot 后的所有列）
        miss_mask: 同维度的布尔掩码（True=有值, False=缺失），可选
        返回:
          C: [out_dim] 或 [B, out_dim] —— 注意力后的临床表征
          alpha_feat: [D] 或 [B, D] —— 各临床维度的注意力权重（便于可视化/正则）
        """
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)  # [1, D]

        if miss_mask is not None:
            x = torch.where(miss_mask, x, torch.zeros_like(x))

        # 生成特征 token：T[b, i, :] = x[b, i] * emb[i]
        T = x.unsqueeze(-1) * self.emb.unsqueeze(0)  # [B, D, token_dim]

        A = torch.tanh(self.att_a(T))  # [B, D, att_dim]
        if self.gated:
            A = A * torch.sigmoid(self.att_b(T))
        A = self.att_c(A).squeeze(-1)  # [B, D]
        alpha = torch.softmax(A, dim=1)  # 沿“维度”归一化

        # 注意力加权求和 -> 临床表征
        M = torch.sum(alpha.unsqueeze(-1) * T, dim=1)  # [B, token_dim]
        C = self.proj(M)  # [B, out_dim]

        if single:
            return C.squeeze(0), alpha.squeeze(0)
        return C, alpha

# ====== 主模型：GABMILFusion ======
class GABMILFusion(nn.Module):
    def __init__(self,
                 in_dim: int,           # patch 特征维度（UNI 输出维）
                 embed_dim: int,        # 投影维度
                 att_dim: int,          # 注意力内部维度
                 clin_dim: int,         # 临床特征维度
                 num_classes: int = 4,
                 gated_att: bool = True,
                 use_simm: bool = False,   # 未来切换：True 时启用 SIMM
                 simm_module: Optional[nn.Module] = None,  # 注入官方 SIMM 实例
                 fusion_hidden: int = 256):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.img_norm  = nn.LayerNorm(embed_dim)
        self.clin_norm = nn.LayerNorm(fusion_hidden // 2)
        self.img_gamma  = nn.Parameter(torch.tensor(1.0))
        self.clin_gamma = nn.Parameter(torch.tensor(1.0))
        self.spatial = SIMMAdapter(embed_dim, use_simm=use_simm, simm_module=simm_module)
        self.clin_att = ClinFeatureAttention(
            clin_dim=clin_dim, token_dim=128, att_dim=128,
            out_dim=fusion_hidden // 2, gated=True, dropout=0.15
        )
        self.pool = ClinCrossAttentionPooling(
            in_dim=embed_dim,
            clin_dim=fusion_hidden // 2,
            att_dim=128,
            num_heads=4,
            temperature=0.8,
            modality_dropout_p=0.1,
            combine="concat_proj"  # 或 "concat_proj"
            )
        self.pool_img = AttentionPooling(in_dim=embed_dim, att_dim=128, gated=gated_att)
        self.bag_norm = nn.LayerNorm(embed_dim)
        self.norm = nn.LayerNorm(fusion_hidden // 2)
        self.clin = ClinEncoder(clin_dim, hidden=fusion_hidden // 2)
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim + fusion_hidden // 2, fusion_hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(fusion_hidden, num_classes)
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: 
                    nn.init.zeros_(m.bias)
    def forward(self, feat: torch.Tensor, xy: Optional[torch.Tensor], clin: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        H = self.patch_embed(feat)
        H = self.spatial(H, xy)                # 无坐标/未启用 -> 直通
        H = self.img_norm(H) * self.img_gamma
        C, alpha_feat = self.clin_att(clin, miss_mask=None)
        #C = self.clin_norm(C) * self.clin_gamma
        M, A = self.pool(H, C)
        M = self.bag_norm(M)
        Z = torch.cat([M, C], dim=-1)
        logits = self.fuse(Z.unsqueeze(0)).squeeze(0)
        return logits, {"att": A, "bag": M, "clin": C, "clin_att": alpha_feat}
