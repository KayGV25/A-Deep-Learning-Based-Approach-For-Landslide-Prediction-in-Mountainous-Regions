import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from torchvision import models
from torch.optim.swa_utils import AveragedModel, update_bn
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import kornia as K
from copy import deepcopy
from tqdm import tqdm
import math
from collections import Counter
import os

IMG_SIZE = 224
BATCH_SIZE = 96 
NUM_WORKERS = 6
NUM_CLASSES = 3
EPOCHS = 60
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DATASET_DIR = "/home/islab/data/khuong/final_dataset_dat" # change if needed
# DATASET_DIR = "/home/islab/data/khuong/final_dataset_gray_dat" # change if needed
# DATASET_DIR = "/home/islab/data/khuong/bijie_dat" # change if needed
# DATASET_DIR = "/home/islab/data/khuong/bijie_binary_dat" # change if needed
MODEL_DIR = "/home/islab/data/khuong/models"
# MODEL_DIR = "/home/islab/data/khuong/models/gray"
# MODEL_DIR = "/home/islab/data/khuong/models/bijie"
# MODEL_DIR = "/home/islab/data/khuong/models/bijie_binary"
# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.synchronize()

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        B, C, H, W = x.shape

        # Channel attention
        avg = F.adaptive_avg_pool2d(x, 1).view(B, C)
        max_ = F.adaptive_max_pool2d(x, 1).view(B, C)

        attn = self.mlp(avg) + self.mlp(max_)
        attn = torch.sigmoid(attn).view(B, C, 1, 1)

        x = x * attn

        # Spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)

        spatial = torch.cat([avg, max_], dim=1)
        spatial = torch.sigmoid(self.spatial(spatial))

        return x * spatial
    
class EdgeExtractor(nn.Module):
    def forward(self, x):
        gray = K.color.rgb_to_grayscale(x)
        sobel = K.filters.sobel(gray)
        lap = K.filters.laplacian(gray, kernel_size=3)
        edge = torch.cat([sobel, lap], dim=1)
        return edge.mean(dim=1, keepdim=True)

class SCAM3DPlus(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()

        mid = in_ch // reduction

        self.reduce = nn.Conv2d(in_ch, mid, 1, bias=False)

        self.conv1 = nn.Conv2d(mid, mid, 1, bias=False)
        self.conv3 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.conv7 = nn.Conv2d(mid, mid, 7, padding=3, bias=False)

        self.expand = nn.Conv2d(mid * 3, in_ch, 1, bias=False)

        self.norm = nn.BatchNorm2d(in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # ===== AVG + MAX =====
        Fc_avg = F.adaptive_avg_pool2d(x, 1)
        Fc_max = F.adaptive_max_pool2d(x, 1)

        Fs_avg = torch.mean(x, dim=1, keepdim=True)
        Fs_max, _ = torch.max(x, dim=1, keepdim=True)

        # Expand
        Fc_avg = Fc_avg.expand(-1, -1, H, W)
        Fc_max = Fc_max.expand(-1, -1, H, W)

        Fs_avg = Fs_avg.expand(-1, C, -1, -1)
        Fs_max = Fs_max.expand(-1, C, -1, -1)

        # Combine
        Fsc = (Fc_avg * Fs_avg) + (Fc_max * Fs_max)

        # Conv block
        x_red = self.reduce(Fsc)

        x1 = self.conv1(x_red)
        x3 = self.conv3(x_red)
        x7 = self.conv7(x_red)

        x_cat = torch.cat([x1, x3, x7], dim=1)
        x_out = self.expand(x_cat)

        attn = self.sigmoid(self.norm(x_out))

        # ✅ residual attention (VERY important)
        return x + x * attn

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.gamma1 = nn.Parameter(torch.ones(1) * 0.5)  # 🔥 stabilize
        self.gamma2 = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        # ---- Attention ----
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        out, _ = self.attn(q, k, v)

        # 🔥 clamp to avoid explosion
        out = torch.clamp(out, -10, 10)

        x = x + self.gamma1 * out

        # ---- FFN ----
        x = x + self.gamma2 * self.ffn(self.norm2(x))

        return x

class ModelGating(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, tokens):
        # tokens: (B, 3, D)

        weights = [self.gate(t) for t in tokens.unbind(dim=1)]
        weights = torch.stack(weights, dim=1)  # (B, 3, 1)

        weights = torch.softmax(weights, dim=1)

        return tokens * weights

class StackedEnsemble(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Backbone models
        self.effnet = build_model("efficientnet_b3")
        self.mobilenet = build_model("mobilenet")
        self.resnet = build_model("resnet18")
        
        self.effnet.load_state_dict(torch.load(f"{MODEL_DIR}/efficientnet_b3_best.pth"))
        self.mobilenet.load_state_dict(torch.load(f"{MODEL_DIR}/mobilenet_best.pth"))
        self.resnet.load_state_dict(torch.load(f"{MODEL_DIR}/resnet18_best.pth"))
        
        # Freeze backbone parameters
        backbones = [self.effnet, self.mobilenet]

        backbones.append(self.resnet)
        for m in backbones:
            for p in m.parameters():
                p.requires_grad = False

        # Meta classifier (9 → hidden → 3)
        input_dim = num_classes * 3

        self.meta = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )


    def forward(self, x):

        # Backbone predictions
        p1 = torch.softmax(self.effnet(x), dim=1)
        p3 = torch.softmax(self.mobilenet(x), dim=1)
        p2 = torch.softmax(self.resnet(x), dim=1)

        # Concatenate → 9 feature vector
        stacked = torch.cat([p1, p2, p3], dim=1)

        # Meta classifier
        out = self.meta(stacked)

        return out
    
class WeightedEnsemble(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.num_classes = num_classes
        self.eps = 1e-6

        embed_dim = 96   # 🔥 increased capacity

        # ===== Backbones =====
        self.effnet = build_model("efficientnet_b3")
        self.mobilenet = build_model("mobilenet")

        self.resnet = build_model("resnet18")

        # ===== Load pretrained =====
        self.effnet.load_state_dict(torch.load(f"{MODEL_DIR}/efficientnet_b3_best.pth"))
        self.mobilenet.load_state_dict(torch.load(f"{MODEL_DIR}/mobilenet_best.pth"))

        self.resnet.load_state_dict(torch.load(f"{MODEL_DIR}/resnet18_best.pth"))

        # ===== Freeze =====
        for m in [self.effnet, self.mobilenet, self.resnet]:
            for p in m.parameters():
                p.requires_grad = False

        # ===============================
        # 🔥 PER-MODEL TEMPERATURE (big gain)
        # ===============================
        self.temperature = nn.Parameter(torch.ones(3) * 1.5)

        # ===============================
        # 🔥 CROSS ATTENTION (stronger)
        # ===============================
        self.token_proj = nn.Linear(num_classes, embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # 🔥 learned token importance (better than mean)
        self.token_weight = nn.Linear(embed_dim, 1)

        self.out_proj = nn.Linear(embed_dim, num_classes)

        # ===============================
        # 🔥 GLOBAL PRIOR
        # ===============================
        self.logit_weights = nn.Parameter(torch.ones(3))

        # 🔥 learnable fusion ratio
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # ===============================
        # 🔥 STRONGER HEAD
        # ===============================
        self.head = nn.Sequential(
            nn.LayerNorm(num_classes),
            nn.Linear(num_classes, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def safe_logits(self, x):
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-6)
        return torch.tanh(x) * 5.0

    def forward(self, x):

        # ===== logits =====
        l1 = self.safe_logits(self.effnet(x))
        l3 = self.safe_logits(self.mobilenet(x))
        l2 = self.safe_logits(self.resnet(x))

        # ===== per-model temperature =====
        T = torch.clamp(self.temperature, 0.5, 3.0)
        l1, l2, l3 = l1 / T[0], l2 / T[1], l3 / T[2]

        # ===== Dirichlet =====
        def evidence(logits):
            return F.softplus(logits)

        a1, a2, a3 = evidence(l1)+1, evidence(l2)+1, evidence(l3)+1

        p1 = a1 / (a1.sum(dim=1, keepdim=True) + self.eps)
        p2 = a2 / (a2.sum(dim=1, keepdim=True) + self.eps)
        p3 = a3 / (a3.sum(dim=1, keepdim=True) + self.eps)

        # ===============================
        # 🔥 CROSS ATTENTION (IMPROVED)
        # ===============================
        tokens = torch.stack([p1, p2, p3], dim=1)   # (B,3,C)
        tokens = self.token_proj(tokens)            # (B,3,D)

        attn_out, _ = self.attn(tokens, tokens, tokens)

        # 🔥 learned pooling instead of mean
        weights = torch.softmax(self.token_weight(attn_out), dim=1)  # (B,3,1)
        fused_attn = (weights * attn_out).sum(dim=1)  # (B,D)

        fused_attn = self.out_proj(fused_attn)

        # ===============================
        # 🔥 GLOBAL PRIOR
        # ===============================
        w = torch.softmax(self.logit_weights, dim=0)
        global_out = w[0]*p1 + w[1]*p2 + w[2]*p3

        # ===============================
        # 🔥 LEARNABLE FUSION
        # ===============================
        alpha = torch.sigmoid(self.alpha)
        fused = alpha * fused_attn + (1 - alpha) * global_out

        # ===============================
        out = self.head(fused)
        return out

class BoostedTransformer(nn.Module):
    def __init__(self, num_classes=3, embed_dim=128):
        super().__init__()

        # ===== Base models =====
        self.m1 = build_model("efficientnet_b3")
        self.m2 = build_model("mobilenet")
        self.m3 = build_model("resnet18")

        # Load pretrained
        self.m1.load_state_dict(torch.load(f"{MODEL_DIR}/efficientnet_b3_best.pth"))
        self.m2.load_state_dict(torch.load(f"{MODEL_DIR}/mobilenet_best.pth"))
        self.m3.load_state_dict(torch.load(f"{MODEL_DIR}/resnet18_best.pth"))

        # Freeze early layers, allow last layers to adapt
        for m in [self.m1, self.m2, self.m3]:
            for name, p in m.named_parameters():
                if "layer4" not in name and "head" not in name:
                    p.requires_grad = False

        # ===== Residual learners =====
        self.res1 = nn.Linear(num_classes, num_classes)
        self.res2 = nn.Linear(num_classes, num_classes)

        # ===== Token projection =====
        self.token_proj = nn.Linear(num_classes, embed_dim)

        # ===== Transformer encoder =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ===== Attention pooling =====
        self.pool = nn.Linear(embed_dim, 1)

        # ===== Final head =====
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        # ===== Stage 1 =====
        l1 = self.m1(x)

        # ===== Stage 2 (residual learning) =====
        l2 = self.m2(x)
        l2 = l2 + self.res1(l1)

        # ===== Stage 3 =====
        l3 = self.m3(x)
        l3 = l3 + self.res2(l2)

        # ===== Stack tokens =====
        tokens = torch.stack([l1, l2, l3], dim=1)  # (B,3,C)

        tokens = self.token_proj(tokens)

        # ===== Transformer =====
        tokens = self.transformer(tokens)

        # ===== Attention pooling =====
        weights = torch.softmax(self.pool(tokens), dim=1)
        fused = (weights * tokens).sum(dim=1)

        return self.head(fused)

class DropPath(nn.Module):
    """Randomly drops entire residual paths during training (stochastic depth)."""
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.rand(shape, device=x.device) < keep
        return x * mask / keep
    
class ConvBNAct(nn.Module):
    """Conv2d → BN → SiLU (or custom act).  padding auto-computed if omitted."""
 
    def __init__(
        self,
        in_ch : int,
        out_ch: int,
        k     : int                 = 3,
        stride: int                 = 1,
        groups: int                 = 1,
        dil   : int                 = 1,
        bias  : bool                = False,
        act   : Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        pad        = (k - 1) // 2 * dil
        self.conv  = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=pad,
                               dilation=dil, groups=groups, bias=bias)
        self.bn    = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.act   = act if act is not None else nn.SiLU(inplace=True)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class MultiScaleEdgeExtractor(nn.Module):
    """
    Fixed gradient-operator bank (7 channels) followed by a learnable
    depthwise-separable refinement head.
 
    Operator bank (each map normalised per-sample to [0, 1]):
        ch 0   Sobel magnitude     ||grad I||_2
        ch 1   |Sobel Ix|          horizontal edge strength
        ch 2   |Sobel Iy|          vertical edge strength
        ch 3   |Laplacian 5×5|     second-order isotropic response
        ch 4   |Scharr Ix|         rotation-invariant horizontal gradient
        ch 5   DoG σ=(1.0, 2.0)    fine-scale blob / edge
        ch 6   DoG σ=(2.0, 4.0)    coarse-scale boundary
 
    The fixed kernels are buffers (not parameters) so they follow
    .to(device) / .half() calls automatically.
 
    Refinement path:
        3×3 ConvBNAct   mix all 7 responses
        DW 3×3 ConvBNAct  spatial refinement (zero cross-channel cost)
        PW 1×1 ConvBNAct  channel mixing / dimensionality control
 
    Args:
        out_ch : refined output channels  (default 16)
    """
 
    _DOG_PAIRS: Tuple[Tuple[float, float], ...] = ((1.0, 2.0), (2.0, 4.0))
 
    def __init__(self, out_ch: int = 16) -> None:
        super().__init__()
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1., 0., 1.],
                          [-2., 0., 2.],
                          [-1., 0., 1.]]).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1., -2., -1.],
                          [ 0.,  0.,  0.],
                          [ 1.,  2.,  1.]]).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "scharr_x",
            torch.tensor([[ -3., 0.,  3.],
                          [-10., 0., 10.],
                          [ -3., 0.,  3.]]).view(1, 1, 3, 3),
        )
 
        self.refine = nn.Sequential(
            ConvBNAct(7, out_ch, k=3),
            ConvBNAct(out_ch, out_ch, k=3, groups=out_ch),  # depthwise
            ConvBNAct(out_ch, out_ch, k=1),                  # pointwise
        )
 
    @staticmethod
    def _minmax_norm(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        B, C, H, W = t.shape
        flat = t.reshape(B, C, -1)
        lo   = flat.min(dim=-1).values.view(B, C, 1, 1)
        hi   = flat.max(dim=-1).values.view(B, C, 1, 1)
        return (t - lo) / (hi - lo + eps)
 
    def _dog(self, gray: torch.Tensor) -> torch.Tensor:
        out: List[torch.Tensor] = []
        for s1, s2 in self._DOG_PAIRS:
            ks1 = max(3, 2 * math.ceil(2 * s1) + 1) | 1
            ks2 = max(3, 2 * math.ceil(2 * s2) + 1) | 1
            out.append(
                K.filters.gaussian_blur2d(gray, (ks1, ks1), (s1, s1))
                - K.filters.gaussian_blur2d(gray, (ks2, ks2), (s2, s2))
            )
        return torch.cat(out, dim=1)   # (B, 2, H, W)
 
    # AFTER — renamed to _conv2d to avoid collision
    def _conv2d(self, gray: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return F.conv2d(gray, kernel.to(dtype=gray.dtype), padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = K.color.rgb_to_grayscale(x)
        raw  = torch.cat([
            K.filters.sobel(gray),
            self._conv2d(gray, self.sobel_x).abs(),
            self._conv2d(gray, self.sobel_y).abs(),
            K.filters.laplacian(gray, kernel_size=5).abs(),
            self._conv2d(gray, self.scharr_x).abs(),
            self._dog(gray),
        ], dim=1)
        return self.refine(self._minmax_norm(raw))

class EdgeCrossAttention(nn.Module):
    """
    Spatial cross-attention: backbone features supply Q; edge maps supply K, V.
 
    This injects geometric structure cues (scarps, drainage channels,
    ridgelines) into backbone features without permanently widening the
    feature tensor.  A learnable scalar γ, initialised at 0, gates the
    module as a warm-up:
        output = feat + γ · cross_attn(feat, edge)
    so the module starts as an exact identity.
 
    Args:
        feat_ch : backbone feature channels (Q dimension)
        edge_ch : refined edge descriptor channels (K/V source)
        heads   : number of attention heads
        dropout : attention-weight dropout
    """
 
    def __init__(
        self,
        feat_ch: int,
        edge_ch: int,
        heads  : int   = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert feat_ch % heads == 0
        self.heads    = heads
        self.head_dim = feat_ch // heads
        self.scale    = self.head_dim ** -0.5
 
        self.q_proj = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 1, bias=False),
            nn.GroupNorm(heads, feat_ch),
        )
        self.k_proj = nn.Conv2d(edge_ch, feat_ch, 1, bias=False)
        self.v_proj = nn.Conv2d(edge_ch, feat_ch, 1, bias=False)
 
        self.out_proj = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 1, bias=False),
            nn.BatchNorm2d(feat_ch, momentum=0.01, eps=1e-3),
        )
        self.drop  = nn.Dropout(p=dropout)
        self.drop_path = DropPath(drop_prob=0.1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
    def forward(self, feat: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        N = H * W
 
        if edge.shape[-2:] != (H, W):
            edge = F.interpolate(edge, (H, W), mode="bilinear", align_corners=False)
 
        def _heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, self.heads, self.head_dim, N).permute(0, 1, 3, 2)
 
        Q  = _heads(self.q_proj(feat))
        K_ = _heads(self.k_proj(edge))
        V_ = _heads(self.v_proj(edge))
 
        w   = self.drop(F.softmax(torch.matmul(Q, K_.transpose(-2, -1)) * self.scale, dim=-1))
        out = torch.matmul(w, V_).permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        return feat + self.gamma * self.drop_path(self.out_proj(out))

class EdgeAttentionModule(nn.Module):
    """
    Multi-Scale Edge Attention Module (MEAM).
 
    Composes MultiScaleEdgeExtractor + EdgeCrossAttention.
    The extractor always receives the original full-resolution image so that
    fine boundary detail is never lost to intermediate downsampling.
 
    Inserted after features[3] (C=48) because:
        - Mid-resolution (~28×28 for 224px) preserves boundary spatial coherence.
        - This is the FPN P3 tap so edge-enriched features feed the finest
          pyramid level directly.
 
    Args:
        feat_ch  : backbone feature channels at insertion point (48 for B3)
        edge_ch  : edge descriptor output channels
        heads    : cross-attention heads
        dropout  : attention dropout
    """
 
    def __init__(
        self,
        feat_ch: int,
        edge_ch: int   = 16,
        heads  : int   = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.extractor  = MultiScaleEdgeExtractor(out_ch=edge_ch)
        self.cross_attn = EdgeCrossAttention(feat_ch, edge_ch, heads, dropout)
        self.post_norm  = nn.BatchNorm2d(feat_ch, momentum=0.01, eps=1e-3)
 
    def forward(self, feat: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        feat : (B, C, H, W)    backbone feature at features[3] output
        img  : (B, 3, H0, W0)  original full-resolution input image
        """
        return self.post_norm(self.cross_attn(feat, self.extractor(img)))

class DilatedMultiScaleBranch(nn.Module):
    """
    Four parallel dilated depthwise-separable convolution branches with
    learnable per-branch softmax mixture weights.
 
    Dilation rates {1, 2, 4, 8} give exponentially growing effective
    receptive fields at identical per-branch parameter counts.  The outputs
    are combined as a learned convex combination and projected back to in_ch.
 
    Each branch:
        DW Conv2d (rate r, 3×3) → BN → SiLU → PW Conv2d (1×1)
 
    Args:
        in_ch  : input (and output) channel count
        mid_ch : internal width per branch
    """
 
    _RATES: Tuple[int, ...] = (1, 2, 4, 8)
 
    def __init__(self, in_ch: int, mid_ch: int) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch, 3,
                          padding=r, dilation=r, groups=mid_ch, bias=False),
                nn.BatchNorm2d(mid_ch, momentum=0.01, eps=1e-3),
                nn.SiLU(inplace=True),
                nn.Conv2d(mid_ch, mid_ch, 1, bias=False),
            )
            for r in self._RATES
        ])
        self.branch_weights = nn.Parameter(torch.ones(len(self._RATES)))
        self.expand = nn.Sequential(
            nn.Conv2d(mid_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch, momentum=0.01, eps=1e-3),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.reduce(x)
        w = F.softmax(self.branch_weights, dim=0)
        out = sum(w_i * branch(h) for w_i, branch in zip(w.unbind(), self.branches))
        return self.expand(out)

class SCAM_Enhanced(nn.Module):
    """
    Enhanced Spatial-Channel Attention Module (SCAM+).
 
    Novel contributions over the original SCAM:
        1. SE-style channel excitation (avg + max → shared MLP → sigmoid)
           pre-weights the input before computing the combined descriptor Fsc.
        2. Dilated multi-scale branch (rates {1,2,4,8} DW-separable) replaces
           fixed-kernel multi-branch for broader receptive field coverage.
        3. Fast 1×1 shortcut branch concatenated with the dilated output for
           dual-path aggregation before the final projection.
        4. Sigmoid residual gate γ initialised at 0.1:
               output = x + x ⊙ attn ⊙ γ
 
    Inserted after features[5] (C=136) because:
        - Mid-semantic depth encodes landslide texture and shape patterns.
        - Dilated branches are most cost-effective at this intermediate depth.
        - This is the FPN P4 tap; calibrating features here improves the
          quality of the mid-level pyramid contribution.
 
    Args:
        in_ch     : input / output channel count
        reduction : channel excitation MLP bottleneck ratio
    """
 
    def __init__(self, in_ch: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(in_ch // reduction, 8)
 
        self.ch_mlp = nn.Sequential(
            nn.Linear(in_ch, mid),
            nn.GELU(),
            nn.Linear(mid, in_ch),
        )
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid, momentum=0.01, eps=1e-3),
            nn.SiLU(inplace=True),
        )
        self.dilated  = DilatedMultiScaleBranch(mid, max(mid // 2, 4))
        self.shortcut = nn.Sequential(
            nn.Conv2d(mid, mid, 1, bias=False),
            nn.BatchNorm2d(mid, momentum=0.01, eps=1e-3),
        )
        self.aggr = nn.Sequential(
            nn.Conv2d(mid * 2, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch, momentum=0.01, eps=1e-3),
        )
        self.gate    = nn.Parameter(torch.tensor(0.1))
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
 
        avg_c   = F.adaptive_avg_pool2d(x, 1).view(B, C)
        max_c   = F.adaptive_max_pool2d(x, 1).view(B, C)
        ch_attn = self.sigmoid(self.ch_mlp(avg_c) + self.ch_mlp(max_c)).view(B, C, 1, 1)
 
        sp_bias = (x.mean(dim=1, keepdim=True) + x.max(dim=1, keepdim=True).values) * 0.5
        Fsc     = x * ch_attn * (1.0 + sp_bias)
 
        h    = self.spatial_enc(Fsc)
        attn = self.sigmoid(self.aggr(torch.cat([self.dilated(h), self.shortcut(h)], dim=1)))
        return x + x * attn * self.gate
 
class ChannelAttention3D(nn.Module):
    """
    Channel Attention with 3-D Convolution over a Spatial-Pyramid Descriptor.
 
    Standard CBAM-CA uses single-scale global pooling → MLP, discarding all
    spatial structure.  This module pools at two scales (1×1, 2×2) yielding
    5 descriptors per channel, packed into a 3-D volume (B, 1, C, 5, 1).
    A Conv3d with kernel (3,1,1) attends to inter-channel neighbourhoods
    jointly with the pooling-scale axis, a strictly richer excitation.
 
    Inserted after features[7] (C=384) because high-level abstract features
    benefit from pyramid-pooling context, and at this depth the spatial
    resolution (~7×7) is too low for dilated convolutions to be meaningful.
 
    Args:
        channels  : input channel count  (384 for B3 features[7])
        reduction : MLP bottleneck ratio
    """
 
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
 
        self.conv3d_avg = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv3d_max = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )
        self.sigmoid = nn.Sigmoid()
 
    def _pack(self, x: torch.Tensor, pool_fn) -> torch.Tensor:
        B, C = x.shape[:2]
        p1 = pool_fn(x, 1).view(B, C, 1)
        p2 = pool_fn(x, 2).view(B, C, 4)
        return torch.cat([p1, p2], dim=2).unsqueeze(1).unsqueeze(-1)  # (B,1,C,5,1)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg_vol = self._pack(x, F.adaptive_avg_pool2d)
        max_vol = self._pack(x, F.adaptive_max_pool2d)
 
        avg_out = self.conv3d_avg(avg_vol).squeeze(1).squeeze(-1)  # (B,C,5)
        max_out = self.conv3d_max(max_vol).squeeze(1).squeeze(-1)
 
        ca = self.sigmoid(
            self.mlp(avg_out.mean(dim=-1)) + self.mlp(max_out.mean(dim=-1))
        )
        return x * ca.view(B, C, 1, 1)

class SpatialAttention3D(nn.Module):
    """
    Spatial Attention with 3-D Convolution over a Four-Statistic Volume.
 
    Standard CBAM-SA stacks {mean, max} → (B, 2, H, W) and uses a 2-D conv,
    losing inter-statistic relationships.  This module computes four statistics
    {mean, max, std, min} reshaped as a 3-D volume (B, 1, 4, H, W).  A Conv3d
    with depth kernel D=3 jointly models inter-statistic correlations and
    spatial context.  A second Conv3d (D=4) collapses the statistic axis with
    learned weights (not a fixed mean).
 
    Args:
        kernel_size : 2-D spatial kernel of the joint Conv3d (default 7)
    """
 
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv3d_joint = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3, kernel_size, kernel_size),
                      padding=(1, pad, pad), bias=False),
            nn.BatchNorm3d(1, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.stat_collapse = nn.Conv3d(1, 1, kernel_size=(4, 1, 1), bias=False)
        self.sigmoid        = nn.Sigmoid()
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_s   = x.mean(dim=1, keepdim=True)
        max_s, _ = x.max(dim=1, keepdim=True)
        std_s    = x.std(dim=1, keepdim=True)
        min_s, _ = x.min(dim=1, keepdim=True)
 
        vol = torch.cat([mean_s, max_s, std_s, min_s], dim=1).unsqueeze(1)  # (B,1,4,H,W)
        out = self.stat_collapse(self.conv3d_joint(vol))                      # (B,1,1,H,W)
        return x * self.sigmoid(out.squeeze(2))

class CBAM3D(nn.Module):
    """
    3-D Convolutional Bottleneck Attention Module.
 
    Applies ChannelAttention3D → SpatialAttention3D sequentially, wrapped in
    a soft residual gate initialised at 0 (module is an identity at the start
    of training and gradually activates):
        output = x + sigmoid(gate) * (CBAM(x) - x)
 
    Inserted after features[7] (C=384) where high-level semantic features
    benefit from pyramid-pooling attention.
 
    Args:
        channels       : input / output channel count
        reduction      : channel MLP bottleneck ratio
        spatial_kernel : spatial kernel size of the 3-D spatial attention
    """
 
    def __init__(
        self,
        channels      : int = 384,
        reduction     : int = 16,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention3D(channels, reduction)
        self.spatial_attn = SpatialAttention3D(spatial_kernel)
        self.gate         = nn.Parameter(torch.zeros(1))
        self.drop_path = DropPath(drop_prob=0.15)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.spatial_attn(self.channel_attn(x))
        return x + torch.sigmoid(self.gate) * self.drop_path(attended - x)

class LateralFusionBlock(nn.Module):
    """
    One FPN lateral connection: coarser top-down map fused with a finer
    lateral map via learnable α-blending.
 
    fused = sigmoid(α) * proj(upsample(top)) + (1-sigmoid(α)) * proj(lat)
 
    A 3×3 ConvBNAct smooths the fused output.
 
    Args:
        top_ch : coarser input channels
        lat_ch : finer input channels
        out_ch : unified output channels
    """
 
    def __init__(self, top_ch: int, lat_ch: int, out_ch: int) -> None:
        super().__init__()
        self.top_proj = nn.Sequential(
            nn.Conv2d(top_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
        )
        self.lat_proj = nn.Sequential(
            nn.Conv2d(lat_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
        )
        self.smooth = ConvBNAct(out_ch, out_ch, k=3)
        self.alpha  = nn.Parameter(torch.tensor(0.5))
 
    def forward(self, top: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        top_up  = F.interpolate(self.top_proj(top), size=lat.shape[-2:],
                                mode="bilinear", align_corners=False)
        a       = torch.sigmoid(self.alpha)
        return self.smooth(a * top_up + (1.0 - a) * self.lat_proj(lat))

class MultiScaleFeatureFusion(nn.Module):
    """
    3-level top-down FPN: P5 (coarse) → P4 → P3 (fine).
 
    The fused P3-resolution output is refined by an additional CBAM3D to
    suppress upsampling artefacts introduced by the top-down path.
 
    EfficientNet-B3 channel widths at each tap:
        P3: 48   (after features[3] + EdgeAttn)
        P4: 136  (after features[5] + SCAM+)
        P5: 1536 (after features[8], top conv)
 
    Args:
        ch_p3  : P3 channel count
        ch_p4  : P4 channel count
        ch_p5  : P5 channel count
        out_ch : unified FPN output channels
    """
 
    def __init__(
        self,
        ch_p3 : int = 48,
        ch_p4 : int = 136,
        ch_p5 : int = 1536,
        out_ch: int = 256,
    ) -> None:
        super().__init__()
        self.p5_to_p4 = LateralFusionBlock(ch_p5, ch_p4, out_ch)
        self.p4_to_p3 = LateralFusionBlock(out_ch, ch_p3, out_ch)
        self.refine   = nn.Sequential(
            CBAM3D(out_ch, reduction=16, spatial_kernel=7),
            ConvBNAct(out_ch, out_ch, k=1),
        )
 
    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> torch.Tensor:
        m4 = self.p5_to_p4(p5, p4)
        m3 = self.p4_to_p3(m4, p3)
        return self.refine(m3)
 
class ClassificationHead(nn.Module):
    """
    Multi-scale pool (1×1 + 2×2) → LayerNorm → 3-layer GELU MLP.
 
    Pools the FPN output at two spatial scales:
        1×1 global average → (B, C)
        2×2 average        → (B, 4C)   four spatial quadrant descriptors
    Concatenated to (B, 5C), normalised by LayerNorm, then MLP → logits.
 
    Args:
        in_ch       : FPN output channel count
        num_classes : number of output classes
        dropout     : dropout after the first two linear layers
    """
 
    def __init__(self, in_ch: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        feat_dim = in_ch * 5
        self.norm = nn.LayerNorm(feat_dim)
        self.mlp  = nn.Sequential(
            nn.Linear(feat_dim, in_ch),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_ch, in_ch // 2),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(in_ch // 2, num_classes),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1 = F.adaptive_avg_pool2d(x, 1).flatten(1)
        p2 = F.adaptive_avg_pool2d(x, 2).flatten(1)
        return self.mlp(self.norm(torch.cat([p1, p2], dim=1)))

class LandslideEfficientNet(nn.Module):
    """
    Pretrained EfficientNet-B3 backbone with novel attention modules inserted
    at three principled tap points, fused by a lightweight FPN.
 
    Forward flow:
        img
         │
         ├─ features[0..3] ──► EdgeAttentionModule(feat, img) ──► P3 (C=48)
         │                                                          │
         ├─ features[4..5] ──► SCAM_Enhanced ──────────────────► P4 (C=136)
         │                                                          │
         ├─ features[6..7] ──► CBAM3D ──► features[8] ──────────► P5 (C=1536)
         │                                                          │
         └────────────────────────────────────── FPN(P3,P4,P5) ──► ClassificationHead
                                                                    │
                                                                 logits (B, num_classes)
 
    Channel widths at tap points are fixed by EfficientNet-B3:
        P3: 48  P4: 136  P5: 1536
 
    Args:
        num_classes : number of output classes  (default 3)
        fpn_ch      : unified FPN output channels  (default 256)
        dropout     : classification head dropout  (default 0.3)
        pretrained  : load ImageNet-1K weights for the backbone  (default True)
    """
 
    # EfficientNet-B3 output channels at each stage boundary
    _CH_P3:  int = 48     # features[3] output
    _CH_P4:  int = 136    # features[5] output
    _CH_PRE_TOP: int = 384    # features[7] output  (CBAM3D applied here)
    _CH_P5:  int = 1536   # features[8] output  (top conv)
 
    def __init__(
        self,
        num_classes: int  = 3,
        fpn_ch     : int  = 256,
        dropout    : float = 0.3,
        pretrained : bool  = True,
    ) -> None:
        super().__init__()
 
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)
 
        # features[0..8]; we split them for manual stage-wise execution
        self.f0_3 = nn.Sequential(
            backbone.features[0],
            backbone.features[1],
            backbone.features[2],
            backbone.features[3],
        )
        self.f4_5 = nn.Sequential(
            backbone.features[4],
            backbone.features[5],
        )
        self.f6_7 = nn.Sequential(
            backbone.features[6],
            backbone.features[7],
        )
        self.f8 = backbone.features[8]   # top conv, out=1536
 
        self.edge_attn = EdgeAttentionModule(
            feat_ch = self._CH_P3,
            edge_ch = 16,
            heads   = 4,           # 48 / 4 = 12 head_dim
            dropout = 0.1,
        )
 
        self.scam = SCAM_Enhanced(
            in_ch     = self._CH_P4,
            reduction = 16,
        )
 
        self.cbam3d = CBAM3D(
            channels       = self._CH_PRE_TOP,
            reduction      = 16,
            spatial_kernel = 7,
        )
 
        self.fpn  = MultiScaleFeatureFusion(
            ch_p3  = self._CH_P3,
            ch_p4  = self._CH_P4,
            ch_p5  = self._CH_P5,
            out_ch = fpn_ch,
        )
        self.head = ClassificationHead(fpn_ch, num_classes, dropout)
 
        # Initialise only the new (non-pretrained) modules
        self._init_new_weights()
 
    # ── Weight initialisation (new modules only) ──────────────────────
 
    def _init_new_weights(self) -> None:
        new_modules = [self.edge_attn, self.scam, self.cbam3d, self.fpn, self.head]
        for module in new_modules:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = x

        # Backbone: strictly sequential, unmodified flow
        f3 = self.f0_3(img)       # raw backbone features at stage 3
        f5 = self.f4_5(f3)        # raw backbone features at stage 5
        f7 = self.f6_7(f5)        # raw backbone features at stage 7

        # FPN P5: apply CBAM3D on f7 then top conv
        p5 = self.f8(self.cbam3d(f7))   # (B, 1536, H/32, W/32) — same as before

        # FPN taps: enrich raw backbone features, NOT the backbone flow
        p3 = self.edge_attn(f3, img)    # side-enrichment of f3
        p4 = self.scam(f5)              # side-enrichment of f5

        return self.head(self.fpn(p3, p4, p5))
 
    def count_parameters(self) -> Dict[str, int]:
        """Per-subsystem trainable parameter counts for the complexity table."""
        def _n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
 
        return {
            "backbone_f0_3" : _n(self.f0_3),
            "backbone_f4_5" : _n(self.f4_5),
            "backbone_f6_7" : _n(self.f6_7),
            "backbone_f8"   : _n(self.f8),
            "edge_attn"     : _n(self.edge_attn),
            "scam"          : _n(self.scam),
            "cbam3d"        : _n(self.cbam3d),
            "fpn"           : _n(self.fpn),
            "head"          : _n(self.head),
            "total"         : _n(self),
        }

class LandslideEfficientNet2(nn.Module):
    """
    EfficientNetV2-M backbone with:
    - GeM + avg dual pooling (NL insight: two "memory levels" of spatial compression)
    - Lightweight SE channel attention (no FPN, no cross-attention)
    - Simple MLP head
    
    Why this beats the complex version:
    - FPN was designed for detection (spatial localization), not classification
    - Full spatial cross-attention (N=784) is unstable for 3-class classification
    - EfficientNetV2-M has stronger backbone than B3 (54M vs 12M params)
    """

    def __init__(
        self,
        num_classes: int   = 3,
        dropout    : float = 0.3,
        pretrained : bool  = True,
    ) -> None:
        super().__init__()

        backbone = models.efficientnet_v2_m(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        self.features = backbone.features   # out: (B, 1280, H/32, W/32)
        feat_ch = 1280

        # SE channel attention — proven lightweight, compatible with classification
        # NL framing: this is a "higher-frequency" inner loop that recalibrates
        # channel activations conditioned on the spatial context of each forward pass
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_ch, feat_ch // 16, bias=False),
            nn.SiLU(),
            nn.Linear(feat_ch // 16, feat_ch, bias=False),
            nn.Sigmoid(),
        )

        # GeM pooling — learns p ∈ [1, 6] interpolating between avg and max
        # Better than fixed avg for remote sensing where discriminative regions
        # are spatially concentrated (landslide scarps, drainage patterns)
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)

        # Dual pooling head — two pooling strategies as parallel "memory reads"
        # GeM: captures discriminative local peaks
        # Avg: captures global context distribution
        # Concatenated so the classifier can weight each signal
        feat_dim = feat_ch * 2   # GeM(1280) + Avg(1280)

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512, bias=False),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256, bias=False),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

        self._init_new_weights()

    def _gem(self, x: torch.Tensor) -> torch.Tensor:
        p = self.gem_p.clamp(min=1.0, max=6.0)
        return F.adaptive_avg_pool2d(
            x.clamp(min=1e-6).pow(p), 1
        ).pow(1.0 / p).flatten(1)

    def _init_new_weights(self) -> None:
        for m in [self.channel_attn, self.head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)                                    # (B, 1280, H', W')

        # SE channel recalibration
        attn = self.channel_attn(feat).view(feat.size(0), -1, 1, 1)
        feat = feat * attn

        # Dual pooling
        gem = self._gem(feat)                                      # (B, 1280)
        avg = F.adaptive_avg_pool2d(feat, 1).flatten(1)           # (B, 1280)

        return self.head(torch.cat([gem, avg], dim=1))

    def count_parameters(self) -> Dict[str, int]:
        def _n(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "features"      : _n(self.features),
            "channel_attn"  : _n(self.channel_attn),
            "gem_p"         : 1,
            "head"          : _n(self.head),
            "total"         : _n(self),
        }

# ── NEW MODULE ────────────────────────────────────────────────────────────────

class CrossScaleGate(nn.Module):
    """
    Per-sample adaptive weighting of 3 semantic scale contributions.
    
    Each scale is projected to a common dimension. A lightweight gating MLP
    then computes soft weights conditioned on all three scale representations
    jointly — allowing the model to attend more to texture (early) vs
    semantics (late) depending on the input sample content.
    
    This is the core novelty of MSHAN: unlike FPN (fixed upsampling) or
    simple concatenation (fixed equal weight), the gate is input-adaptive.
    """
    def __init__(self, proj_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim // 4, bias=False),
            nn.GELU(),
            nn.Linear(proj_dim // 4, 3, bias=False),
        )

    def forward(self, e: torch.Tensor, m: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        # gates: (B, 3) → softmax weights
        gates = torch.softmax(self.gate(torch.cat([e, m, l], dim=1)), dim=1)
        return gates[:, 0:1] * e + gates[:, 1:2] * m + gates[:, 2:3] * l

class LandslideEfficientNet3(nn.Module):
    """
    Multi-Scale Hierarchical Aggregation Network (MSHAN).

    Architecture rationale for Q1 paper:
    ─────────────────────────────────────
    Plain EfficientNet discards early/mid feature maps entirely —
    the classification head only sees the final 1280-ch top conv.
    For remote sensing scenes, early-stage features encode fine-grained
    texture cues (drainage channels, scarp roughness) that are absent
    in the deep semantic representation. MSHAN taps three semantic levels
    and fuses them via a per-sample cross-scale gate.

    Contributions vs. plain EfficientNetV2-M:
      1. Three-level feature tapping (early/mid/late) preserves texture +
         semantic information simultaneously.
      2. Cross-scale gating (CSG): lightweight input-adaptive MLP computes
         soft per-scale weights — replacing fixed equal-weight fusion.
      3. Dual-pooling at the top level: GeM (discriminative peaks) + avg
         (global context), concatenated before projection.
      4. SE channel recalibration at each tap point, keeping the
         per-level representation compact before projection.

    Forward flow:
        img → stage_early (features[0..3]) → SE → avg pool → proj_e ─┐
            → stage_mid   (features[4..6]) → SE → avg pool → proj_m ─┼→ CrossScaleGate → head → logits
            → stage_late  (features[7..8]) → SE → GeM+avg  → proj_l ─┘

    EfficientNetV2-M channel widths at each tap:
        early : 80    (features[3] output)
        mid   : 304   (features[6] output)
        late  : 1280  (features[8] output, top conv)
    """

    _CH_EARLY: int = 80
    _CH_MID:   int = 304
    _CH_LATE:  int = 1280

    def __init__(
        self,
        num_classes: int   = 3,
        proj_dim   : int   = 384,
        dropout    : float = 0.3,
        pretrained : bool  = True,
    ) -> None:
        super().__init__()

        backbone = models.efficientnet_v2_m(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        feats = backbone.features  # 9 blocks (indices 0-8)

        # Split backbone at principled semantic boundaries
        self.stage_early = nn.Sequential(*[feats[i] for i in range(4)])    # → 80ch
        self.stage_mid   = nn.Sequential(*[feats[i] for i in range(4, 7)]) # → 304ch
        self.stage_late  = nn.Sequential(*[feats[i] for i in range(7, 9)]) # → 1280ch

        # SE channel recalibration per level (lightweight, no spatial mixing)
        self.se_early = self._make_se(self._CH_EARLY)
        self.se_mid   = self._make_se(self._CH_MID)
        self.se_late  = self._make_se(self._CH_LATE)

        # GeM pooling parameter for top level
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)

        # Per-level projections to unified dimension
        self.proj_early = nn.Sequential(
            nn.Linear(self._CH_EARLY, proj_dim, bias=False),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.proj_mid = nn.Sequential(
            nn.Linear(self._CH_MID, proj_dim, bias=False),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.proj_late = nn.Sequential(
            # GeM(1280) + avg(1280) → concatenated before projection
            nn.Linear(self._CH_LATE * 2, proj_dim, bias=False),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # Cross-scale gating: input-adaptive per-sample scale weights
        self.cross_gate = CrossScaleGate(proj_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim // 2, bias=False),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(proj_dim // 2, num_classes),
        )

        self._init_new_weights()

    @staticmethod
    def _make_se(ch: int, reduction: int = 16) -> nn.Module:
        mid = max(ch // reduction, 8)
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, ch, bias=False),
            nn.Sigmoid(),
        )

    def _gem(self, x: torch.Tensor) -> torch.Tensor:
        p = self.gem_p.clamp(min=1.0, max=6.0)
        return F.adaptive_avg_pool2d(x.clamp(min=1e-6).pow(p), 1).pow(1.0 / p).flatten(1)

    def _init_new_weights(self) -> None:
        new = [self.se_early, self.se_mid, self.se_late,
               self.proj_early, self.proj_mid, self.proj_late,
               self.cross_gate, self.head]
        for module in new:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sequential backbone — each stage feeds the next
        f_early = self.stage_early(x)          # (B, 80,  H/8,  W/8)
        f_mid   = self.stage_mid(f_early)       # (B, 304, H/16, W/16)
        f_late  = self.stage_late(f_mid)        # (B, 1280,H/32, W/32)

        # SE recalibration (channel-wise, no spatial ops)
        f_early = f_early * self.se_early(f_early).view(f_early.size(0), -1, 1, 1)
        f_mid   = f_mid   * self.se_mid(f_mid).view(f_mid.size(0), -1, 1, 1)
        f_late  = f_late  * self.se_late(f_late).view(f_late.size(0), -1, 1, 1)

        # Level-appropriate pooling
        p_early = F.adaptive_avg_pool2d(f_early, 1).flatten(1)              # avg: texture
        p_mid   = F.adaptive_avg_pool2d(f_mid, 1).flatten(1)                # avg: mid-semantic
        p_late  = torch.cat([self._gem(f_late),                             # GeM: discriminative
                              F.adaptive_avg_pool2d(f_late, 1).flatten(1)], # avg: global
                             dim=1)

        # Project each level to common dimension
        e = self.proj_early(p_early)    # (B, proj_dim)
        m = self.proj_mid(p_mid)        # (B, proj_dim)
        l = self.proj_late(p_late)      # (B, proj_dim)

        # Cross-scale gating → fused representation
        fused = self.cross_gate(e, m, l)    # (B, proj_dim)

        return self.head(fused)

    def count_parameters(self) -> Dict[str, int]:
        def _n(mod): return sum(p.numel() for p in mod.parameters() if p.requires_grad)
        return {
            "stage_early" : _n(self.stage_early),
            "stage_mid"   : _n(self.stage_mid),
            "stage_late"  : _n(self.stage_late),
            "se_modules"  : _n(self.se_early) + _n(self.se_mid) + _n(self.se_late),
            "projections" : _n(self.proj_early) + _n(self.proj_mid) + _n(self.proj_late),
            "cross_gate"  : _n(self.cross_gate),
            "head"        : _n(self.head),
            "total"       : _n(self),
        }
    
def load_balance_loss(router_logits):
    # router_logits: (B, num_experts)

    probs = torch.softmax(router_logits, dim=-1)  # (B, E)

    # importance = how often each expert is used
    importance = probs.mean(dim=0)  # (E,)

    # encourage uniform usage
    loss = (importance * importance).sum()

    return loss

def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)

    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def unfreeze_last_layers(model, ratio=0.3):
    params = [p for p in model.parameters() if not p.requires_grad]
    n = int(len(params) * ratio)

    for p in params[-n:]:
        p.requires_grad = True

def build_model(name):
    
    if name == "stack_ensembled":
        return StackedEnsemble(num_classes=NUM_CLASSES)
    
    elif name == "weighted_ensembled":
        return WeightedEnsemble(num_classes=NUM_CLASSES)
    
    elif name == "boosted_transformer":
        return BoostedTransformer(num_classes=NUM_CLASSES)
    
    elif name == "landslide_efficientnet":
        return LandslideEfficientNet(num_classes=NUM_CLASSES)
    
    elif name == "landslide_efficientnet_2":
        return LandslideEfficientNet2(num_classes=NUM_CLASSES)
    
    elif name == "landslide_efficientnet_3":
        return LandslideEfficientNet3(num_classes=NUM_CLASSES)
    
    elif name == "resnet18":
        try:
            m = models.resnet18(weights="IMAGENET1K_V1")
        except:
            m = models.resnet18(pretrained=True)

        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        return m
    
    elif name == "mobilenet":
        try:
            m = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        except:
            m = models.mobilenet_v3_small(pretrained=True)

        # Replace classifier head safely
        if hasattr(m, "classifier") and len(m.classifier) >= 4:
            m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
        else:
            m.classifier = nn.Sequential(
                nn.Linear(m.classifier[0].in_features, NUM_CLASSES)
            )
        return m
    
    elif name == "efficientnet_b3":
        try:
            m = models.efficientnet_b3(weights="IMAGENET1K_V1")
        except:
            m = models.efficientnet_b3(pretrained=True)
        # m = models.efficientnet_b3(pretrained=False)
        # Replace classification layer
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
        return m
    
    else:
        raise ValueError("Unknown model: %s" % name)
    
def clean_state_dict(state_dict):
    """Remove 'module.' prefix when loading DataParallel models."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # remove "module."
        else:
            new_state_dict[k] = v
    return new_state_dict

def _set_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag
 
 
def freeze_all_except_head(model: LandslideEfficientNet) -> None:
    """
    Freeze everything; train the classification head only.
 
    Use case: warm-up phase when adapting to a new dataset.  Avoids
    destructive gradient updates to pretrained backbone weights before the
    output layer has learned a reasonable mapping.
    """
    _set_grad(model, False)
    _set_grad(model.head, True)
  
def freeze_backbone_train_attention_and_head(model: LandslideEfficientNet) -> None:
    """
    Freeze the entire backbone (f0_3, f4_5, f6_7, f8).
    Train attention modules (edge_attn, scam, cbam3d), FPN, and head.
 
    Use case: ablation study isolating the contribution of the novel modules.
    Directly answers the research question: how much do the custom attention
    designs contribute beyond the pretrained backbone features?
    """
    for m in [model.f0_3, model.f4_5, model.f6_7, model.f8]:
        _set_grad(m, False)
    for m in [model.edge_attn, model.scam, model.cbam3d, model.fpn, model.head]:
        _set_grad(m, True)
  
def freeze_early_backbone(model: LandslideEfficientNet) -> None:
    """
    Freeze f0_3 (stem + S1-S3).
    Train f4_5, f6_7, f8, attention modules, FPN, and head.
 
    Use case: standard transfer-learning recipe.  Low-level filters in early
    stages (Gabor-like edges, colour blobs) are domain-invariant and rarely
    need updating for a remote-sensing target dataset.
    """
    _set_grad(model.f0_3, False)
    for m in [model.f4_5, model.f6_7, model.f8,
              model.edge_attn, model.scam, model.cbam3d,
              model.fpn, model.head]:
        _set_grad(m, True)
 
def unfreeze_all(model: LandslideEfficientNet) -> None:
    """
    Make all parameters trainable.
    Use case: final full fine-tune after the head has converged.
    """
    _set_grad(model, True)
 
def zero_lr_for_frozen(optimizer, model):
    """Set LR=0 for any param group whose params are all frozen."""
    for group in optimizer.param_groups:
        all_frozen = all(not p.requires_grad for p in group["params"])
        if all_frozen:
            group["lr"] = 0.0


def gradual_unfreeze_v1(
    model        : "LandslideEfficientNet",
    current_epoch: int,
    warmup       : int = 5,
    phase1_end   : int = 10,
    phase2_end   : int = 15,
    phase3_end   : int = 20,
) -> str:
    """Safe 4-phase unfreeze for LandslideEfficientNet (B3 + FPN)."""
    if current_epoch < warmup:
        _set_grad(model, False)
        _set_grad(model.head, True)
        return "phase_0__head_only"
    if current_epoch < phase1_end:
        _set_grad(model, False)
        for m in [model.f6_7, model.f8, model.cbam3d, model.fpn, model.head]:
            _set_grad(m, True)
        return "phase_1__late_backbone+cbam3d+fpn"
    if current_epoch < phase2_end:
        _set_grad(model, False)
        for m in [model.f4_5, model.f6_7, model.f8,
                  model.edge_attn, model.scam, model.cbam3d,
                  model.fpn, model.head]:
            _set_grad(m, True)
        return "phase_2__mid+late_backbone+all_attn"
    if current_epoch < phase3_end:
        _set_grad(model, False)
        for m in [model.f4_5, model.f6_7, model.f8,
                  model.edge_attn, model.scam, model.cbam3d,
                  model.fpn, model.head]:
            _set_grad(m, True)
        return "phase_3__mid+late_backbone+all_attn_continued"
    _set_grad(model, False)
    for m in [model.f0_3, model.f4_5, model.f6_7, model.f8,
              model.edge_attn, model.scam, model.cbam3d,
              model.fpn, model.head]:
        _set_grad(m, True)
    return "phase_4__full_finetune"

def gradual_unfreeze_v2(
    model        : "LandslideEfficientNet2",
    current_epoch: int,
    warmup       : int = 5,
    mid_start    : int = 10,
    late_start   : int = 15,
    full_start   : int = 20,
) -> str:
    """Safe 4-phase unfreeze for LandslideEfficientNet2 (V2-M simple)."""
    feats = list(model.features.children())
    early, mid, late = feats[:2], feats[2:5], feats[5:]

    def enable(*lists):
        for lst in lists:
            for m in lst:
                _set_grad(m, True)

    _set_grad(model, False)
    new_modules = [model.channel_attn, model.head]

    if current_epoch < warmup:
        enable(new_modules)
        model.gem_p.requires_grad = True
        return "phase_0__head_only"
    if current_epoch < mid_start:
        enable(new_modules, late)
        model.gem_p.requires_grad = True
        return "phase_1__head+late_backbone"
    if current_epoch < late_start:
        enable(new_modules, mid, late)
        model.gem_p.requires_grad = True
        return "phase_2__head+mid+late_backbone"
    if current_epoch < full_start:
        enable(new_modules, mid, late)
        model.gem_p.requires_grad = True
        return "phase_3__head+mid+late_backbone_continued"
    enable(new_modules, early, mid, late)
    model.gem_p.requires_grad = True
    return "phase_4__full_model"

# def gradual_unfreeze_v3(
#     model        : "LandslideEfficientNet3",
#     current_epoch: int,
#     warmup       : int = 5,
#     mid_start    : int = 10,
#     late_start   : int = 15,
#     full_start   : int = 20,
# ) -> str:
#     """
#     Safe 4-phase unfreeze for LandslideEfficientNet3 (MSHAN).
    
#     Phase 0: head + projections + gate (new modules only)
#     Phase 1: + stage_late + SE_late
#     Phase 2: + stage_mid  + SE_mid
#     Phase 3: same as phase 2 (stabilize before full)
#     Phase 4: + stage_early + SE_early
#     """
#     _set_grad(model, False)

#     new = [model.proj_early, model.proj_mid, model.proj_late,
#            model.cross_gate, model.head]
#     late_parts = [model.stage_late, model.se_late]
#     mid_parts  = [model.stage_mid,  model.se_mid]
#     early_parts= [model.stage_early,model.se_early]

#     def enable(*lists):
#         for lst in lists:
#             for m in lst:
#                 _set_grad(m, True)

#     model.gem_p.requires_grad = True

#     if current_epoch < warmup:
#         enable(new)
#         return "phase_0__head+projections+gate"
#     if current_epoch < mid_start:
#         enable(new, late_parts)
#         return "phase_1__+stage_late"
#     if current_epoch < late_start:
#         enable(new, late_parts, mid_parts)
#         return "phase_2__+stage_mid"
#     if current_epoch < full_start:
#         enable(new, late_parts, mid_parts)
#         return "phase_3__mid+late_continued"
#     enable(new, late_parts, mid_parts, early_parts)
#     return "phase_4__full_model"

def gradual_unfreeze_v3(
    model        : LandslideEfficientNet3,
    current_epoch: int,
    warmup       : int = 3,   # was 5 — shorter, less wasted time
    full_start   : int = 8,   # was 20 — much earlier full unfreeze
) -> str:
    """
    Simplified 2-phase unfreeze matching plain EfficientNet's dynamics.
    
    Plain EfficientNet trains ALL params from epoch 1 and gets 95%.
    Custom models froze backbone for 20 epochs — they never caught up.
    
    Phase 0 [0, warmup):   new modules only (let projections/gate initialize)
    Phase 1 [warmup, full): + late + mid backbone (main learning)
    Phase 2 [full, ∞):     everything including early backbone
    
    Full backbone active by epoch 8, giving 52 epochs of full training
    vs the old setup which only had 40 epochs of full training.
    """
    _set_grad(model, False)
    model.gem_p.requires_grad = True

    new_modules = [
        model.proj_early, model.proj_mid, model.proj_late,
        model.cross_gate, model.head,
        model.se_early, model.se_mid, model.se_late,
    ]

    def enable(*lists):
        for lst in lists:
            for m in lst:
                _set_grad(m, True)

    if current_epoch < warmup:
        enable(new_modules)
        return "phase_0__new_modules_only"

    if current_epoch < full_start:
        enable(new_modules, [model.stage_late], [model.stage_mid])
        return "phase_1__+mid+late_backbone"

    # Full unfreeze — all backbone stages active
    enable(new_modules, [model.stage_late], [model.stage_mid], [model.stage_early])
    return "phase_2__full_model"

def freeze_batchnorm(model: LandslideEfficientNet) -> None:
    """
    Set all BN layers to eval mode and freeze their affine parameters.
 
    Use case: small target datasets where ImageNet BN running statistics
    are more reliable than those estimated from the small target set.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            _set_grad(m, False)
 
def unfreeze_batchnorm(model: LandslideEfficientNet) -> None:
    """Reverse of freeze_batchnorm."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()
            _set_grad(m, True)

def build_optimizer_v1(model: "LandslideEfficientNet"):
    """9 param groups for LandslideEfficientNet (B3 + FPN)."""
    return torch.optim.AdamW([
        {"params": model.f0_3.parameters(),       "lr": 1e-6, "weight_decay": 1e-4},
        {"params": model.f4_5.parameters(),       "lr": 1e-6, "weight_decay": 1e-4},
        {"params": model.f6_7.parameters(),       "lr": 5e-6, "weight_decay": 1e-4},
        {"params": model.f8.parameters(),         "lr": 5e-6, "weight_decay": 1e-4},
        {"params": model.edge_attn.parameters(),  "lr": 1e-4, "weight_decay": 5e-3},
        {"params": model.scam.parameters(),       "lr": 1e-4, "weight_decay": 5e-3},
        {"params": model.cbam3d.parameters(),     "lr": 1e-4, "weight_decay": 5e-3},
        {"params": model.fpn.parameters(),        "lr": 1e-4, "weight_decay": 5e-3},
        {"params": model.head.parameters(),       "lr": 1e-4, "weight_decay": 1e-3},
    ], weight_decay=1e-2)

def build_optimizer_v2(model: "LandslideEfficientNet2"):
    """4 param groups for LandslideEfficientNet2 (V2-M simple)."""
    feats = list(model.features.children())
    early, mid, late = feats[:2], feats[2:5], feats[5:]
    def params(modules): return [p for m in modules for p in m.parameters()]
    return torch.optim.AdamW([
        {"params": params(early),                                 "lr": 1e-6, "weight_decay": 1e-5},
        {"params": params(mid),                                   "lr": 5e-6, "weight_decay": 1e-4},
        {"params": params(late),                                  "lr": 2e-5, "weight_decay": 1e-4},
        {"params": list(model.channel_attn.parameters())
                 + [model.gem_p]
                 + list(model.head.parameters()),                 "lr": 1e-4, "weight_decay": 1e-3},
    ])

# def build_optimizer_v3(model: "LandslideEfficientNet3"):
#     """
#     5 param groups for LandslideEfficientNet3 (MSHAN).
#     Multi-timescale update: earlier stages → slower LR.
#     """
#     return torch.optim.AdamW([
#         # Early backbone — near-frozen (ImageNet texture filters, domain-invariant)
#         {"params": list(model.stage_early.parameters())
#                  + list(model.se_early.parameters()),             "lr": 1e-6,  "weight_decay": 1e-5},
#         # Mid backbone — cautious update
#         {"params": list(model.stage_mid.parameters())
#                  + list(model.se_mid.parameters()),               "lr": 5e-6,  "weight_decay": 1e-4},
#         # Late backbone — fine-tune top layers
#         {"params": list(model.stage_late.parameters())
#                  + list(model.se_late.parameters()),              "lr": 2e-5,  "weight_decay": 1e-4},
#         # Projections + gate — new modules, moderate LR
#         {"params": list(model.proj_early.parameters())
#                  + list(model.proj_mid.parameters())
#                  + list(model.proj_late.parameters())
#                  + list(model.cross_gate.parameters())
#                  + [model.gem_p],                                 "lr": 5e-4,  "weight_decay": 5e-3},
#         # Head — highest LR, small WD
#         {"params": list(model.head.parameters()),                 "lr": 1e-3,  "weight_decay": 1e-3},
#     ])

def build_optimizer_v3(model: LandslideEfficientNet3):
    """
    Match plain EfficientNet's LR scale — backbone at 1e-4, not 1e-6.
    The 300× LR gap was the main reason custom models couldn't catch up.
    Layer-wise decay: early=0.1x, mid=0.3x, late=0.6x, new=1.0x
    """
    return torch.optim.AdamW([
        {
            "params": list(model.stage_early.parameters())
                    + list(model.se_early.parameters()),
            "lr": 1e-5,   # was 1e-6 — 10× increase
            "weight_decay": 1e-4,
        },
        {
            "params": list(model.stage_mid.parameters())
                    + list(model.se_mid.parameters()),
            "lr": 5e-5,   # was 5e-6 — 10× increase
            "weight_decay": 1e-4,
        },
        {
            "params": list(model.stage_late.parameters())
                    + list(model.se_late.parameters()),
            "lr": 1e-4,   # was 2e-5 — 5× increase
            "weight_decay": 1e-4,
        },
        {
            "params": list(model.proj_early.parameters())
                    + list(model.proj_mid.parameters())
                    + list(model.proj_late.parameters())
                    + list(model.cross_gate.parameters())
                    + [model.gem_p],
            "lr": 3e-4,   # new modules — same as plain EfficientNet's single LR
            "weight_decay": 5e-3,
        },
        {
            "params": list(model.head.parameters()),
            "lr": 3e-4,
            "weight_decay": 1e-3,
        },
    ])

def build_optimizer_param_groups(
    model         : LandslideEfficientNet,
    base_lr       : float = 3e-4,
    head_lr       : float = 1e-3,
    fpn_lr        : float = 5e-4,
    attn_lr       : float = 5e-4,
    backbone_decay: float = 0.1,
) -> List[Dict]:
    """
    Parameter groups with differentiated learning rates.
 
    Rate schedule (lowest → highest):
        f0_3  early backbone   base_lr * decay^2
        f4_5  mid backbone     base_lr * decay^1
        f6_7 + f8  late        base_lr * decay^0  (= base_lr)
        attention modules      attn_lr
        FPN                    fpn_lr
        head                   head_lr   (highest)
 
    backbone_decay controls how aggressively earlier layers are slowed.
    A value of 0.1 makes early layers train at 1% of base_lr (typical for
    a well-pretrained ImageNet backbone on a small target dataset).
 
    Usage:
        groups    = build_optimizer_param_groups(model, base_lr=3e-4)
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    """
    attn_params = (
        list(model.edge_attn.parameters())
        + list(model.scam.parameters())
        + list(model.cbam3d.parameters())
    )
 
    groups = [
        {"name": "backbone_early_f0_3",
         "params": list(model.f0_3.parameters()),
         "lr": base_lr * backbone_decay ** 2},
        {"name": "backbone_mid_f4_5",
         "params": list(model.f4_5.parameters()),
         "lr": base_lr * backbone_decay},
        {"name": "backbone_late_f6_7_f8",
         "params": list(model.f6_7.parameters()) + list(model.f8.parameters()),
         "lr": base_lr},
        {"name": "attention_modules",
         "params": attn_params,
         "lr": attn_lr},
        {"name": "fpn",
         "params": list(model.fpn.parameters()),
         "lr": fpn_lr},
        {"name": "head",
         "params": list(model.head.parameters()),
         "lr": head_lr},
    ]
    return [g for g in groups if len(g["params"]) > 0]

def compute_class_weights(y_array, indices, num_classes=3):
    labels = y_array[indices]   # single vectorized memmap read
    counts = Counter(labels.tolist())
    total  = sum(counts.values())
    weights = torch.tensor([
        total / (num_classes * counts[c]) for c in range(num_classes)
    ], dtype=torch.float32)
    print("Class counts:", counts)
    print("Class weights:", weights)
    return weights

class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples so the model focuses on hard ones.
    alpha : per-class weights (class imbalance correction)
    gamma : focusing parameter — 2.0 is standard
    """
    def __init__(
        self,
        alpha      : torch.Tensor,
        gamma      : float = 2.0,
        smoothing  : float = 0.1,
        reduction  : str   = "mean",
    ) -> None:
        super().__init__()
        self.alpha     = alpha        # (num_classes,)
        self.gamma     = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)

        # Safety clamp — prevents -inf in log_softmax for overconfident predictions
        logits = torch.clamp(logits, -50.0, 50.0)

        with torch.no_grad():
            smooth_targets = torch.full_like(
                logits, self.smoothing / (num_classes - 1)
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Use softmax separately to avoid 0 * -inf = NaN
        log_prob = F.log_softmax(logits, dim=1).clamp(min=-100.0)  # clamp -inf
        prob     = F.softmax(logits, dim=1)                         # NOT log_prob.exp()

        pt      = prob.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7)
        focal_w = (1.0 - pt) ** self.gamma

        alpha_t = self.alpha.to(logits.device)[targets]

        # Now smooth_targets * log_prob is safe (log_prob >= -100, never -inf)
        loss = -(smooth_targets * log_prob).sum(dim=1)
        loss = alpha_t * focal_w * loss

        # Final NaN guard
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        return loss.mean() if self.reduction == "mean" else loss.sum()

class EarlyStopper:
    def __init__(self, patience=5, min_delta=1e-4):
        self.best_loss = np.inf
        self.patience = patience
        self.counter = 0
        self.min_delta = min_delta
    
    
    def should_stop(self, loss):
        if loss + self.min_delta < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class MemmapImageDataset(Dataset):
    def __init__(self, X, y, meta_path):
        """
        X: np.memmap of images (N, 3, H, W)
        y: np.memmap or np.ndarray of labels (N,)
        meta_path: path to meta.npy
        """
        self.X = X
        self.y = y

        # ---- Load metadata ----
        meta = np.load(meta_path, allow_pickle=True).item()
        class_map = meta["class_map"]

        # Ensure correct index order
        self.classes = [None] * len(class_map)
        for class_name, idx in class_map.items():
            self.classes[idx] = class_name

        self.class_to_idx = class_map

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # clone() is REQUIRED for memmap safety
        x = torch.from_numpy(self.X[idx].copy()).float()
        y = int(self.y[idx])
        return x, y

meta = np.load(f"{DATASET_DIR}/meta_train.npy", allow_pickle=True).item()

N_SAMPLES = meta["shape"][0]          # total samples
IMG_SHAPE = meta["shape"][1:]          # (3,224,224)

meta_test = np.load(f"{DATASET_DIR}/meta_test.npy", allow_pickle=True).item()

N_SAMPLES_T = meta_test["shape"][0]          # total samples
IMG_SHAPE_T = meta_test["shape"][1:]          # (3,224,224)

X = np.memmap(
    f"{DATASET_DIR}/X_train.dat",
    dtype="float16",
    mode="r",
    shape=meta["shape"]
)

y = np.memmap(
    f"{DATASET_DIR}/y_train.dat",
    dtype="int64",
    mode="r",
    shape=(N_SAMPLES,)
)

X_test = np.memmap(
    f"{DATASET_DIR}/X_test.dat",
    dtype="float16",
    mode="r",
    shape=meta_test["shape"]
)

y_test = np.memmap(
    f"{DATASET_DIR}/y_test.dat",
    dtype="int64",
    mode="r",
    shape=(N_SAMPLES_T,)
)

full_dataset = MemmapImageDataset(
    X=X,
    y=y,
    meta_path=f"{DATASET_DIR}/meta_train.npy"
)

test_dataset = MemmapImageDataset(
    X=X_test,
    y=y_test,
    meta_path=f"{DATASET_DIR}/meta_test.npy"
)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

classes = full_dataset.classes
print(classes)

VAL_RATIO = 0.2  # 80 / 20 split

indices = np.arange(len(full_dataset))

train_idx, val_idx = train_test_split(
    indices,
    test_size=VAL_RATIO,
    random_state=SEED,
    shuffle=True,
    stratify=y          # IMPORTANT: keep class balance
)

train_dataset = Subset(full_dataset, train_idx)
val_dataset   = Subset(full_dataset, val_idx)

def tta_predict(model, x):
    model.eval()  # ensure BN/dropout stable
    preds = []

    with torch.no_grad():
        for flip in [False, True]:
            x_aug = torch.flip(x, dims=[3]) if flip else x

            for scale in [1.0, 0.9]:
                if scale != 1.0:
                    size = int(224 * scale)
                    x_scaled = F.interpolate(x_aug, size=size, mode="bilinear", align_corners=False)
                    x_scaled = F.interpolate(x_scaled, size=224, mode="bilinear", align_corners=False)
                else:
                    x_scaled = x_aug

                preds.append(model(x_scaled))

    return torch.stack(preds).mean(0)


# def hero_train(
#     model_name,
#     train_loader,
#     val_loader,
#     device,
#     epochs,
#     use_amp=True
# ):
#     print(f"\n===== Training {model_name} =====")

#     model = build_model(model_name).to(device)

#     if torch.cuda.device_count() > 1:
#         print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
#         model = nn.DataParallel(model)

#     # ================= OPTIMIZER =================
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=2e-4,              # 🔥 slightly lower (more stable)
#         weight_decay=5e-5     # 🔥 less regularization
#     )


#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#         optimizer,
#         T_0=5,
#         T_mult=2,
#         eta_min=1e-6
#     )


#     # ================= LOSS =================
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.08)

#     scaler = torch.amp.GradScaler(enabled=(use_amp and device == "cuda"))

#     early_stop = EarlyStopper(patience=10)

#     best_state = None
#     best_val_loss = float("inf")
#     best_val_acc = 0

#     history = {
#         "train_loss": [],
#         "val_loss": [],
#         "train_acc": [],
#         "val_acc": []
#     }

#     # ================= TRAIN LOOP =================
#     for epoch in range(epochs):

#         model.train()

#         total_loss, correct, total = 0, 0, 0

#         for x, y in tqdm(train_loader, desc=f"Train E{epoch+1}", leave=False):

#             x = x.to(device)
#             y = y.to(device)

#             # ===== LIGHT MIXUP (early only) =====
#             # use_mixup = epoch < 5 and random.random() < 0.2
#             use_mixup = False

#             if use_mixup:
#                 x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
#             else:
#                 y_a, y_b, lam = y, y, 1.0

#             optimizer.zero_grad(set_to_none=True)

#             # AMP after warmup
#             use_amp_epoch = use_amp and device == "cuda" and epoch >= 3

#             with autocast(device_type="cuda", enabled=use_amp_epoch):
#                 logits = model(x)

#                 # 🔥 HARD SAFETY (never NaN)
#                 logits = torch.nan_to_num(
#                     logits,
#                     nan=0.0,
#                     posinf=10.0,
#                     neginf=-10.0
#                 )

#                 loss = (
#                     mixup_criterion(criterion, logits, y_a, y_b, lam)
#                     if use_mixup else criterion(logits, y)
#                 )

#             if not torch.isfinite(loss):
#                 continue

#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)

#             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

#             scaler.step(optimizer)
#             scaler.update()

#             total_loss += loss.item() * x.size(0)

#             if not use_mixup:
#                 preds = logits.argmax(dim=1)
#                 correct += (preds == y).sum().item()
#                 total += y.size(0)

#         train_loss = total_loss / max(total, 1)
#         train_acc = correct / max(total, 1)

#         # ================= VALIDATION =================
#         model.eval()

#         val_loss, v_correct, v_total = 0, 0, 0

#         with torch.no_grad():
#             for x, y in tqdm(val_loader, desc=f"Val E{epoch+1}", leave=False):

#                 x = x.to(device)
#                 y = y.to(device)

#                 # logits = model(x)
#                 logits = tta_predict(model, x)
#                 logits = logits / 0.9

#                 loss = criterion(logits, y)

#                 val_loss += loss.item() * x.size(0)

#                 preds = logits.argmax(dim=1)
#                 v_correct += (preds == y).sum().item()
#                 v_total += y.size(0)

#         val_loss /= max(v_total, 1)
#         val_acc = v_correct / max(v_total, 1)

#         history["train_loss"].append(train_loss)
#         history["val_loss"].append(val_loss)
#         history["train_acc"].append(train_acc)
#         history["val_acc"].append(val_acc)

#         print(
#             f"[{model_name}] Epoch {epoch+1} | "
#             f"Train {train_loss:.4f} | Val {val_loss:.4f} | Acc {val_acc:.4f}"
#         )

#         # ===== SAVE BEST =====
#         if val_loss < best_val_loss or val_acc > best_val_acc:
#             best_val_loss = val_loss
#             best_val_acc = val_acc
#             best_state = {
#                 k: v.cpu() for k, v in model.state_dict().items()
#             }

#         scheduler.step()

#         if early_stop.should_stop(val_loss):
#             print(">>> Early stopping triggered")
#             break

#     return best_state, best_val_loss, best_val_acc, history

# def hero_train(model_name, train_loader, val_loader, device, epochs,
#                use_amp=True, accum_steps=2):
#     print(f"\n===== Training {model_name} =====")
 
#     model = build_model(model_name).to(device)
#     class_weights = compute_class_weights(y, train_idx, NUM_CLASSES).to(device)
 
#     if torch.cuda.device_count() > 1:
#         print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
#         model = nn.DataParallel(model)
 
#     # ── OPTIMIZER ─────────────────────────────────────────────────────────
#     # ← CHANGED: LandslideEfficientNet uses differential LRs per backbone
#     #   depth; every other model keeps the original flat AdamW.
#     base_model = model.module if isinstance(model, nn.DataParallel) else model
#     if isinstance(base_model, LandslideEfficientNet):
#         # param_groups = build_optimizer_param_groups(
#         #     base_model,
#         #     base_lr       = 3e-4,   # late backbone / new modules
#         #     head_lr       = 1e-3,   # classification head
#         #     fpn_lr        = 5e-4,   # FPN
#         #     attn_lr       = 5e-4,   # edge_attn / scam / cbam3d
#         #     backbone_decay= 0.1,    # early layers: 3e-4 * 0.01
#         # )
#         # optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
#         optimizer = build_optimizer(base_model, base_lr=1e-4)
#     else:
#         optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
 
#     # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     #     optimizer, T_0=5, T_mult=2, eta_min=1e-6
#     # )
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#         optimizer,
#         max_lr          = [1e-6, 1e-6, 5e-6, 5e-6,
#                         1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
#         steps_per_epoch = len(train_loader),
#         epochs          = EPOCHS,
#         pct_start       = 0.05,    # shorter warmup now that model is partly trained
#         anneal_strategy = "cos",
#         final_div_factor= 1e4,     # end at very low LR for fine convergence
#     )
#     # criterion  = nn.CrossEntropyLoss(label_smoothing=0.0)
#     criterion = FocalLoss(
#         alpha    = class_weights,
#         gamma    = 2.0,
#         smoothing= 0.1,
#     )
#     scaler     = GradScaler(enabled=(use_amp and device == "cuda"))
#     early_stop = EarlyStopper(patience=20)
 
#     ema_model = deepcopy(base_model).to(device)
#     ema_model.eval()
#     ema_decay = 1.0 - (1.0 - 0.999) * max(0.0, (10 - epoch) / 10)
 
#     def update_ema(model, ema_model):
#         with torch.no_grad():
#             msd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
#             for k, ema_v in ema_model.state_dict().items():
#                 ema_v.copy_(ema_v * ema_decay + msd[k] * (1.0 - ema_decay))
 
#     best_state, best_val_loss, best_val_acc = None, float("inf"), 0
#     history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
#     global_step = 0
 
#     for epoch in range(epochs):
 
#         # ── GRADUAL UNFREEZE ───────────────────────────────────────────────
#         # ← CHANGED: only applied to LandslideEfficientNet; other models are
#         #   unaffected.  Thresholds are tuned to EPOCHS=35:
#         #     ph0  0–4   head only        (let output layer stabilise first)
#         #     ph1  5–9   + late backbone  (CBAM3D, FPN)
#         #     ph2 10–14  + mid backbone   (SCAM+, EdgeAttn)
#         #     ph3 15–19  + early backbone (stem frozen)
#         #     ph4 20+    full model
#         if isinstance(base_model, LandslideEfficientNet):
#             phase = gradual_unfreeze(
#                 base_model, epoch,
#                 warmup=5, phase1_end=10, phase2_end=15, phase3_end=20
#             )
#             # zero_lr_for_frozen(optimizer, base_model)
#             print(f"  [unfreeze] epoch={epoch}  {phase}")
 
#         model.train()
#         total_loss, correct, total = 0, 0, 0
#         optimizer.zero_grad(set_to_none=True)
 
#         for step, (x, y_batch) in enumerate(tqdm(train_loader, desc=f"Train E{epoch+1}", leave=False)):
#             x = x.to(device)
#             y_batch = y_batch.to(device)
 
#             use_mixup = epoch < int(epochs * 0.4) and random.random() < 0.5
#             if use_mixup:
#                 x, y_a, y_b, lam = mixup_data(x, y_batch, alpha=0.2)
#             else:
#                 y_a, y_b, lam = y_batch, y_batch, 1.0
 
#             with autocast(device_type="cuda", enabled=(use_amp and device == "cuda")):
#                 outputs = model(x)
#                 if isinstance(outputs, tuple):
#                     logits, aux_loss = outputs
#                 else:
#                     logits, aux_loss = outputs, None
 
#                 logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
#                 ce_loss = (mixup_criterion(criterion, logits, y_a, y_b, lam)
#                            if use_mixup else criterion(logits, y_batch))
#                 loss = ce_loss
#                 if aux_loss is not None:
#                     loss = loss + 0.01 * (1 - epoch / epochs) * aux_loss
#                 loss = loss / accum_steps
 
#             if not torch.isfinite(loss):
#                 continue
 
#             scaler.scale(loss).backward()
 
#             if (step + 1) % accum_steps == 0:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad(set_to_none=True)
#                 update_ema(model, ema_model)
#                 scheduler.step()
 
#             # scheduler.step(epoch + step / len(train_loader))
#             total_loss += loss.item() * x.size(0) * accum_steps
 
#             if not use_mixup:
#                 correct += (logits.argmax(dim=1) == y_batch).sum().item()
#                 total   += y_batch.size(0)
 
#             global_step += 1
 
#         train_loss = total_loss / max(total, 1)
#         train_acc  = correct   / max(total, 1)
 
#         ema_model.eval()
#         val_loss, v_correct, v_total = 0, 0, 0
 
#         with torch.no_grad():
#             for x, y_batch in tqdm(val_loader, ...):
#                 x, y_batch = x.to(device), y_batch.to(device)
#                 outputs = ema_model(x)
#                 logits  = outputs[0] if isinstance(outputs, tuple) else outputs
#                 logits  = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)  # ADD THIS
#                 v_loss  = criterion(logits, y_batch).item()
#                 if not math.isfinite(v_loss):      # ADD THIS guard
#                     continue
#                 val_loss  += v_loss * x.size(0)
#                 v_correct += (logits.argmax(dim=1) == y_batch).sum().item()
#                 v_total   += y_batch.size(0)
 
#         val_loss /= max(v_total, 1)
#         val_acc   = v_correct / max(v_total, 1)
 
#         history["train_loss"].append(train_loss)
#         history["val_loss"].append(val_loss)
#         history["train_acc"].append(train_acc)
#         history["val_acc"].append(val_acc)
 
#         print(f"[{model_name}] Epoch {epoch+1} | "
#               f"Train {train_loss:.4f} | Val {val_loss:.4f} | Acc {val_acc:.4f}")
 
#         if val_loss < best_val_loss or val_acc > best_val_acc:
#             best_val_loss = val_loss
#             best_val_acc  = val_acc
#             best_state    = {k: v.cpu() for k, v in ema_model.state_dict().items()}
 
#         if early_stop.should_stop(val_loss):
#             print(">>> Early stopping triggered")
#             break
 
#     return best_state, best_val_loss, best_val_acc, history

# def hero_train(model_name, train_loader, val_loader, device, epochs, 
#                use_amp=True, accum_steps=2):
#     print(f"\n===== Training {model_name} =====")

#     model = build_model(model_name).to(device)
#     class_weights = compute_class_weights(y, train_idx, NUM_CLASSES).to(device)

#     if torch.cuda.device_count() > 1:
#         print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
#         model = nn.DataParallel(model)

#     base_model = model.module if isinstance(model, nn.DataParallel) else model

#     if isinstance(base_model, LandslideEfficientNet):
#         optimizer = build_optimizer(base_model)
#         max_lr = [1e-6, 1e-6, 5e-6, 5e-6, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
#     else:
#         optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
#         max_lr = 3e-4

#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#         optimizer,
#         max_lr          = max_lr,
#         steps_per_epoch = len(train_loader),
#         epochs          = epochs,
#         pct_start       = 0.05,
#         anneal_strategy = "cos",
#         final_div_factor= 1e4,
#     )

#     class AdaptiveFocalLoss(nn.Module):
#         def __init__(self, alpha, smoothing=0.1):
#             super().__init__()
#             self.alpha    = alpha
#             self.smoothing = smoothing
#             self.gamma    = 2.0

#         def forward(self, logits, targets):
#             num_classes = logits.size(1)
#             logits = torch.clamp(logits, -50.0, 50.0)
#             with torch.no_grad():
#                 smooth_targets = torch.full_like(logits, self.smoothing / (num_classes - 1))
#                 smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
#             log_prob = F.log_softmax(logits, dim=1).clamp(min=-100.0)
#             prob     = F.softmax(logits, dim=1)
#             pt       = prob.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7)
#             focal_w  = (1.0 - pt) ** self.gamma
#             alpha_t  = self.alpha.to(logits.device)[targets]
#             loss     = -(smooth_targets * log_prob).sum(dim=1)
#             loss     = alpha_t * focal_w * loss
#             loss     = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
#             return loss.mean()

#     criterion = AdaptiveFocalLoss(alpha=class_weights, smoothing=0.1)
#     scaler    = GradScaler(enabled=(use_amp and device == "cuda"))

#     es_best_acc, es_counter, es_patience = 0.0, 0, 20

#     ema_model = deepcopy(base_model).to(device)
#     ema_model.eval()

#     def update_ema(model, ema_model, epoch):
#         # decay ramps from 0.990 (epoch 0) → 0.999 (epoch 10+)
#         decay = 0.999 - (0.999 - 0.990) * max(0.0, (10 - epoch) / 10)
#         with torch.no_grad():
#             msd = model.module.state_dict() if isinstance(model, nn.DataParallel) \
#                 else model.state_dict()
#             for k, ema_v in ema_model.state_dict().items():
#                 model_v = msd[k]
#                 # FIX: skip non-float buffers (num_batches_tracked is int64).
#                 # EMA-ing an integer tensor produces float → cast back to int
#                 # which silently corrupts BN state over many epochs.
#                 if not ema_v.is_floating_point():
#                     ema_v.copy_(model_v)   # just copy directly, no EMA
#                 else:
#                     ema_v.copy_(ema_v * decay + model_v * (1.0 - decay))

#     def set_head_dropout(model, p):
#         for m in model.head.mlp.modules():
#             if isinstance(m, nn.Dropout):
#                 m.p = p

#     best_state, best_val_loss, best_val_acc = None, float("inf"), 0.0
#     history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
#     global_step = 0

#     for epoch in range(epochs):

#         # ── REPLACE the gamma schedule (3 lines) ──────────────────────────────
#         # BEFORE — gamma=0.5 causes gradient singularity when pt → 1
#         if epoch < 20:
#             criterion.gamma = 2.0
#         elif epoch < 40:
#             criterion.gamma = 1.0
#         else:
#             criterion.gamma = 0.5   # ← CAUSES EXPLOSION

#         # AFTER — minimum gamma=1.0 ensures d/dpt[(1-pt)^gamma] → 0 as pt → 1
#         if epoch < 20:
#             criterion.gamma = 2.0
#         elif epoch < 40:
#             criterion.gamma = 1.5
#         else:
#             criterion.gamma = 1.0   # safe floor: gradient vanishes at high confidence

#         if isinstance(base_model, LandslideEfficientNet):
#             phase = gradual_unfreeze(
#                 base_model, epoch,
#                 warmup=5, phase1_end=10, phase2_end=15, phase3_end=20
#             )
#             print(f"  [unfreeze] epoch={epoch}  {phase}  gamma={criterion.gamma:.1f}")

#             if epoch < 20:
#                 set_head_dropout(base_model, 0.3)
#             else:
#                 set_head_dropout(base_model, 0.1)

#         # Ensure training model is in train mode — called AFTER all eval from
#         # previous epoch and BEFORE any training this epoch
#         model.train()
#         total_loss, correct, total = 0, 0, 0
#         optimizer.zero_grad(set_to_none=True)

#         for step, (x, y_batch) in enumerate(tqdm(train_loader, desc=f"Train E{epoch+1}", leave=False)):
#             x       = x.to(device)
#             y_batch = y_batch.to(device)

#             use_mixup = epoch < int(epochs * 0.2) and random.random() < 0.5
#             if use_mixup:
#                 x, y_a, y_b, lam = mixup_data(x, y_batch, alpha=0.2)
#             else:
#                 y_a, y_b, lam = y_batch, y_batch, 1.0

#             with autocast(device_type="cuda", enabled=(use_amp and device == "cuda")):
#                 outputs  = model(x)
#                 logits   = outputs[0] if isinstance(outputs, tuple) else outputs
#                 aux_loss = outputs[1] if isinstance(outputs, tuple) else None

#                 logits  = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
#                 ce_loss = (mixup_criterion(criterion, logits, y_a, y_b, lam)
#                            if use_mixup else criterion(logits, y_batch))
#                 loss = ce_loss
#                 if aux_loss is not None:
#                     loss = loss + 0.01 * (1 - epoch / epochs) * aux_loss
#                 loss = loss / accum_steps

#             if not torch.isfinite(loss):
#                 continue

#             scaler.scale(loss).backward()

#             if (step + 1) % accum_steps == 0:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad(set_to_none=True)
#                 update_ema(model, ema_model, epoch)
#                 scheduler.step()

#             total_loss += loss.item() * x.size(0) * accum_steps

#             if not use_mixup:
#                 correct += (logits.argmax(dim=1) == y_batch).sum().item()
#                 total   += y_batch.size(0)

#             global_step += 1

#         train_loss = total_loss / max(total, 1)
#         train_acc  = correct   / max(total, 1)

#         # ── VALIDATION: only evaluate ema_model, single pass ──────────
#         # KEY FIX: never call eval() on base_model/model during training —
#         # it corrupts BN running stats and causes NaN outputs next epoch.
#         # ema_model is a separate copy so it's safe to eval at any time.
#         ema_model.eval()
#         val_loss_sum, v_correct_fast, v_correct_tta, v_total = 0.0, 0, 0, 0

#         with torch.no_grad():
#             for x, y_batch in tqdm(val_loader, desc=f"Val E{epoch+1}", leave=False):
#                 x, y_batch = x.to(device), y_batch.to(device)

#                 # Fast forward for loss and early stopping
#                 logits_fast = ema_model(x)
#                 logits_fast = torch.nan_to_num(logits_fast, nan=0.0, posinf=50.0, neginf=-50.0)

#                 v_loss = criterion(logits_fast, y_batch).item()
#                 if math.isfinite(v_loss):
#                     val_loss_sum += v_loss * x.size(0)

#                 v_correct_fast += (logits_fast.argmax(dim=1) == y_batch).sum().item()

#                 # TTA for the reported accuracy (same ema_model, still in eval mode)
#                 logits_tta = tta_predict(ema_model, x)
#                 logits_tta = torch.nan_to_num(logits_tta, nan=0.0, posinf=50.0, neginf=-50.0)
#                 v_correct_tta += (logits_tta.argmax(dim=1) == y_batch).sum().item()

#                 v_total += y_batch.size(0)

#         val_loss     = val_loss_sum / max(v_total, 1)
#         val_acc_fast = v_correct_fast / max(v_total, 1)   # for early stopping
#         val_acc_tta  = v_correct_tta  / max(v_total, 1)   # reported metric

#         history["train_loss"].append(train_loss)
#         history["val_loss"].append(val_loss)
#         history["train_acc"].append(train_acc)
#         history["val_acc"].append(val_acc_tta)

#         print(f"[{model_name}] Epoch {epoch+1} | "
#               f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
#               f"Val loss={val_loss:.4f} acc(TTA)={val_acc_tta:.4f} acc(fast)={val_acc_fast:.4f}")

#         # Save on fast accuracy (stable, not TTA noise)
#         if val_acc_fast > best_val_acc:
#             best_val_loss = val_loss
#             best_val_acc  = val_acc_fast
#             best_state    = {k: v.cpu() for k, v in ema_model.state_dict().items()}
#             print(f"  >>> New best: acc={best_val_acc:.4f}")
            
#             save_path = f"{MODEL_DIR}/{model_name}_best.pth"
#             torch.save(clean_state_dict(best_state), save_path)
#             print(f">>> Model saved to {save_path}")

#         # Early stopping on fast accuracy
#         if val_acc_fast > es_best_acc + 1e-4:
#             es_best_acc = val_acc_fast
#             es_counter  = 0
#         else:
#             es_counter += 1
#             print(f"  [early_stop] {es_counter}/{es_patience} (best={es_best_acc:.4f})")
#             if es_counter >= es_patience:
#                 print(">>> Early stopping triggered")
#                 break

#     return best_state, best_val_loss, best_val_acc, history


def hero_train(model_name, train_loader, val_loader, device, epochs,
               use_amp=True, accum_steps=3):
    print(f"\n===== Training {model_name} =====")

    model = build_model(model_name).to(device)
    class_weights = compute_class_weights(y, train_idx, NUM_CLASSES).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    base_model = model.module if isinstance(model, nn.DataParallel) else model

    if isinstance(base_model, LandslideEfficientNet) and not isinstance(base_model, (LandslideEfficientNet2, LandslideEfficientNet3)):
        optimizer = build_optimizer_v1(base_model)
        max_lr = [1e-6, 1e-6, 5e-6, 5e-6, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
    elif isinstance(base_model, LandslideEfficientNet2):
        optimizer = build_optimizer_v2(base_model)
        max_lr = [1e-6, 5e-6, 2e-5, 1e-4]
    # elif isinstance(base_model, LandslideEfficientNet3):
    #     optimizer = build_optimizer_v3(base_model)
    #     max_lr = [1e-6, 5e-6, 2e-5, 5e-4, 1e-3]   # 5 groups
    elif isinstance(base_model, LandslideEfficientNet3):
        optimizer = build_optimizer_v3(base_model)
        max_lr = [1e-5, 5e-5, 1e-4, 3e-4, 3e-4]   # 5 groups, peak matches plain EfficientNet
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        max_lr = 3e-4

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = max_lr,
        steps_per_epoch = len(train_loader),
        epochs          = epochs,
        pct_start       = 0.05,
        anneal_strategy = "cos",
        final_div_factor= 1e4,
    )

    class AdaptiveFocalLoss(nn.Module):
        def __init__(self, alpha, smoothing=0.15):
            super().__init__()
            self.alpha     = alpha
            self.smoothing = smoothing
            self.gamma     = 2.0

        def forward(self, logits, targets):
            num_classes = logits.size(1)
            logits = torch.clamp(logits, -50.0, 50.0)
            with torch.no_grad():
                smooth_targets = torch.full_like(logits, self.smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            log_prob = F.log_softmax(logits, dim=1).clamp(min=-100.0)
            prob     = F.softmax(logits, dim=1)
            pt       = prob.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7)
            focal_w  = (1.0 - pt) ** self.gamma
            alpha_t  = self.alpha.to(logits.device)[targets]
            loss     = -(smooth_targets * log_prob).sum(dim=1)
            loss     = alpha_t * focal_w * loss
            loss     = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
            return loss.mean()

    criterion = AdaptiveFocalLoss(alpha=class_weights, smoothing=0.15)
    scaler    = GradScaler(enabled=(use_amp and device == "cuda"))

    es_best_acc, es_counter, es_patience = 0.0, 0, 5

    ema_model = deepcopy(base_model).to(device)
    ema_model.eval()

    # SWA — no mid-training eval (BN stats would be wrong without update_bn)
    # Evaluated properly with update_bn only at the end of training
    swa_model  = AveragedModel(base_model)
    swa_start  = max(epochs - 15, epochs // 2)
    swa_active = False

    def update_ema(model, ema_model, epoch):
        decay = 0.999 - (0.999 - 0.990) * max(0.0, (10 - epoch) / 10)
        with torch.no_grad():
            msd = model.module.state_dict() if isinstance(model, nn.DataParallel) \
                  else model.state_dict()
            for k, ema_v in ema_model.state_dict().items():
                model_v = msd[k]
                if not ema_v.is_floating_point():
                    ema_v.copy_(model_v)
                else:
                    ema_v.copy_(ema_v * decay + model_v * (1.0 - decay))

    def set_head_dropout(model, p):
    # head MLP is model.head, not model.head.mlp
        for m in model.head.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

    def rdrop_kl(logits1, logits2):
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        kl1 = F.kl_div(F.log_softmax(logits1, dim=1), p2.detach(), reduction='batchmean')
        kl2 = F.kl_div(F.log_softmax(logits2, dim=1), p1.detach(), reduction='batchmean')
        return (kl1 + kl2) / 2.0

    best_state, best_val_loss, best_val_acc = None, float("inf"), 0.0
    save_path = f"{MODEL_DIR}/{model_name}_best.pth"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    global_step = 0

    for epoch in range(epochs):

        if epoch < 20:
            criterion.gamma = 2.0
        elif epoch < 40:
            criterion.gamma = 1.5
        else:
            criterion.gamma = 1.0

        rdrop_alpha = 0.0 if epoch < 10 else min(0.5, (epoch - 10) / 30 * 0.5)

        if isinstance(base_model, LandslideEfficientNet3):
            phase = gradual_unfreeze_v3(base_model, epoch)
            print(f"  [unfreeze] epoch={epoch}  {phase}  "
                  f"gamma={criterion.gamma:.1f}  rdrop={rdrop_alpha:.2f}")
            set_head_dropout(base_model, 0.3 if epoch < 20 else 0.15)

        elif isinstance(base_model, LandslideEfficientNet2):
            phase = gradual_unfreeze_v2(base_model, epoch)
            print(f"  [unfreeze] epoch={epoch}  {phase}  "
                  f"gamma={criterion.gamma:.1f}  rdrop={rdrop_alpha:.2f}")
            set_head_dropout(base_model, 0.3 if epoch < 20 else 0.15)

        elif isinstance(base_model, LandslideEfficientNet):
            phase = gradual_unfreeze_v1(
                base_model, epoch,
                warmup=5, phase1_end=10, phase2_end=15, phase3_end=20
            )
            print(f"  [unfreeze] epoch={epoch}  {phase}  "
                  f"gamma={criterion.gamma:.1f}  rdrop={rdrop_alpha:.2f}")
            set_head_dropout(base_model, 0.3 if epoch < 20 else 0.15)

        model.train()
        total_loss, correct, total = 0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y_batch) in enumerate(tqdm(train_loader, desc=f"Train E{epoch+1}", leave=False)):
            x       = x.to(device)
            y_batch = y_batch.to(device)

            use_mixup = epoch < int(epochs * 0.2) and random.random() < 0.5
            if use_mixup:
                x, y_a, y_b, lam = mixup_data(x, y_batch, alpha=0.2)
            else:
                y_a, y_b, lam = y_batch, y_batch, 1.0

            with autocast(device_type="cuda", enabled=(use_amp and device == "cuda")):
                outputs1 = model(x)
                logits1  = outputs1[0] if isinstance(outputs1, tuple) else outputs1
                aux_loss = outputs1[1] if isinstance(outputs1, tuple) else None
                logits1  = torch.nan_to_num(logits1, nan=0.0, posinf=10.0, neginf=-10.0)

                ce_loss = (mixup_criterion(criterion, logits1, y_a, y_b, lam)
                           if use_mixup else criterion(logits1, y_batch))
                loss = ce_loss

                if rdrop_alpha > 0 and not use_mixup:
                    outputs2 = model(x)
                    logits2  = outputs2[0] if isinstance(outputs2, tuple) else outputs2
                    logits2  = torch.nan_to_num(logits2, nan=0.0, posinf=10.0, neginf=-10.0)
                    ce_loss2 = criterion(logits2, y_batch)
                    loss = (ce_loss + ce_loss2) / 2.0 + rdrop_alpha * rdrop_kl(logits1, logits2)

                if aux_loss is not None:
                    loss = loss + 0.01 * (1 - epoch / epochs) * aux_loss

                loss = loss / accum_steps

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(model, ema_model, epoch)
                scheduler.step()

                if epoch >= swa_start:
                    swa_model.update_parameters(model)
                    swa_active = True

            total_loss += loss.item() * x.size(0) * accum_steps

            if not use_mixup:
                correct += (logits1.argmax(dim=1) == y_batch).sum().item()
                total   += y_batch.size(0)

            global_step += 1

        train_loss = total_loss / max(total, 1)
        train_acc  = correct   / max(total, 1)

        # ── VALIDATION: EMA only (SWA needs update_bn before it's reliable) ──
        ema_model.eval()
        val_loss_sum, v_correct_fast, v_correct_tta, v_total = 0.0, 0, 0, 0

        with torch.no_grad():
            for x, y_batch in tqdm(val_loader, desc=f"Val E{epoch+1}", leave=False):
                x, y_batch = x.to(device), y_batch.to(device)

                logits_fast = ema_model(x)
                logits_fast = torch.nan_to_num(logits_fast, nan=0.0, posinf=50.0, neginf=-50.0)

                v_loss = criterion(logits_fast, y_batch).item()
                if math.isfinite(v_loss):
                    val_loss_sum += v_loss * x.size(0)

                v_correct_fast += (logits_fast.argmax(dim=1) == y_batch).sum().item()

                logits_tta = tta_predict(ema_model, x)
                logits_tta = torch.nan_to_num(logits_tta, nan=0.0, posinf=50.0, neginf=-50.0)
                v_correct_tta += (logits_tta.argmax(dim=1) == y_batch).sum().item()

                v_total += y_batch.size(0)

        val_loss     = val_loss_sum / max(v_total, 1)
        val_acc_fast = v_correct_fast / max(v_total, 1)
        val_acc_tta  = v_correct_tta  / max(v_total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc_tta)

        print(f"[{model_name}] Epoch {epoch+1} | "
              f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val loss={val_loss:.4f} acc(TTA)={val_acc_tta:.4f} acc(fast)={val_acc_fast:.4f}")

        if val_acc_fast > best_val_acc:
            best_val_loss = val_loss
            best_val_acc  = val_acc_fast
            best_state    = {k: v.cpu() for k, v in ema_model.state_dict().items()}
            torch.save(clean_state_dict(best_state), save_path)
            print(f"  >>> New best: acc={best_val_acc:.4f} — saved to {save_path}")

        if val_acc_fast < 0.5 and best_state is not None:
            print(f"  !!! Collapse detected — restoring best weights")
            ema_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            es_counter = 0
        else:
            if val_acc_fast > es_best_acc + 1e-4:
                es_best_acc = val_acc_fast
                es_counter  = 0
            else:
                es_counter += 1
                print(f"  [early_stop] {es_counter}/{es_patience} (best={es_best_acc:.4f})")
                if es_counter >= es_patience:
                    print(">>> Early stopping triggered")
                    break

    # ── FINAL SWA EVALUATION (proper BN update first) ─────────────────
    if swa_active:
        print(">>> Updating SWA BatchNorm stats (full train pass)...")
        update_bn(train_loader, swa_model, device=device)
        swa_model.eval()
        sv_correct, sv_total = 0, 0
        with torch.no_grad():
            for x, y_batch in val_loader:
                x, y_batch = x.to(device), y_batch.to(device)
                logits_swa = swa_model(x)
                logits_swa = torch.nan_to_num(logits_swa, nan=0.0, posinf=50.0, neginf=-50.0)
                sv_correct += (logits_swa.argmax(dim=1) == y_batch).sum().item()
                sv_total   += y_batch.size(0)
        final_swa_acc = sv_correct / max(sv_total, 1)
        print(f">>> Final SWA acc (after BN update): {final_swa_acc:.4f}  EMA best: {best_val_acc:.4f}")

        if final_swa_acc > best_val_acc:
            best_val_acc = final_swa_acc
            best_state   = {k: v.cpu() for k, v in swa_model.state_dict().items()}
            torch.save(clean_state_dict(best_state), save_path)
            print(f">>> SWA is best — saved to {save_path}")
        else:
            print(f">>> EMA is best — keeping existing checkpoint")

    return best_state, best_val_loss, best_val_acc, history

def hero_train_ensemble(model_name, train_loader, val_loader, device, epochs=40,
                        use_amp=True):
    """
    Stable training loop for ensemble models.
    Separate from hero_train because ensemble models:
      1. Have frozen backbones (no gradual unfreeze needed)
      2. Converge in 30-50 epochs (not 60)
      3. Need different LR scales (1e-3 for new modules, not 1e-4)
      4. Don't benefit from R-Drop (too few trainable params)
    """
    print(f"\n===== Training Ensemble: {model_name} =====")

    model = build_model(model_name).to(device)
    class_weights = compute_class_weights(y, train_idx, NUM_CLASSES).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")

    base_model = model.module if isinstance(model, nn.DataParallel) else model

    # ── Optimizer: model-specific ──────────────────────────────────────
    if isinstance(base_model, StackedEnsemble):
        # Only meta classifier trains (~600 params)
        # High LR is fine — small network, no risk of backbone corruption
        optimizer = torch.optim.AdamW(
            base_model.meta.parameters(),
            lr=5e-3,
            weight_decay=1e-2,
        )
        max_lr    = 5e-3
        clip_norm = 1.0
        patience  = 15

    elif isinstance(base_model, WeightedEnsemble):
        # Temperature + attention + head train; backbone fully frozen
        trainable = (
            list(base_model.temperature.unsqueeze(0))  # single param
            + list(base_model.token_proj.parameters())
            + list(base_model.attn.parameters())
            + list(base_model.token_weight.parameters())
            + list(base_model.out_proj.parameters())
            + [base_model.logit_weights, base_model.alpha]
            + list(base_model.head.parameters())
        )
        optimizer = torch.optim.AdamW(
            [{"params": [base_model.temperature,
                         base_model.logit_weights,
                         base_model.alpha],           "lr": 1e-2, "weight_decay": 0.0},
             {"params": list(base_model.token_proj.parameters())
                      + list(base_model.attn.parameters())
                      + list(base_model.token_weight.parameters())
                      + list(base_model.out_proj.parameters()), "lr": 1e-3, "weight_decay": 1e-3},
             {"params": list(base_model.head.parameters()),      "lr": 1e-3, "weight_decay": 1e-2}]
        )
        max_lr    = [1e-2, 1e-3, 1e-3]
        clip_norm = 1.0
        patience  = 15

    elif isinstance(base_model, BoostedTransformer):
        # Dual LR: unfrozen backbone parts (layer4/head) get 100× lower LR
        # than new transformer modules — prevents gradient explosion into backbone
        backbone_unfrozen = []
        for m in [base_model.m1, base_model.m2, base_model.m3]:
            backbone_unfrozen += [p for name, p in m.named_parameters()
                                  if ("layer4" in name or "head" in name)
                                  and p.requires_grad]
        new_modules = (
            list(base_model.res1.parameters())
            + list(base_model.res2.parameters())
            + list(base_model.token_proj.parameters())
            + list(base_model.transformer.parameters())
            + list(base_model.pool.parameters())
            + list(base_model.head.parameters())
        )
        optimizer = torch.optim.AdamW([
            {"params": backbone_unfrozen, "lr": 1e-5, "weight_decay": 1e-4},
            {"params": new_modules,       "lr": 1e-3, "weight_decay": 1e-3},
        ])
        max_lr    = [1e-5, 1e-3]
        clip_norm = 0.5   # tighter — transformer gradients can spike
        patience  = 20
    else:
        raise ValueError(f"hero_train_ensemble does not support {model_name}")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = max_lr,
        steps_per_epoch = len(train_loader),
        epochs          = epochs,
        pct_start       = 0.1,       # 10% warmup (longer than usual — helps attention)
        anneal_strategy = "cos",
        final_div_factor= 1e3,
    )

    # Label smoothing CrossEntropy — simpler than FocalLoss for ensemble
    # (backbone outputs are already calibrated; focal loss double-counts difficulty)
    criterion = nn.CrossEntropyLoss(
        weight        = class_weights,
        label_smoothing = 0.1,
    )
    scaler = GradScaler(enabled=(use_amp and device == "cuda"))

    # EMA on the ensemble head — backbone frozen so only head weights matter
    ema_model = deepcopy(base_model).to(device)
    ema_model.eval()

    def update_ema(model, ema_model, epoch, total_epochs):
        # Fixed 0.995 decay — ensemble converges fast, don't need slow warmup
        decay = 0.995
        with torch.no_grad():
            msd = model.module.state_dict() if isinstance(model, nn.DataParallel) \
                  else model.state_dict()
            for k, ema_v in ema_model.state_dict().items():
                model_v = msd[k]
                if not ema_v.is_floating_point():
                    ema_v.copy_(model_v)
                else:
                    ema_v.copy_(ema_v * decay + model_v * (1.0 - decay))

    save_path = f"{MODEL_DIR}/{model_name}_best.pth"
    best_state, best_val_acc = None, 0.0
    es_best_acc, es_counter  = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):

        # BoostedTransformer: set backbone parts to eval to keep BN stable
        # (they're partially unfrozen but we don't want BN stats to drift)
        model.train()
        if isinstance(base_model, BoostedTransformer):
            for m in [base_model.m1, base_model.m2, base_model.m3]:
                m.eval()   # keep backbone BN in eval mode
                for name, module in m.named_modules():
                    if "layer4" in name or "head" in name:
                        module.train()   # only partially-unfrozen parts in train

        total_loss, correct, total = 0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y_batch) in enumerate(tqdm(train_loader, desc=f"Train E{epoch+1}", leave=False)):
            x       = x.to(device)
            y_batch = y_batch.to(device)

            with autocast(device_type="cuda", enabled=(use_amp and device == "cuda")):
                logits = model(x)
                # Ensemble outputs are already soft — nan_to_num just in case
                logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
                loss   = criterion(logits, y_batch)

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss at step {step}, skipping")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_ema(model, ema_model, epoch, epochs)
            scheduler.step()

            total_loss += loss.item() * x.size(0)
            correct    += (logits.argmax(dim=1) == y_batch).sum().item()
            total      += y_batch.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc  = correct   / max(total, 1)

        # ── Validation ─────────────────────────────────────────────────
        ema_model.eval()
        val_loss_sum, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for x, y_batch in tqdm(val_loader, desc=f"Val E{epoch+1}", leave=False):
                x, y_batch = x.to(device), y_batch.to(device)
                logits = ema_model(x)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
                v_loss = criterion(logits, y_batch).item()
                if math.isfinite(v_loss):
                    val_loss_sum += v_loss * x.size(0)
                v_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                v_total   += y_batch.size(0)

        val_loss = val_loss_sum / max(v_total, 1)
        val_acc  = v_correct   / max(v_total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[{model_name}] Epoch {epoch+1} | "
              f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val loss={val_loss:.4f} acc={val_acc:.4f}")

        # Save on new best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu() for k, v in ema_model.state_dict().items()}
            torch.save(clean_state_dict(best_state), save_path)
            print(f"  >>> New best: {best_val_acc:.4f} — saved to {save_path}")

        # Collapse guard
        if val_acc < 0.4 and best_state is not None:
            print("  !!! Collapse — restoring best weights")
            ema_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            es_counter = 0
            continue

        # Early stopping
        if val_acc > es_best_acc + 1e-4:
            es_best_acc = val_acc
            es_counter  = 0
        else:
            es_counter += 1
            print(f"  [early_stop] {es_counter}/{patience} (best={es_best_acc:.4f})")
            if es_counter >= patience:
                print(">>> Early stopping triggered")
                break

    return best_state, val_loss, best_val_acc, history

model_names = [
    # "landslide_efficientnet_3",
    # "landslide_efficientnet_2",
    # "landslide_efficientnet",
    "resnet18",
    "mobilenet",
    "efficientnet_b3",
    # "boosted_transformer",
    # "weighted_ensembled",
    "stack_ensembled",
]

all_results = {}
model_histories = {}
train_time = {}

for model_name in model_names:
    start_time = time.time()

    # -------- DataLoaders --------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # -------- TRAIN --------

    ensemble_models = {"stack_ensembled", "weighted_ensembled", "boosted_transformer"}

    if model_name in ensemble_models:
        best_state, val_loss, val_acc, history = hero_train_ensemble(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            epochs=50,
            use_amp=True,
        )
    else:
        best_state, val_loss, val_acc, history = hero_train(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            epochs=EPOCHS,
            use_amp=True,
        )

    model_histories[model_name] = history

    print(
        f"\n>>> BEST {model_name}: "
        f"ValLoss={val_loss:.4f} "
        f"ValAcc={val_acc:.4f}"
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    train_time[model_name] = elapsed_time
    print(f"Total training time (GPU synchronized) of {model_name}: {elapsed_time:.4f} seconds")

    # -------- SAVE MODEL --------

    save_path = f"{MODEL_DIR}/{model_name}_best.pth"

    torch.save(clean_state_dict(best_state), save_path)

    print(f">>> Model saved to {save_path}")

for model_name, m_train_time in train_time.items():
    print(f"{model_name} - {m_train_time:.4f}s")

print(model_histories)
print(model_names)