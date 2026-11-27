"""
snapvla.py — Self-Contained Vision-Language-Action (VLA) Policy, MuJoCo Environment,
Data-Efficiency Ablation (ResNet-FiLM-SpatialSoftmax vs. ViT), and ONNX INT8 Quantization.
"""
import time
import numpy as np
import mujoco
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

# ==============================================================================
# 1. 2D Spatial Softmax & Vision-Language-Action Architectures
# ==============================================================================
class SpatialSoftmax2D(nn.Module):
    """Extracts expected 2D (x, y) keypoint coordinates from CNN feature maps."""
    def __init__(self, height: int = 16, width: int = 16, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij"
        )
        self.register_buffer("grid", torch.stack([xs, ys], dim=-1).view(-1, 2))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        attn = F.softmax(feat.view(B, C, H * W) / self.temperature, dim=-1)
        return torch.einsum("bcn,nd->bcd", attn, self.grid)  # [B, C, 2]

class ResNetSpatialSoftmaxVLA(nn.Module):
    """Proposed VLA: ConvNet + CLIP Language FiLM + 2D Spatial Softmax Coordinate Bottleneck."""
    def __init__(self, d_model: int = 128, proprio_dim: int = 3, lang_dim: int = 512, action_dim: int = 2):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.ReLU())
        self.lang_film = nn.Linear(lang_dim, 48 * 2)
        self.conv3 = nn.Conv2d(48, 16, kernel_size=1)
        self.ssm = SpatialSoftmax2D(height=16, width=16)
        self.lang_short = nn.Linear(lang_dim, 64)
        self.head = nn.Sequential(
            nn.Linear(16 * 2 + proprio_dim + 64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim)
        )

    def forward(self, img: torch.Tensor, proprio: torch.Tensor, lang_vec: torch.Tensor) -> torch.Tensor:
        x = self.conv2(self.conv1(img))
        gamma, beta = self.lang_film(lang_vec).chunk(2, dim=-1)
        x = F.relu(x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None])
        keypoints = self.ssm(self.conv3(x)).flatten(1)
        l_emb = F.relu(self.lang_short(lang_vec))
        fused = torch.cat([keypoints, proprio, l_emb], dim=-1)
        return torch.tanh(self.head(fused))

class ViTPatchVLA(nn.Module):
    """Baseline VLA: Vision Transformer (ViT) Patch Encoder + Language Cross-Attention."""
    def __init__(self, d_model: int = 128, proprio_dim: int = 3, lang_dim: int = 512, action_dim: int = 2):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=8, stride=8)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.lang_q = nn.Linear(lang_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_model + proprio_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim)
        )

    def forward(self, img: torch.Tensor, proprio: torch.Tensor, lang_vec: torch.Tensor) -> torch.Tensor:
        vis_tokens = self.transformer(self.patch_embed(img).flatten(2).transpose(1, 2) + self.pos_embed)
        q = self.lang_q(lang_vec).unsqueeze(1)
        attended, _ = self.cross_attn(q, vis_tokens, vis_tokens)
        fused = torch.cat([attended.squeeze(1), proprio], dim=-1)
        return torch.tanh(self.head(fused))

# ==============================================================================
# 2. Training & Static INT8 ONNX Quantization Entrypoints
# ==============================================================================
def train_policy(model_cls, episodes, epochs=35, lr=1.5e-3, device="cpu"):
    imgs = np.concatenate([e["images"] for e in episodes])
    props = np.concatenate([e["proprio"] for e in episodes])
    acts = np.concatenate([e["targets"] for e in episodes])
    langs = np.concatenate([np.repeat(e["lang_vec"][None, :], len(e["images"]), axis=0) for e in episodes])
    ds = TensorDataset(
        torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0,
        torch.from_numpy(props).float(),
        torch.from_numpy(langs).float(),
        torch.from_numpy(acts).float(),
    )
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model = model_cls().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for img, prop, lang, act in loader:
            img, prop, lang, act = img.to(device), prop.to(device), lang.to(device), act.to(device)
            opt.zero_grad()
            loss = loss_fn(model(img, prop, lang), act)
            loss.backward()
            opt.step()
        sched.step()
    return model.eval()

def export_and_quantize_int8(model: nn.Module, calib_reader: CalibrationDataReader):
    model.cpu().eval()
    dummy_in = (torch.randn(1, 3, 64, 64), torch.randn(1, 3), torch.randn(1, 512))
    torch.onnx.export(
        model, dummy_in, "snapvla_fp32.onnx",
        input_names=["image", "proprio", "lang_vec"],
        output_names=["target_xy"],
        opset_version=17, do_constant_folding=True, dynamo=False
    )
    quantize_static(
        model_input="snapvla_fp32.onnx",
        model_output="snapvla_int8.onnx",
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
