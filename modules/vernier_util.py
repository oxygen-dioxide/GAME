from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


@dataclass
class NoteVernierConfig:
    """Note Vernier 配置"""
    note_min: float = 0.0
    note_max: float = 127.0
    periods: Tuple = (48,24,12.0, 4,1.0,)

    @property
    def output_dim(self):
        # 1 (UV) + 1 (note) + len(periods) * 2 (Sin/Cos)
        return 1 + 1 + len(self.periods) * 2


class NoteVernierLoss(nn.Module):
    """Note Vernier Loss"""

    def __init__(self, config: NoteVernierConfig, w_uv=1.0, w_mse=10.0, w_cos=1.0):
        super().__init__()
        self.config = config
        self.w_uv = w_uv
        self.w_mse = w_mse
        self.w_cos = w_cos

    def forward(self, pred_vec, gt_note, gt_uv):
        cfg = self.config
        is_voiced = (gt_uv < 0.5)

        # UV Loss
        loss_uv = F.binary_cross_entropy_with_logits(pred_vec[..., 0], gt_uv)

        if is_voiced.sum() == 0:
            return self.w_uv * loss_uv

        p_voiced = pred_vec[is_voiced]
        g_voiced = gt_note[is_voiced]

        # MSE Loss (归一化)
        gt_norm = (g_voiced - cfg.note_min) / (cfg.note_max - cfg.note_min)
        pred_norm = torch.sigmoid(p_voiced[..., 1])  # 用 sigmoid 约束到 [0,1]
        loss_mse = F.mse_loss(pred_norm, gt_norm)

        # Cyclic Loss
        loss_cos = 0.0
        start_idx = 2
        for period in cfg.periods:
            phase = (g_voiced / period) * 2 * np.pi
            gt_vec = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)

            pred_ring = F.normalize(p_voiced[..., start_idx:start_idx + 2], dim=-1)
            loss_cos += 1.0 - (pred_ring * gt_vec).sum(dim=-1).mean()
            start_idx += 2

        return self.w_uv * loss_uv + self.w_mse * loss_mse + self.w_cos * loss_cos


def decode_note_vernier(pred_vec, config: NoteVernierConfig):
    """Note Vernier 解码"""
    cfg = config

    uv_prob = torch.sigmoid(pred_vec[..., 0])
    pred_uv = (uv_prob > 0.5)


    note = torch.sigmoid(pred_vec[..., 1]) * (cfg.note_max - cfg.note_min) + cfg.note_min

    start_idx = 2
    for period in cfg.periods:
        ring = F.normalize(pred_vec[..., start_idx:start_idx + 2], dim=-1)
        phase_val = torch.atan2(ring[..., 0], ring[..., 1])
        pred_offset = (phase_val / (2 * np.pi)) * period

        k = torch.round((note - pred_offset) / period)
        note = k * period + pred_offset
        start_idx += 2

    return note, pred_uv