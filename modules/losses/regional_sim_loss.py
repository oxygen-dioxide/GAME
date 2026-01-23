import torch
import torch.nn as nn
from torch import Tensor


class RegionalCosineSimilarityLoss(nn.Module):
    """
    This loss computes the cosine similarity between neighboring regions,
    encouraging similar features within the same region and dissimilar features across different regions.
    Arguments:
        neighborhood_size: int, regions with index difference within this size are considered neighbors.
    Inputs:
        x: Tensor of shape [B, T, C] where B is batch size, T is time frames, C is feature dimension.
        mapping: Tensor of shape [B, T] mapping each frame to a region index (1, 2, ..., N). 0 indicates no region.
    Outputs:
        loss: Scalar tensor representing the average regional cosine similarity loss.
    """

    def __init__(self, neighborhood_size: int = 3):
        super().__init__()
        self.neighborhood_size = neighborhood_size
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.mse_loss = nn.MSELoss(reduction="none")

    def generate_target_similarity(self, mapping: Tensor) -> tuple[Tensor, Tensor]:
        mapping1 = mapping.unsqueeze(2)  # [B, T, 1]
        mapping2 = mapping.unsqueeze(1)  # [B, 1, T]
        cos_sim_target = (mapping1 == mapping2).float()  # [B, T, T]
        symmetry_mask = torch.triu(torch.ones_like(cos_sim_target), diagonal=1).bool()  # [B, T, T]
        neighbor_mask = torch.abs(mapping1 - mapping2) <= self.neighborhood_size  # [B, T, T]
        padding_mask = (mapping1 != 0) & (mapping2 != 0)  # [B, T, T]
        mask = symmetry_mask & neighbor_mask & padding_mask  # [B, T, T]
        return cos_sim_target, mask

    def forward(self, x: Tensor, mapping: Tensor):
        cos_sim_target, mask = self.generate_target_similarity(mapping)  # [B, T, T]
        cos_sim_pred = self.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1))  # [B, T, T]
        unmask = ~mask
        cos_sim_target[unmask] = 0.0
        cos_sim_pred[unmask] = 0.0
        regional_loss = self.mse_loss(cos_sim_pred, cos_sim_target)
        loss = regional_loss.sum() / (mask.float().sum() + 1e-6)
        return loss
