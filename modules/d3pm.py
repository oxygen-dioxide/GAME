import torch
from torch.nn import functional as F


def d3pm_region_noise(regions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    p = (1 + torch.cos(t * torch.pi)) / 2
    regions = merge_random_regions(regions, p=p)
    return regions


def merge_random_regions(regions: torch.Tensor, p: torch.Tensor):
    """
    :param regions: [..., T]
    :param p: [...]
    :return: [..., T]
    """
    N = regions.max()
    if N <= 1:
        return regions.clone()
    *B, _ = regions.shape
    drops = torch.rand((*B, N - 1), device=regions.device) < p.unsqueeze(-1)  # [..., N-1]
    shifts = F.pad(drops.long().cumsum(dim=-1), (2, 0), mode="constant", value=0)  # [..., N+1]
    regions_merged = regions - shifts.gather(dim=-1, index=regions)  # [..., T]
    return regions_merged


def split_random_regions(regions: torch.Tensor, p: torch.Tensor):
    """
    :param regions: [..., T]
    :param p: [...]
    :return: [..., T]
    """
    p = p.clamp(max=0.99).unsqueeze(-1)  # [..., 1]
    N = regions.amax(dim=-1, keepdim=True).clamp(min=1)  # [..., 1]
    L = (regions != 0).sum(dim=-1, keepdim=True).clamp(min=1)  # [..., 1]
    pi_hat = (N.float() / L.float() / (1 - p)).clamp(max=1.0)  # [..., 1]
    P = p * pi_hat / (p * pi_hat + (1 - pi_hat))  # [..., 1]
    boundaries = F.pad(torch.diff(regions, dim=-1) > 0, (1, 0), mode="constant", value=0)  # [..., T]
    inserts = (torch.rand_like(regions, dtype=torch.float32) < P) & ~boundaries  # [..., T]
    shifts = inserts.long().cumsum(dim=-1)  # [..., T]
    regions_split = regions + shifts * (regions != 0).long()  # [..., T]
    return regions_split
