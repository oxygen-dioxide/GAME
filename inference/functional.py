from torch import Tensor
from torch.nn import functional as F

from modules.commons.tts_modules import LengthRegulator


def format_regions(
        durations: Tensor,
        length: int | Tensor,
        timestep: float,
        lr: LengthRegulator,
) -> Tensor:
    boundary_indices = durations.cumsum(dim=1).div(timestep).round().long()  # [B, Nr]
    boundary_indices = F.pad(boundary_indices, (1, 0), mode="constant", value=0)  # [B, Nr+1]
    boundary_indices = boundary_indices.clamp(min=0, max=length)  # [B, Nr+1]
    region_durations = boundary_indices[:, 1:] - boundary_indices[:, :-1]  # [B, Nr]
    regions = lr(region_durations)  # [B, T']
    regions = F.pad(regions, (0, length - regions.size(1)), mode="constant", value=0)  # [B, T]
    return regions
