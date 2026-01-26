import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn

from lib.functional import self_cosine_similarity, distance_transform
from lib.plot import similarity_to_figure, boundary_to_figure, distance_to_figure
from modules.losses import (
    ApproachingMomentumLoss,
    RegionalCosineSimilarityLoss,
)
from modules.syllable_splitter import SyllableSplitter
from .data import BaseDataset
from .pl_module_base import BaseLightningModule


class SyllablesDataset(BaseDataset):
    pass


class SyllablesLightningModule(BaseLightningModule):
    __dataset__ = SyllablesDataset

    def build_model(self) -> nn.Module:
        return SyllableSplitter(self.model_config)

    def register_losses_and_metrics(self) -> None:
        self.register_loss("region_loss", RegionalCosineSimilarityLoss(
            neighborhood_size=self.training_config.loss.region_loss.neighborhood_size,
            exponential_decay=self.training_config.loss.region_loss.exponential_decay,
        ))
        self.register_loss("boundary_loss", ApproachingMomentumLoss(
            constant_radius=self.training_config.loss.boundary_loss.constant_radius,
            cutoff_radius=self.training_config.loss.boundary_loss.cutoff_radius,
            decay_power=self.training_config.loss.boundary_loss.decay_power,
            decay_alpha=self.training_config.loss.boundary_loss.decay_alpha,
        ))

    def forward_model(self, sample: dict[str, torch.Tensor], infer: bool) -> dict[str, torch.Tensor]:
        spectrogram = sample["spectrogram"]
        language_ids = sample["language_id"]
        if not infer:
            language_ids = torch.where(
                torch.rand(language_ids.shape, device=language_ids.device) < 0.5,
                language_ids,
                torch.zeros_like(language_ids)
            )
        regions = sample["regions"]
        boundaries = sample["boundaries"]
        mask = regions != 0

        features, velocities = self.model(spectrogram, language_ids, mask=mask)  # [B, T, T]
        if infer:
            similarities = self_cosine_similarity(features)  # [B, T, T]
            return {
                "similarities": similarities,
                "velocities": velocities,
            }
        else:
            region_loss = self.losses["region_loss"](features, regions)
            boundary_loss = self.losses["boundary_loss"](velocities, boundaries, mask=mask)
            return {
                "region_loss": region_loss,
                "boundary_loss": boundary_loss,
            }

    def plot_validation_results(self, sample: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
        for i in range(len(sample["indices"])):
            data_idx = sample['indices'][i].item()
            T = self.valid_dataset.info["lengths"][data_idx]
            N = self.valid_dataset.info["durations"][data_idx]
            if data_idx >= self.training_config.validation.max_plots:
                continue
            similarities = outputs["similarities"][i, :T, :T]  # [T, T]
            velocities = outputs["velocities"][i, :T]  # [T]
            durations = sample["durations"][i, :N]  # [N]
            boundaries = sample["boundaries"][i, :T]  # [T]
            distance_gt = distance_transform(boundaries)
            distance_pred = velocities.cumsum(dim=0)
            self.plot_regions(
                data_idx, similarities, durations,
                title=self.valid_dataset.info["item_paths"][data_idx]
            )
            self.plot_distance(
                data_idx, distance_gt, distance_pred,
                title=self.valid_dataset.info["item_paths"][data_idx]
            )

    def plot_regions(
            self, idx: int,
            similarities: torch.Tensor, durations: torch.Tensor,
            title=None
    ):
        similarities = similarities.cpu().numpy()
        durations = durations.cpu().numpy()
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(f"regions/regions_{idx}", similarity_to_figure(
            similarities, durations, title=title
        ), global_step=self.global_step)

    def plot_boundaries(
            self, idx: int,
            boundaries_gt: torch.Tensor, boundaries_pred: torch.Tensor,
            durations_gt: torch.Tensor = None, durations_pred: torch.Tensor = None,
            title=None
    ):
        boundaries_gt = boundaries_gt.cpu().numpy()
        boundaries_pred = boundaries_pred.cpu().numpy()
        if durations_gt is not None:
            durations_gt = durations_gt.cpu().numpy()
        if durations_pred is not None:
            durations_pred = durations_pred.cpu().numpy()
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(f"boundaries/boundaries_{idx}", boundary_to_figure(
            boundaries_gt, boundaries_pred, dur_gt=durations_gt, dur_pred=durations_pred, title=title
        ), global_step=self.global_step)

    def plot_distance(
            self, idx: int,
            distance_gt: torch.Tensor, distance_pred: torch.Tensor,
            title=None
    ):
        distance_gt = distance_gt.cpu().numpy()
        distance_pred = distance_pred.cpu().numpy()
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(f"boundaries/boundaries_{idx}", distance_to_figure(
            distance_gt, distance_pred, title=title
        ), global_step=self.global_step)
