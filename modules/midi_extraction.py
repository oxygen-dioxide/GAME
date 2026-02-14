import torch
from torch import nn
import torch.nn.functional as F
from lib.config.schema import ModelConfig
from lib.reflection import build_object_from_class_name
from modules.commons.common_layers import CyclicRegionEmbedding, LocalDownsample
from modules.d3pm import d3pm_region_noise


class GeneratorSegmentationModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.spectrogram_projection = nn.Linear(config.in_channels, config.embedding_dim)
        self.region_embedding = CyclicRegionEmbedding(
            config.embedding_dim,
            cycle_length=config.region_cycle_len
        )
        self.use_language_embedding = config.use_languages
        if self.use_language_embedding:
            self.language_embedding = nn.Embedding(config.num_languages + 1, config.embedding_dim, padding_idx=0)
        self.segmenter = build_object_from_class_name(
            config.segmenter.cls, nn.Module,
            config.embedding_dim, 1, True,
            **config.segmenter.kwargs
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(1, config.embedding_dim * 4),
            nn.GELU(),
            nn.Linear(config.embedding_dim * 4, config.embedding_dim)
        )

    def forward(self, spectrogram, regions, times, language=None, mask=None):
        x = self.spectrogram_projection(spectrogram) + self.region_embedding(regions)
        if self.use_language_embedding:
            x = x + self.language_embedding(language.unsqueeze(-1))
        time_emb = self.time_embedding(times.unsqueeze(-1))
        x = x + time_emb
        x, latent = self.segmenter(x, times, mask=mask)
        velocities = x.squeeze(-1).tanh()
        return velocities, latent


class SegmentationModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.spectrogram_projection = nn.Linear(config.in_channels, config.embedding_dim)
        self.region_embedding = CyclicRegionEmbedding(
            config.embedding_dim,
            cycle_length=config.region_cycle_len
        )
        self.mode = config.mode
        if self.mode == "d3pm":
            self.time_embedding = nn.Sequential(
                nn.Linear(1, config.embedding_dim * 4),
                nn.GELU(),
                nn.Linear(config.embedding_dim * 4, config.embedding_dim)
            )
        self.use_language_embedding = config.use_languages
        if self.use_language_embedding:
            self.language_embedding = nn.Embedding(config.num_languages + 1, config.embedding_dim, padding_idx=0)
        self.segmenter = build_object_from_class_name(
            config.segmenter.cls, nn.Module,
            config.embedding_dim, 1, True,
            **config.segmenter.kwargs
        )

    def forward(self, spectrogram, regions, t=None, language=None, mask=None):
        x = self.spectrogram_projection(spectrogram) + self.region_embedding(regions)
        if self.mode == "d3pm":
            x = x + self.time_embedding(t[..., None, None])
        if self.use_language_embedding:
            x = x + self.language_embedding(language.unsqueeze(-1))
        x, latent = self.segmenter(x, mask=mask)
        x = x.squeeze(-1)
        return x, latent


class EstimationModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.spectrogram_projection = nn.Linear(config.in_channels, config.embedding_dim)
        self.region_embedding = CyclicRegionEmbedding(
            config.embedding_dim,
            cycle_length=config.region_cycle_len
        )
        self.use_glu = config.use_glu
        if self.use_glu:
            adaptor_out_dim = config.embedding_dim * 2
        else:
            adaptor_out_dim = config.embedding_dim
        self.adaptor = build_object_from_class_name(
            config.adaptor.cls, nn.Module,
            config.embedding_dim, adaptor_out_dim, True,
            **config.adaptor.kwargs
        )
        if self.use_glu:
            self.glu = nn.GLU(dim=-1)
        self.downsample = LocalDownsample()
        self.estimator = build_object_from_class_name(
            config.estimator.cls, nn.Module,
            config.embedding_dim, config.estimator_out_channels, False,
            **config.estimator.kwargs
        )

    def forward(self, spectrogram, regions, max_n: int, t_mask=None, n_mask=None):
        x = self.spectrogram_projection(spectrogram) + self.region_embedding(regions)
        x, latent = self.adaptor(x, mask=t_mask)
        if self.use_glu:
            x = self.glu(x)
        x_down = self.downsample(x, regions, max_n=max_n)
        estimations = self.estimator(x_down, mask=n_mask)
        return estimations, latent


class GAMEModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.spectrogram_projection = nn.Linear(config.in_channels, config.embedding_dim)
        self.use_spectrogram_encoder_glu = config.use_spectrogram_encoder_glu
        self.spectrogram_encoder = build_object_from_class_name(
            config.spectrogram_encoder.cls, nn.Module,
            config.embedding_dim,
            config.embedding_dim * 4 if self.use_spectrogram_encoder_glu else config.embedding_dim * 2, True,
            **config.spectrogram_encoder.kwargs
        )

        self.segmentation_mode = config.mode
        if self.segmentation_mode == "d3pm":
            self.segmentation_time_embedding = nn.Sequential(
                nn.Linear(1, config.embedding_dim * 4),
                nn.GELU(),
                nn.Linear(config.embedding_dim * 4, config.embedding_dim)
            )
        self.segmentation_use_language_embedding = config.use_languages
        if self.segmentation_use_language_embedding:
            self.segmentation_language_embedding = nn.Embedding(config.num_languages + 1, config.embedding_dim,
                                                                padding_idx=0)

        self.segmentation_region_embedding = CyclicRegionEmbedding(
            config.embedding_dim,
            cycle_length=config.region_cycle_len
        )
        self.segmenter = build_object_from_class_name(
            config.segmenter.cls, nn.Module,
            config.embedding_dim, 1, True,
            **config.segmenter.kwargs
        )

        self.estimator_region_embedding = CyclicRegionEmbedding(
            config.embedding_dim,
            cycle_length=config.region_cycle_len
        )
        self.estimator_use_glu = config.use_glu
        if self.estimator_use_glu:
            adaptor_out_dim = config.embedding_dim * 2
        else:
            adaptor_out_dim = config.embedding_dim
        self.adaptor = build_object_from_class_name(
            config.adaptor.cls, nn.Module,
            config.embedding_dim, adaptor_out_dim, True,
            **config.adaptor.kwargs
        )

        self.downsample = LocalDownsample()
        self.estimator = build_object_from_class_name(
            config.estimator.cls, nn.Module,
            config.embedding_dim, config.estimator_out_channels, False,
            **config.estimator.kwargs
        )

    def segmentation_forward(self, spectrogram, regions, language, mask=None, time_steps=None, ):
        x_segmentation = spectrogram + self.segmentation_region_embedding(regions)
        if self.segmentation_mode == "d3pm":
            x_segmentation = x_segmentation + self.segmentation_time_embedding(time_steps[..., None, None])
        if self.segmentation_use_language_embedding:
            x_segmentation = x_segmentation + self.segmentation_language_embedding(language.unsqueeze(-1))
        x_segmentation, segmentation_latent = self.segmenter(x_segmentation, mask=mask)
        x_segmentation = x_segmentation.squeeze(-1)
        return x_segmentation, segmentation_latent

    def segmentation_infer(self, spectrogram, mask, language_ids, d3pm_sample_steps):
        B = spectrogram.shape[0]
        if self.segmentation_mode == "d3pm":
            # 1. Initialize with a whole region (no boundaries).
            # 2. Merge regions by p(t) before each step.
            # 3. Predict full boundaries.
            latent = None
            soft_boundaries = None
            boundaries_pred = None
            num_steps = d3pm_sample_steps
            timestep = torch.full(
                (B,), fill_value=1 / num_steps,
                dtype=torch.float32, device=spectrogram.device
            )
            regions_pred = mask.long()
            for i in range(num_steps):
                t = i * timestep
                noise = d3pm_region_noise(regions_pred, t=t)  # [B, T]
                logits, latent_ = self.segmentation_forward(
                    spectrogram, regions=noise, time_steps=t,
                    language=language_ids, mask=mask,
                )  # [B, T]
                if i == 0:
                    latent = latent_
                soft_boundaries, boundaries_pred = self._decode_boundaries(logits, mask)
                regions_pred = (boundaries_pred.long().cumsum(dim=-1) + 1) * mask.long()
        elif self.segmentation_mode == "completion":
            # One-step prediction from a whole region (no boundaries).
            logits, latent = self.segmentation_forward(
                spectrogram, regions=mask.long(),
                language=language_ids, mask=mask,
            )  # [B, T]
            soft_boundaries, boundaries_pred = self._decode_boundaries(logits, mask)
        else:
            raise ValueError(f"Unknown mode: {self.segmentation_mode}.")

        return soft_boundaries, boundaries_pred, latent

    def encoer_forward(self, spectrogram, mask=None):
        x = self.spectrogram_projection(spectrogram)

        x, encoder_latent = self.spectrogram_encoder(x, mask=mask)
        if self.use_spectrogram_encoder_glu:
            x = F.glu(x, dim=-1)
        x_segmentation, x_estimation = x.chunk(2, dim=-1)
        return x_segmentation, x_estimation, encoder_latent

    def estimator_forward(self, x_estimation, regions, max_n: int, t_mask=None, n_mask=None):
        x = x_estimation + self.estimator_region_embedding(regions)
        x, adaptor_latent = self.adaptor(x, mask=t_mask)
        if self.estimator_use_glu:
            x = F.glu(x, dim=-1)
        x_down = self.downsample(x, regions, max_n=max_n)
        estimations = self.estimator(x_down, mask=n_mask)
        return estimations, adaptor_latent

    def forward(
            self,
            spectrogram,
            max_n: int,
            regions_segmentation=None,
            regions_estimation=None,
            time_steps=None,
            language=None,
            t_mask=None,
            n_mask=None,
            infer=False,
            d3pm_sample_steps: int = 10
    ):

        if infer:

            x_segmentation, x_estimation, _ = self.encoer_forward(
                spectrogram,
                mask=t_mask
            )

            soft_boundaries, boundaries_pred, latent_segmentation = self.segmentation_infer(
                spectrogram=spectrogram,
                mask=t_mask,
                language_ids=language,
                d3pm_sample_steps=d3pm_sample_steps
            )
            regions_pred = (boundaries_pred.long().cumsum(dim=-1) + 1)
            if t_mask is not None:
                regions_pred = regions_pred * t_mask.long()

            estimations_out_pred_regions, adaptor_latent_pred_regions = self.estimator_forward(
                x_estimation,
                regions=regions_pred,
                max_n=max_n,
                t_mask=t_mask,
                n_mask=n_mask
            )

            if regions_estimation is not None:
                estimations_out_gt, adaptor_latent_gt = self.estimator_forward(
                    x_estimation,
                    regions=regions_pred,
                    max_n=max_n,
                    t_mask=t_mask,
                    n_mask=n_mask
                )
            else:
                estimations_out_gt = None
                adaptor_latent_gt = None

            return (soft_boundaries, boundaries_pred, latent_segmentation), (
                (estimations_out_pred_regions, adaptor_latent_pred_regions), (estimations_out_gt, adaptor_latent_gt))


        else:

            x_segmentation, x_estimation, encoder_latent = self.encoer_forward(
                spectrogram,
                mask=t_mask
            )

            segmentation_out, segmentation_latent = self.segmentation_forward(
                x_segmentation,
                regions=regions_segmentation,
                language=language
                , mask=t_mask,
                time_steps=time_steps
            )
            estimations_out, adaptor_latent = self.estimator_forward(
                x_estimation,
                regions=regions_estimation,
                max_n=max_n,
                t_mask=t_mask,
                n_mask=n_mask
            )
            return (encoder_latent, segmentation_latent, adaptor_latent), (segmentation_out, estimations_out)

            # x = self.spectrogram_projection(spectrogram)
            #
            # x,encoder_latent=self.spectrogram_encoder(x,mask=t_mask)
            # if self.use_spectrogram_encoder_glu:
            #     x = F.glu(x,dim=-1)
            # x_segmentation,x_estimation=x.chunk(2, dim=-1)
            #
            # x_segmentation= x_segmentation + self.segmentation_region_embedding(regions_segmentation)
            # if self.segmentation_mode == "d3pm":
            #     x_segmentation = x_segmentation + self.segmentation_time_embedding(time_steps[..., None, None])
            # if self.segmentation_use_language_embedding:
            #     x_segmentation = x_segmentation + self.segmentation_language_embedding(language.unsqueeze(-1))
            # x_segmentation, segmentation_latent = self.segmenter(x_segmentation, mask=t_mask)
            # x_segmentation = x_segmentation.squeeze(-1)
            #
            # x = x_estimation + self.estimator_region_embedding(regions_estimation)
            # x, adaptor_latent = self.adaptor(x, mask=t_mask)
            # if self.estimator_use_glu:
            #     x = F.glu(x,dim=-1)
            # x_down = self.downsample(x, regions_estimation, max_n=max_n)
            # estimations = self.estimator(x_down, mask=n_mask)
