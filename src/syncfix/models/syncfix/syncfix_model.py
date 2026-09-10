from typing import Any, Dict, List, Optional, Tuple, Union

import lpips
import numpy as np
import torch
import torch.nn as nn
import torchvision
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from ..base.base_model import BaseModel
from ..embedders import ConditionerWrapper
from ..loss import gram_loss
from ..unets import DiffusersUNet2DCondWrapper, DiffusersUNet2DWrapper
from ..vae import AutoencoderKLDiffusers
from .syncfix_config import SyncFixConfig

# Relative weight of the VGG Gram (style) term inside the pixel loss.
GRAM_LOSS_WEIGHT = 0.1


def slice_ref(x, B, V):
    x = x.view(B, V, *x.shape[1:])
    x = x[:, :-1, ...]
    x = x.reshape(B * (V - 1), *x.shape[2:])
    return x


def rescale(x):
    x = (x + 1.0) / 2.0
    return x


class SyncFixModel(BaseModel):
    """This is the SyncFix class which defines the model.

    Args:

        config (SyncFixConfig):
            Configuration for the model

        denoiser (Union[DiffusersUNet2DWrapper, DiffusersTransformer2DWrapper]):
            Denoiser to use for the diffusion model. Defaults to None

        training_noise_scheduler (EulerDiscreteScheduler):
            Noise scheduler to use for training. Defaults to None

        sampling_noise_scheduler (EulerDiscreteScheduler):
            Noise scheduler to use for sampling. Defaults to None

        vae (AutoencoderKLDiffusers):
            VAE to use for the diffusion model. Defaults to None

        conditioner (ConditionerWrapper):
            Conditioner to use for the diffusion model. Defaults to None
    """

    @classmethod
    def load_from_config(cls, config: SyncFixConfig):
        return cls(config=config)

    def __init__(
        self,
        config: SyncFixConfig,
        denoiser: Union[
            DiffusersUNet2DWrapper,
            DiffusersUNet2DCondWrapper,
        ] = None,
        training_noise_scheduler: FlowMatchEulerDiscreteScheduler = None,
        sampling_noise_scheduler: FlowMatchEulerDiscreteScheduler = None,
        vae: AutoencoderKLDiffusers = None,
        conditioner: ConditionerWrapper = None,
    ):
        BaseModel.__init__(self, config)

        self.vae = vae
        self.denoiser = denoiser
        self.conditioner = conditioner
        self.sampling_noise_scheduler = sampling_noise_scheduler
        self.training_noise_scheduler = training_noise_scheduler
        self.timestep_sampling = config.timestep_sampling
        self.latent_loss_type = config.latent_loss_type
        self.latent_loss_weight = config.latent_loss_weight
        self.pixel_loss_type = config.pixel_loss_type
        self.pixel_loss_weight = config.pixel_loss_weight
        self.logit_mean = config.logit_mean
        self.logit_std = config.logit_std
        self.prob = config.prob
        self.selected_timesteps = config.selected_timesteps
        self.source_key = config.source_key
        self.target_key = config.target_key
        self.bridge_noise_sigma = config.bridge_noise_sigma
        self.use_l1_pixel_loss = config.use_l1_pixel_loss
        self.use_ref = config.use_ref
        self.use_ssim_loss = config.use_ssim_loss
        self.use_gram_loss = config.use_gram_loss

        self.num_iterations = nn.Parameter(
            torch.tensor(0, dtype=torch.float32), requires_grad=False
        )
        if self.pixel_loss_type == "lpips" and self.pixel_loss_weight > 0:
            self.lpips_loss = lpips.LPIPS(net="vgg").requires_grad_(False)

        else:
            self.lpips_loss = None

        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

        if self.use_gram_loss:
            self.net_vgg = torchvision.models.vgg16(pretrained=True).features
            for param in self.net_vgg.parameters():
                param.requires_grad_(False)
            self.t_vgg_renorm = torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )

    def on_fit_start(self, device: torch.device | None = None, *args, **kwargs):
        """Called when the training starts"""
        super().on_fit_start(device=device, *args, **kwargs)
        if self.vae is not None:
            self.vae.on_fit_start(device=device, *args, **kwargs)
        if self.conditioner is not None:
            self.conditioner.on_fit_start(device=device, *args, **kwargs)

    def forward(self, batch: Dict[str, Any], *args, **kwargs):
        """Bridge matching training step.

        The batch holds `V` views per sample, stacked as (B, V, C, H, W). When
        `use_ref` is set, the last view is a clean reference: it conditions the other
        views through the joint self-attention but is excluded from the losses.
        """
        self.num_iterations += 1

        B = batch[self.target_key].shape[0]
        V = batch[self.target_key].shape[1] if batch[self.target_key].dim() == 5 else 1

        target_pixels = self._flatten_views(batch[self.target_key])
        source_pixels = self._flatten_views(batch[self.source_key])

        # Encode both ends of the bridge
        z = self.vae.encode(target_pixels) if self.vae is not None else target_pixels
        source_pixels = torch.nn.functional.interpolate(
            source_pixels,
            size=target_pixels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(z.dtype)
        z_source = (
            self.vae.encode(source_pixels) if self.vae is not None else source_pixels
        )

        conditioning = self._get_conditioning(batch, *args, **kwargs)

        # Interpolate along the bridge: sigma=1 is the degraded end, sigma=0 the clean one
        timestep = self._timestep_sampling(n_samples=z.shape[0], device=z.device)
        sigmas = self._get_sigmas(
            self.training_noise_scheduler, timestep, n_dim=4, device=z.device
        )
        noisy_sample = (
            sigmas * z_source
            + (1.0 - sigmas) * z
            + self.bridge_noise_sigma
            * (sigmas * (1.0 - sigmas)) ** 0.5
            * torch.randn_like(z)
        )

        # The largest timestep is the degraded latent itself, with no bridge noise
        first_timestep = self.training_noise_scheduler.timesteps[0].item()
        noisy_sample = torch.where(
            (timestep == first_timestep).view(-1, 1, 1, 1), z_source, noisy_sample
        )

        prediction = self.denoiser(
            sample=noisy_sample,
            timestep=timestep,
            conditioning=conditioning,
            num_views=V,
        )

        # Flipped with respect to the usual flow target because of the sigma convention
        target = z_source - z

        if self.latent_loss_weight > 0:
            latent_recon_loss = self.latent_loss(prediction, target.detach(), B, V).mean()
        else:
            latent_recon_loss = torch.zeros((), device=z.device, dtype=prediction.dtype)

        denoised_sample = self._predicted_x_0(
            model_output=prediction,
            sample=noisy_sample,
            sigmas=sigmas,
        )

        if self.pixel_loss_weight > 0:
            pixel_recon_loss = self.pixel_loss(
                denoised_sample, target_pixels.detach(), B, V
            ).mean()
        else:
            pixel_recon_loss = torch.zeros((), device=z.device, dtype=prediction.dtype)

        return {
            "loss": self.latent_loss_weight * latent_recon_loss + pixel_recon_loss,
            "latent_recon_loss": latent_recon_loss,
            "pixel_recon_loss": pixel_recon_loss,
            "predicted_hr": denoised_sample,
            "noisy_sample": noisy_sample,
        }

    @staticmethod
    def _flatten_views(x: torch.Tensor) -> torch.Tensor:
        """Fold the view dimension of a (B, V, C, H, W) tensor into the batch."""
        return rearrange(x, "b v c h w -> (b v) c h w") if x.dim() == 5 else x

    def latent_loss(self, prediction, target, B, V):
        if self.use_ref:
            prediction = slice_ref(prediction, B, V)
            target = slice_ref(target, B, V)

        if self.latent_loss_type == "l2":
            error = (prediction - target) ** 2
        elif self.latent_loss_type == "l1":
            error = torch.abs(prediction - target)
        else:
            raise NotImplementedError(
                f"Loss type {self.latent_loss_type} not implemented"
            )

        return error.reshape(target.shape[0], -1).mean(1)

    def pixel_loss(self, prediction, target, B, V):
        if self.use_ref:
            prediction = slice_ref(prediction, B, V)
            target = slice_ref(target, B, V)

        decoded_prediction = self.vae.decode(prediction).clamp(-1, 1)

        # Decoding a latent rounds the spatial size down to a multiple of the VAE
        # downsampling factor, so the target has to follow.
        target = torch.nn.functional.interpolate(
            target,
            size=decoded_prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if self.pixel_loss_type == "l2":
            error = (decoded_prediction - target) ** 2
            return error.reshape(target.shape[0], -1).mean(1)

        elif self.pixel_loss_type == "l1":
            error = torch.abs(decoded_prediction - target)
            return error.reshape(target.shape[0], -1).mean(1)

        elif self.pixel_loss_type == "lpips":
            loss = (
                self.pixel_loss_weight
                * self.lpips_loss(decoded_prediction, target).mean()
            )
            if self.use_l1_pixel_loss:
                loss = loss + torch.abs(decoded_prediction - target).mean()
            if self.use_ssim_loss:
                loss = loss + (
                    1.0
                    - self.ssim_metric(rescale(decoded_prediction), rescale(target))
                )
            if self.use_gram_loss:
                loss = loss + GRAM_LOSS_WEIGHT * gram_loss(
                    self.t_vgg_renorm(rescale(decoded_prediction)),
                    self.t_vgg_renorm(rescale(target)),
                    self.net_vgg,
                )
            return loss

        else:
            raise NotImplementedError(
                f"Loss type {self.pixel_loss_type} not implemented"
            )

    def _get_conditioning(
        self,
        batch: Dict[str, Any],
        ucg_keys: List[str] = None,
        set_ucg_rate_zero=False,
        *args,
        **kwargs,
    ):
        """
        Get the conditionings
        """
        if self.conditioner is not None:
            return self.conditioner(
                batch,
                ucg_keys=ucg_keys,
                set_ucg_rate_zero=set_ucg_rate_zero,
                vae=self.vae,
                *args,
                **kwargs,
            )
        else:
            return None

    def _timestep_sampling(self, n_samples=1, device="cpu"):
        if self.timestep_sampling == "uniform":
            idx = torch.randint(
                0,
                self.training_noise_scheduler.config.num_train_timesteps,
                (n_samples,),
                device="cpu",
            )
            return self.training_noise_scheduler.timesteps[idx].to(device=device)

        elif self.timestep_sampling == "log_normal":
            u = torch.normal(
                mean=self.logit_mean,
                std=self.logit_std,
                size=(n_samples,),
                device="cpu",
            )
            u = torch.nn.functional.sigmoid(u)
            indices = (
                u * self.training_noise_scheduler.config.num_train_timesteps
            ).long()
            return self.training_noise_scheduler.timesteps[indices].to(device=device)

        elif self.timestep_sampling == "custom_timesteps":
            idx = np.random.choice(len(self.selected_timesteps), n_samples, p=self.prob)
            return torch.tensor(
                self.selected_timesteps, device=device, dtype=torch.long
            )[idx]

    def _predicted_x_0(
        self,
        model_output,
        sample,
        sigmas=None,
    ):
        """
        Predict x_0, the denoised sample, using the model output and the timesteps depending on the prediction type.
        """
        pred_x_0 = sample - model_output * sigmas
        return pred_x_0

    def _get_sigmas(
        self, scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cpu"
    ):
        sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        num_steps: int = 20,
        conditioner_inputs: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
        num_views: Optional[int] = None,
        num_reference_samples: int = 0,
        verbose: bool = False,
    ):
        self.sampling_noise_scheduler.set_timesteps(
            sigmas=np.linspace(1, 1 / num_steps, num_steps)
        )

        sample = z

        # Get conditioning
        conditioning = self._get_conditioning(
            conditioner_inputs, set_ucg_rate_zero=True, device=z.device
        )

        if (
            num_views is None
            and conditioner_inputs is not None
            and self.source_key in conditioner_inputs
            and conditioner_inputs[self.source_key].dim() == 5
        ):
            num_views = conditioner_inputs[self.source_key].shape[1]

        # If max_samples parameter is provided, limit the number of samples
        if max_samples is not None:
            sample = sample[:max_samples]

        if conditioning:
            conditioning["cond"] = {
                k: v[:max_samples] for k, v in conditioning["cond"].items()
            }

        clean_reference = None
        if num_reference_samples > 0:
            if num_reference_samples > sample.shape[0]:
                raise ValueError(
                    "num_reference_samples cannot be larger than the sampled batch size"
                )
            clean_reference = sample[-num_reference_samples:].clone()

        for i, t in tqdm(
            enumerate(self.sampling_noise_scheduler.timesteps), disable=not verbose
        ):
            if hasattr(self.sampling_noise_scheduler, "scale_model_input"):
                denoiser_input = self.sampling_noise_scheduler.scale_model_input(
                    sample, t
                )
            else:
                denoiser_input = sample

            # Predict noise level using denoiser using conditionings
            pred = self.denoiser(
                sample=denoiser_input,
                timestep=t.to(z.device).repeat(denoiser_input.shape[0]),
                conditioning=conditioning,
                num_views=num_views,
            )

            # Make one step on the reverse diffusion process
            sample = self.sampling_noise_scheduler.step(
                pred, t, sample, return_dict=False
            )[0]
            if clean_reference is not None:
                sample[-num_reference_samples:] = clean_reference

            # multi-step sampling
            if i < len(self.sampling_noise_scheduler.timesteps) - 1:
                timestep = (
                    self.sampling_noise_scheduler.timesteps[i + 1]
                    .to(z.device)
                    .repeat(sample.shape[0])
                )
                sigmas = self._get_sigmas(
                    self.sampling_noise_scheduler, timestep, n_dim=4, device=z.device
                )
                sample = sample + self.bridge_noise_sigma * (
                    sigmas * (1.0 - sigmas)
                ) ** 0.5 * torch.randn_like(sample)
                sample = sample.to(z.dtype)
                if clean_reference is not None:
                    sample[-num_reference_samples:] = clean_reference

        decoded_sample = self.vae.decode(sample) if self.vae is not None else sample

        return decoded_sample

    def log_samples(
        self,
        batch: Dict[str, Any],
        input_shape: Optional[Tuple[int, int, int]] = None,
        max_samples: Optional[int] = None,
        num_steps: Union[int, List[int]] = 20,
    ):
        if isinstance(num_steps, int):
            num_steps = [num_steps]

        logs = {}

        N = max_samples if max_samples is not None else len(batch[self.source_key])

        batch = {k: v[:N] for k, v in batch.items()}
        # infer input shape based on VAE configuration if not passed
        if input_shape is None:
            if self.vae is not None:
                # get input pixel size of the vae
                input_shape = batch[self.target_key].shape[2:]
                # rescale to latent size
                input_shape = (
                    self.vae.latent_channels,
                    input_shape[0] // self.vae.downsampling_factor,
                    input_shape[1] // self.vae.downsampling_factor,
                )
            else:
                raise ValueError(
                    "input_shape must be passed when no VAE is used in the model"
                )
        num_views = (
            batch[self.source_key].shape[1]
            if batch[self.source_key].dim() == 5
            else 1
        )
        source_image = self._flatten_views(batch[self.source_key])
        source_image = torch.nn.functional.interpolate(
            source_image,
            size=batch[self.target_key].shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(dtype=self.dtype)

        z = self.vae.encode(source_image) if self.vae is not None else source_image

        for num_step in num_steps:
            with torch.autocast(dtype=self.dtype, device_type="cuda"):
                logs[f"samples_{num_step}_steps"] = self.sample(
                    z,
                    num_steps=num_step,
                    conditioner_inputs=batch,
                    max_samples=N * num_views,
                )

        return logs

    def compute_metrics(self, batch: Dict[str, Any], predicted_sample):
        target_pixels = self._flatten_views(batch[self.target_key])

        decoded_image_prediction = self.vae.decode(predicted_sample)
        decoded_image_prediction = torch.nn.functional.interpolate(
            decoded_image_prediction,
            size=batch[self.target_key].shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(dtype=self.dtype)
        psnr = self.psnr_metric(decoded_image_prediction, target_pixels)
        ssim = self.ssim_metric(decoded_image_prediction, target_pixels)
        lpips = self.lpips_loss(decoded_image_prediction, target_pixels)
        return {
            "psnr": float(psnr),
            "ssim": float(ssim),
            "lpips": float(lpips.mean()),
        }
