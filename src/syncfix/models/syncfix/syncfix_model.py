from typing import Any, Dict, List, Optional, Tuple, Union

import lpips
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
from einops import rearrange, repeat
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torchvision
# from dreamsim import dreamsim

from ..base.base_model import BaseModel
from ..embedders import ConditionerWrapper
from ..unets import DiffusersUNet2DCondWrapper, DiffusersUNet2DWrapper
from ..vae import AutoencoderKLDiffusers
from .syncfix_config import SyncFixConfig

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
        self.pixel_loss_max_size = config.pixel_loss_max_size
        self.pixel_loss_weight = config.pixel_loss_weight
        self.logit_mean = config.logit_mean
        self.logit_std = config.logit_std
        self.prob = config.prob
        self.selected_timesteps = config.selected_timesteps
        self.source_key = config.source_key
        self.target_key = config.target_key
        self.mask_key = config.mask_key
        self.bridge_noise_sigma = config.bridge_noise_sigma
        self.use_l1_pixel_loss = config.use_l1_pixel_loss
        self.use_depth = config.use_depth
        self.use_ref = config.use_ref

        self.num_iterations = nn.Parameter(
            torch.tensor(0, dtype=torch.float32), requires_grad=False
        )
        if self.pixel_loss_type == "lpips" and self.pixel_loss_weight > 0:
            self.lpips_loss = lpips.LPIPS(net="vgg").requires_grad_(False)

        else:
            self.lpips_loss = None

        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

        self.use_ssim_loss = config.use_ssim_loss
        self.use_gram_loss = config.use_gram_loss
        self.use_dsim_loss = config.use_dsim_loss
        self.dsim_loss_weight = config.dsim_loss_weight

        if self.use_gram_loss:
            self.net_vgg = torchvision.models.vgg16(pretrained=True).features
            for param in self.net_vgg.parameters():
                param.requires_grad_(False)
            self.t_vgg_renorm = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def on_fit_start(self, device: torch.device | None = None, *args, **kwargs):
        """Called when the training starts"""
        super().on_fit_start(device=device, *args, **kwargs)
        if self.vae is not None:
            self.vae.on_fit_start(device=device, *args, **kwargs)
        if self.conditioner is not None:
            self.conditioner.on_fit_start(device=device, *args, **kwargs)

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
        for num_step in num_steps:
            if batch[self.source_key].dim() == 5:
                num_view = batch[self.source_key].shape[1]
                source_image = rearrange(batch[self.source_key], "b v c h w -> (b v) c h w")
            else:
                source_image = batch[self.source_key]
            source_image = torch.nn.functional.interpolate(
                source_image,
                size=batch[self.target_key].shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(dtype=self.dtype)

            z = self.vae.encode(source_image) if self.vae is not None else source_image

            with torch.autocast(dtype=self.dtype, device_type="cuda"):
                logs[f"samples_{num_step}_steps"] = self.sample(
                    z,
                    num_steps=num_step,
                    conditioner_inputs=batch,
                    max_samples=N*num_view,
                )

        return logs
    
    def compute_metrics(self, batch: Dict[str, Any], predicted_sample):
        target_pixels = rearrange(batch[self.target_key], "b v c h w -> (b v) c h w")

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




