from typing import List, Literal, Optional

from pydantic.dataclasses import dataclass

from ..base import ModelConfig


@dataclass
class SyncFixConfig(ModelConfig):
    """This is the Config for SyncFixModel class which defines all the useful parameters to be used in the model.

    Args:

        source_key (str):
            Key for the degraded (source) views. Defaults to "source_image"

        target_key (str):
            Key for the clean (target) views. Defaults to "target_image"

        latent_loss_weight (float):
            Weight of the latent loss. Defaults to 1.0

        latent_loss_type (str):
            Loss type to use in latent space. Defaults to "l2". Choices are "l2", "l1"

        pixel_loss_type (str):
            Pixel loss type to use. Defaults to "l2". Choices are "l2", "l1", "lpips"

        pixel_loss_weight (float):
            Weight of the pixel loss. Defaults to 0.0

        timestep_sampling (str):
            Timestep sampling to use. Defaults to "uniform".
            Choices are "uniform", "log_normal", "custom_timesteps"

        logit_mean (Optional[float]):
            Mean of the logit for the log normal distribution. Defaults to 0.0

        logit_std (Optional[float]):
            Standard deviation of the logit for the log normal distribution. Defaults to 1.0

        selected_timesteps (Optional[List[float]]):
            List of selected timesteps to be sampled from if using `custom_timesteps` timestep sampling. Defaults to None

        prob (Optional[List[float]]):
            List of probabilities for the selected timesteps if using `custom_timesteps` timestep sampling. Defaults to None

        bridge_noise_sigma (float):
            Amount of noise injected along the bridge interpolant. Defaults to 0.001

        use_l1_pixel_loss (bool):
            Add an L1 term to the pixel loss. Defaults to False

        use_ref (bool):
            Whether the last view of each sample is a clean reference. When True it is
            excluded from the losses. Defaults to False

        use_ssim_loss (bool):
            Add a (1 - SSIM) term to the pixel loss. Defaults to False

        use_gram_loss (bool):
            Add a VGG Gram (style) term to the pixel loss. Defaults to False
    """

    source_key: str = "source_image"
    target_key: str = "target_image"
    latent_loss_weight: float = 1.0
    latent_loss_type: Literal["l2", "l1"] = "l2"
    pixel_loss_type: Literal["l2", "l1", "lpips"] = "l2"
    pixel_loss_weight: float = 0.0
    timestep_sampling: Literal["uniform", "log_normal", "custom_timesteps"] = "uniform"
    logit_mean: Optional[float] = 0.0
    logit_std: Optional[float] = 1.0
    selected_timesteps: Optional[List[float]] = None
    prob: Optional[List[float]] = None
    bridge_noise_sigma: float = 0.001
    use_l1_pixel_loss: bool = False
    use_ref: bool = False
    use_ssim_loss: bool = False
    use_gram_loss: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.timestep_sampling == "log_normal":
            assert isinstance(self.logit_mean, float) and isinstance(
                self.logit_std, float
            ), "logit_mean and logit_std should be float for log_normal timestep sampling"

        if self.timestep_sampling == "custom_timesteps":
            assert isinstance(self.selected_timesteps, list) and isinstance(
                self.prob, list
            ), "timesteps and prob should be list for custom_timesteps timestep sampling"
            assert len(self.selected_timesteps) == len(
                self.prob
            ), "timesteps and prob should be of same length for custom_timesteps timestep sampling"
            assert (
                sum(self.prob) == 1
            ), "prob should sum to 1 for custom_timesteps timestep sampling"
