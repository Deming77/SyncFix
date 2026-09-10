"""Shared construction of the SyncFix backbone, used by both training and inference."""

from typing import List, Optional

import torch

from .embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)
from .unets import DiffusersUNet2DCondWrapper
from .vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig


def build_denoiser(
    unet_input_channels: int = 4,
    vae_num_channels: int = 4,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> DiffusersUNet2DCondWrapper:
    """Build the SDXL-derived UNet used as the bridge denoiser.

    The architecture is a trimmed SDXL UNet (3 resolution levels, no added time/text
    embeddings), so an SDXL checkpoint can be adapted into it with `StateDictAdapter`.
    """
    return DiffusersUNet2DCondWrapper(
        in_channels=unet_input_channels,
        out_channels=vae_num_channels,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=[320, 640, 1280],
        transformer_layers_per_block=[1, 2, 10],
        reverse_transformer_layers_per_block=None,
        encoder_hid_dim=None,
        encoder_hid_dim_type=None,
        attention_head_dim=[5, 10, 20],
        num_attention_heads=None,
        dual_cross_attention=False,
        use_linear_projection=True,
        class_embed_type=None,
        addition_embed_type=None,
        addition_time_embed_dim=None,
        num_class_embeds=None,
        upcast_attention=None,
        resnet_time_scale_shift="default",
        resnet_skip_time_act=False,
        resnet_out_scale_factor=1.0,
        time_embedding_type="positional",
        time_embedding_dim=None,
        time_embedding_act_fn=None,
        timestep_post_act=None,
        time_cond_proj_dim=None,
        conv_in_kernel=3,
        conv_out_kernel=3,
        projection_class_embeddings_input_dim=None,
        attention_type="default",
        class_embeddings_concat=False,
        mid_block_only_cross_attention=None,
        cross_attention_norm=None,
        addition_embed_type_num_heads=64,
    ).to(torch_dtype)


def build_vae(
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> AutoencoderKLDiffusers:
    """Build the frozen SDXL VAE used to move between pixel and latent space."""
    vae = AutoencoderKLDiffusers(
        AutoencoderKLDiffusersConfig(
            version=backbone_signature,
            subfolder="vae",
            tiling_size=(128, 128),
        )
    )
    vae.freeze()
    return vae.to(torch_dtype)


def build_conditioner(
    conditioning_images_keys: Optional[List[str]] = None,
    conditioning_masks_keys: Optional[List[str]] = None,
) -> ConditionerWrapper:
    """Build the conditioner wrapper, empty unless extra latents are concatenated."""
    conditioning_images_keys = conditioning_images_keys or []
    conditioning_masks_keys = conditioning_masks_keys or []

    conditioners = []
    if conditioning_images_keys or conditioning_masks_keys:
        latent_concat_embedder = LatentsConcatEmbedder(
            LatentsConcatEmbedderConfig(
                image_keys=conditioning_images_keys,
                mask_keys=conditioning_masks_keys,
            )
        )
        latent_concat_embedder.freeze()
        conditioners.append(latent_concat_embedder)

    return ConditionerWrapper(conditioners=conditioners)
