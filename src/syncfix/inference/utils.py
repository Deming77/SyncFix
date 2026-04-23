import logging
import os
from typing import List, Optional

import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from syncfix.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)
from syncfix.models.syncfix import SyncFixConfig, SyncFixModel
from syncfix.models.unets import DiffusersUNet2DCondWrapper
from syncfix.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig

import numpy as np
from syncfix.inference.colmap_read_write_model import qvec2rotmat
from pathlib import Path
from typing import Dict

def camera_center_from_qt(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    COLMAP Image uses world-to-camera:
        X_cam = R * X_world + t
    Camera center in world coordinates:
        C = -R^T * t
    """
    R = qvec2rotmat(qvec)
    return (-R.T @ tvec.reshape(3)).reshape(3)


def build_name_to_center(colmap_images) -> Dict[str, np.ndarray]:
    """
    Map both the stored COLMAP image name and its basename to camera center.
    """
    out: Dict[str, np.ndarray] = {}
    for _, im in colmap_images.items():
        C = camera_center_from_qt(np.asarray(im.qvec), np.asarray(im.tvec))
        name = str(im.name)
        out[name] = C
        out[Path(name).name] = C
    return out

def resolve_center(name: str, name_to_center: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Resolve camera center for a given image name.
    """
    if name in name_to_center:
        return name_to_center[name]
    b = Path(name).name
    if b in name_to_center:
        return name_to_center[b]
    raise KeyError(f"Image '{name}' not found in COLMAP model (tried exact and basename).")


def closest_train_for_eval(
    train_names: List[str],
    eval_names: List[str],
    name_to_center: Dict[str, np.ndarray],
) -> Dict[str, str]:
    """
    Map eval image name -> closest train image name by Euclidean distance of camera centers.
    """
    train_centers = []
    train_ok = []
    for tn in train_names:
        if tn not in name_to_center.keys():
            continue
        C = resolve_center(tn, name_to_center)
        train_centers.append(C)
        train_ok.append(tn)
    train_centers = np.stack(train_centers, axis=0)  # (T,3)

    out: Dict[str, str] = {}
    for en in eval_names:
        if en not in name_to_center.keys():
            print('NO EVAL IMAGE FOUND IN COLMAP')
            continue
        Ce = resolve_center(en, name_to_center)
        d2 = np.sum((train_centers - Ce[None, :]) ** 2, axis=1)
        j = int(np.argmin(d2))
        out[en] = train_ok[j]
    return out

def get_pretrained_model(
    model_dir: str,
    save_dir: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    continue_train: bool = False
) -> SyncFixModel:
    """Download the model from the model directory using either a local path or a path to HuggingFace Hub

    Args:
        model_dir (str): The path to the model directory containing the model weights and config, can be a local path or a path to HuggingFace Hub
        save_dir (Optional[str]): The local path to save the model if downloading from HuggingFace Hub. Defaults to None.
        torch_dtype (torch.dtype): The torch dtype to use for the model. Defaults to torch.bfloat16.
        device (str): The device to use for the model. Defaults to "cuda".

    Returns:
        SyncFixModel: The loaded model
    """
    if not os.path.exists(model_dir):
        local_dir = snapshot_download(
            model_dir,
            local_dir=save_dir,
        )
        model_dir = local_dir

    model_files = os.listdir(model_dir)

    # check yaml config file is present
    yaml_file = [f for f in model_files if f.endswith(".yaml")]
    if len(yaml_file) == 0:
        raise ValueError("No yaml file found in the model directory.")

    # check safetensors weights file is present
    safetensors_files = sorted([f for f in model_files if f.endswith(".safetensors")])
    ckpt_files = sorted([f for f in model_files if f.endswith(".ckpt")])
    if len(safetensors_files) == 0 and len(ckpt_files) == 0:
        raise ValueError("No safetensors or ckpt file found in the model directory")

    if len(model_files) == 0:
        raise ValueError("No model files found in the model directory")

    with open(os.path.join(model_dir, yaml_file[0]), "r") as f:
        config = yaml.safe_load(f)

    model = _get_model_from_config(**config, torch_dtype=torch_dtype)

    if len(safetensors_files) > 0:
        logging.info(f"Loading safetensors file: {safetensors_files[-1]}")
        sd = load_file(os.path.join(model_dir, safetensors_files[-1]))
        model.load_state_dict(sd, strict=False)
    elif len(ckpt_files) > 0:
        logging.info(f"Loading ckpt file: {ckpt_files[-1]}")
        sd = torch.load(
            os.path.join(model_dir, ckpt_files[-1]),
            map_location="cpu",
            weights_only=False
        )["state_dict"]
        sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
        model.load_state_dict(
            sd,
            strict=False,
        )

    if not continue_train:
        model.to(device).to(torch_dtype)
        model.eval()
    else:
        model.to(torch_dtype)

    return model


def _get_model_from_config(
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    timestep_sampling: str = "log_normal",
    selected_timesteps: Optional[List[float]] = None,
    prob: Optional[List[float]] = None,
    conditioning_images_keys: Optional[List[str]] = [],
    conditioning_masks_keys: Optional[List[str]] = [],
    source_key: str = "source_image",
    target_key: str = "source_image_paste",
    bridge_noise_sigma: float = 0.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
    torch_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
):

    conditioners = []

    denoiser = DiffusersUNet2DCondWrapper(
        in_channels=unet_input_channels,  # Add downsampled_image
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

    if conditioning_images_keys != [] or conditioning_masks_keys != []:

        latents_concat_embedder_config = LatentsConcatEmbedderConfig(
            image_keys=conditioning_images_keys,
            mask_keys=conditioning_masks_keys,
        )
        latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
        latent_concat_embedder.freeze()
        conditioners.append(latent_concat_embedder)

        # Wrap conditioners and set to device
    conditioner = ConditionerWrapper(
        conditioners=conditioners,
    )

    ## VAE ##
    # Get VAE model
    vae_config = AutoencoderKLDiffusersConfig(
        version=backbone_signature,
        subfolder="vae",
        tiling_size=(128, 128),
    )
    vae = AutoencoderKLDiffusers(vae_config).to(torch_dtype)
    vae.freeze()
    vae.to(torch_dtype)

    ## Diffusion Model ##
    # Get diffusion model
    config = SyncFixConfig(
        source_key=source_key,
        target_key=target_key,
        latent_loss_weight=latent_loss_weight,
        latent_loss_type=latent_loss_type,
        pixel_loss_type=pixel_loss_type,
        pixel_loss_weight=pixel_loss_weight,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )

    model = SyncFixModel(
        config,
        denoiser=denoiser,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=conditioner,
    ).to(torch_dtype)

    return model
