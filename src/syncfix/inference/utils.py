import logging
import os
from typing import List, Optional

import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from syncfix.models.builder import build_conditioner, build_denoiser, build_vae
from syncfix.models.syncfix import SyncFixConfig, SyncFixModel

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

    denoiser = build_denoiser(
        unet_input_channels=unet_input_channels,
        vae_num_channels=vae_num_channels,
        torch_dtype=torch_dtype,
    )
    conditioner = build_conditioner(
        conditioning_images_keys=conditioning_images_keys,
        conditioning_masks_keys=conditioning_masks_keys,
    )
    vae = build_vae(backbone_signature=backbone_signature, torch_dtype=torch_dtype)

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
