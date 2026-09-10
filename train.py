import datetime
import logging
import os
import random
import shutil
from typing import List, Optional

import braceexpand
import fire
import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusionXLPipeline
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torchvision.transforms import InterpolationMode

from syncfix.data.datasets import DataModule, DataModuleConfig
from syncfix.data.filters import KeyFilter, KeyFilterConfig
from syncfix.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
    RescaleMapper,
    RescaleMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)
from syncfix.models.builder import build_conditioner, build_denoiser, build_vae
from syncfix.models.syncfix import SyncFixConfig, SyncFixModel
from syncfix.trainer import TrainingConfig, TrainingPipeline
from syncfix.trainer.loggers import WandbSampleLogger
from syncfix.trainer.utils import StateDictAdapter

logging.basicConfig(level=logging.INFO)

# Resolution the shards are resized to before being fed to the VAE.
TRAIN_SIZE = (540, 960)


def get_model(
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    timestep_sampling: str = "log_normal",
    selected_timesteps: Optional[List[float]] = None,
    prob: Optional[List[float]] = None,
    conditioning_images_keys: Optional[List[str]] = [],
    conditioning_masks_keys: Optional[List[str]] = [],
    source_key: str = "image",
    target_key: str = "target",
    bridge_noise_sigma: float = 0.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
    use_l1_pixel_loss: bool = False,
    use_ssim_loss: bool = False,
    use_gram_loss: bool = False,
    use_ref: bool = False,
    pretrained_path: str = None,
) -> SyncFixModel:
    """Build a SyncFix model, initialized either from SDXL or from a previous run."""

    denoiser = build_denoiser(
        unet_input_channels=unet_input_channels,
        vae_num_channels=vae_num_channels,
        torch_dtype=torch.bfloat16,
    )

    if pretrained_path is None:
        # Adapt the SDXL UNet weights to the (narrower) SyncFix denoiser
        pipe = StableDiffusionXLPipeline.from_pretrained(
            backbone_signature,
            torch_dtype=torch.bfloat16,
        )
        state_dict = pipe.unet.state_dict()

        for key in [
            "add_embedding.linear_1.weight",
            "add_embedding.linear_1.bias",
            "add_embedding.linear_2.weight",
            "add_embedding.linear_2.bias",
        ]:
            del state_dict[key]

        state_dict = StateDictAdapter()(
            model_state_dict=denoiser.state_dict(),
            checkpoint_state_dict=state_dict,
            regex_keys=[
                r"class_embedding.linear_\d+.(weight|bias)",
                r"conv_in.weight",
                r"(down_blocks|up_blocks)\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
                r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
            ],
            strategy="zeros",
        )

        denoiser.load_state_dict(state_dict, strict=True)

        del pipe

    config = SyncFixConfig(
        source_key=source_key,
        target_key=target_key,
        latent_loss_weight=latent_loss_weight,
        latent_loss_type=latent_loss_type,
        pixel_loss_type=pixel_loss_type,
        pixel_loss_weight=pixel_loss_weight,
        use_l1_pixel_loss=use_l1_pixel_loss,
        use_ssim_loss=use_ssim_loss,
        use_gram_loss=use_gram_loss,
        use_ref=use_ref,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    model = SyncFixModel(
        config,
        denoiser=denoiser,
        training_noise_scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
            backbone_signature, subfolder="scheduler"
        ),
        sampling_noise_scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
            backbone_signature, subfolder="scheduler"
        ),
        vae=build_vae(backbone_signature, torch_dtype=torch.bfloat16),
        conditioner=build_conditioner(
            conditioning_images_keys=conditioning_images_keys,
            conditioning_masks_keys=conditioning_masks_keys,
        ),
    ).to(torch.bfloat16)

    if pretrained_path is not None:
        ckpt_files = sorted(
            f for f in os.listdir(pretrained_path) if f.endswith(".ckpt")
        )
        if not ckpt_files:
            raise FileNotFoundError(f"No .ckpt file found in {pretrained_path}")
        logging.info(f"Loading ckpt file: {ckpt_files[-1]}")
        sd = torch.load(
            os.path.join(pretrained_path, ckpt_files[-1]),
            map_location="cpu",
            weights_only=False,
        )["state_dict"]
        sd = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
        model.load_state_dict(sd, strict=False)

    return model


def get_filter_mappers(num_views: int):
    """Filters and mappers turning a paired shard sample into stacked multi-view tensors.

    `TrainMultiViewStage` emits `num_views` degraded/clean pairs (`img{i}.png` /
    `gt{i}.png`) plus a single clean reference (`ref.png`). The reference is appended as
    the last view of both the source and the target stacks, so it stays clean end to end.
    """

    class PackMultiViews:
        def __call__(self, sample: dict) -> dict:
            ref = sample.pop("ref")

            images = [sample.pop(f"image{i + 1}") for i in range(num_views)]
            images.append(ref)
            sample["image"] = torch.stack(images, dim=0)

            targets = [sample.pop(f"target{i + 1}") for i in range(num_views)]
            targets.append(ref)
            sample["target"] = torch.stack(targets, dim=0)

            return sample

    def resize_to_tensor(key: str) -> TorchvisionMapper:
        return TorchvisionMapper(
            TorchvisionMapperConfig(
                key=key,
                transforms=["ToTensor", "Resize"],
                transforms_kwargs=[
                    {},
                    {
                        "size": TRAIN_SIZE,
                        "interpolation": InterpolationMode.NEAREST_EXACT,
                    },
                ],
            )
        )

    keys = ["ref.png"]
    key_map = {"ref.png": "ref"}
    for i in range(1, num_views + 1):
        keys.extend([f"img{i}.png", f"gt{i}.png"])
        key_map.update({f"img{i}.png": f"image{i}", f"gt{i}.png": f"target{i}"})

    mappers = [KeyRenameMapper(KeyRenameMapperConfig(key_map=key_map))]
    mappers.extend(resize_to_tensor(key) for key in key_map.values())
    mappers.extend(RescaleMapper(RescaleMapperConfig(key=key)) for key in key_map.values())
    mappers.append(PackMultiViews())

    return [KeyFilter(KeyFilterConfig(keys=keys)), MapperWrapper(mappers)]


def get_data_module(
    train_shards: List[str],
    validation_shards: List[str],
    batch_size: int,
    pair_num_views: int,
    pair_window: int = 15,
):
    filters_mappers = get_filter_mappers(pair_num_views)

    def unbrace(shards: List[str]) -> List[str]:
        unbraced = []
        for shard in shards:
            unbraced.extend(braceexpand.braceexpand(shard))
        return unbraced

    train_shards = unbrace(train_shards)
    # Shard order is otherwise the scene order on disk, which would correlate batches
    random.Random(42).shuffle(train_shards)

    train_config = DataModuleConfig(
        shards_path_or_urls=train_shards,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=20,
        shuffle_before_split_by_workers_buffer_size=20,
        shuffle_before_filter_mappers_buffer_size=20,
        shuffle_after_filter_mappers_buffer_size=20,
        per_worker_batch_size=batch_size,
        num_workers=5,
        pair_window=pair_window,
        pair_num_views=pair_num_views,
    )

    validation_config = DataModuleConfig(
        shards_path_or_urls=unbrace(validation_shards),
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=10,
        shuffle_before_split_by_workers_buffer_size=10,
        shuffle_before_filter_mappers_buffer_size=10,
        shuffle_after_filter_mappers_buffer_size=10,
        per_worker_batch_size=batch_size,
        num_workers=1,
        pair_window=pair_window,
        pair_num_views=pair_num_views,
    )

    return DataModule(
        train_config=train_config,
        train_filters_mappers=filters_mappers,
        eval_config=validation_config,
        eval_filters_mappers=get_filter_mappers(pair_num_views),
    )


def main(
    train_shards: List[str] = ["pipe:cat path/to/train/shards"],
    validation_shards: List[str] = ["pipe:cat path/to/validation/shards"],
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    source_key: str = "image",
    target_key: str = "target",
    wandb_project: str = "syncfix",
    wandb_offline: bool = False,
    batch_size: int = 2,
    num_steps: List[int] = [1],
    learning_rate: float = 4e-5,
    learning_rate_scheduler: str = None,
    learning_rate_scheduler_kwargs: dict = {},
    optimizer: str = "AdamW",
    optimizer_kwargs: dict = {},
    timestep_sampling: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
    use_l1_pixel_loss: bool = False,
    use_ssim_loss: bool = False,
    use_gram_loss: bool = False,
    use_ref: bool = False,
    selected_timesteps: List[float] = None,
    prob: List[float] = None,
    conditioning_images_keys: Optional[List[str]] = [],
    conditioning_masks_keys: Optional[List[str]] = [],
    config_yaml: dict = None,
    save_ckpt_path: str = "./checkpoints",
    log_interval: int = 1000,
    resume_from_checkpoint: bool = True,
    max_steps: int = 100000,
    bridge_noise_sigma: float = 0.005,
    save_interval: int = 10000,
    save_top_k: int = 3,
    path_config: str = None,
    pretrained_path: str = None,
    pair_num_views: int = 2,
    pair_window: int = 15,
):
    model = get_model(
        backbone_signature=backbone_signature,
        vae_num_channels=vae_num_channels,
        unet_input_channels=unet_input_channels,
        source_key=source_key,
        target_key=target_key,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        pixel_loss_type=pixel_loss_type,
        latent_loss_type=latent_loss_type,
        latent_loss_weight=latent_loss_weight,
        pixel_loss_weight=pixel_loss_weight,
        use_l1_pixel_loss=use_l1_pixel_loss,
        use_ssim_loss=use_ssim_loss,
        use_gram_loss=use_gram_loss,
        use_ref=use_ref,
        selected_timesteps=selected_timesteps,
        prob=prob,
        conditioning_images_keys=conditioning_images_keys,
        conditioning_masks_keys=conditioning_masks_keys,
        bridge_noise_sigma=bridge_noise_sigma,
        pretrained_path=pretrained_path,
    )

    data_module = get_data_module(
        train_shards=train_shards,
        validation_shards=validation_shards,
        batch_size=batch_size,
        pair_num_views=pair_num_views,
        pair_window=pair_window,
    )

    # Only the denoiser is trained: the VAE, LPIPS and VGG are frozen
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=[source_key, target_key],
        trainable_params=["denoiser.*"],
        optimizer_name=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        log_samples_model_kwargs={
            "input_shape": None,
            "num_steps": num_steps,
            "max_samples": 1,
        },
    )

    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)

    pipeline.save_hyperparameters(
        {
            f"embedder_{i}": embedder.config.to_dict()
            for i, embedder in enumerate(model.conditioner.conditioners)
        }
    )
    pipeline.save_hyperparameters(
        {
            "denoiser": model.denoiser.config,
            "vae": model.vae.config.to_dict(),
            "config_yaml": config_yaml,
            "training": training_config.to_dict(),
            "training_noise_scheduler": model.training_noise_scheduler.config,
            "sampling_noise_scheduler": model.sampling_noise_scheduler.config,
        }
    )

    start_ckpt = None
    if resume_from_checkpoint and os.path.exists(
        os.path.join(save_ckpt_path, "last.ckpt")
    ):
        start_ckpt = os.path.join(save_ckpt_path, "last.ckpt")
        logging.info(f"Resuming from checkpoint: {start_ckpt}")

    run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-SyncFix"
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        os.makedirs(f"{save_ckpt_path}/logs/{run_name}", exist_ok=True)
        if path_config is not None:
            # This copy is what `get_pretrained_model` reads back at inference time
            shutil.copy(path_config, f"{save_ckpt_path}/config.yaml")

    devices = torch.cuda.device_count()
    if devices == 0:
        raise RuntimeError("No CUDA GPUs found but accelerator='gpu' was requested.")

    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        default_root_dir="logs",
        logger=loggers.WandbLogger(
            project=wandb_project,
            offline=wandb_offline,
            name=run_name,
            save_dir=save_ckpt_path,
        ),
        callbacks=[
            WandbSampleLogger(log_batch_freq=log_interval),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=save_ckpt_path,
                every_n_train_steps=save_interval,
                save_top_k=save_top_k,
                monitor="step",
                mode="max",
                save_last=True,
            ),
        ],
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        limit_val_batches=1,
        val_check_interval=100000,
        max_steps=max_steps,
    )

    trainer.fit(pipeline, data_module, ckpt_path=start_ckpt)


def main_from_config(path_config: str = None):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    logging.info(
        f"Running main with config: {yaml.dump(config, default_flow_style=False)}"
    )
    main(**config, config_yaml=config, path_config=path_config)


if __name__ == "__main__":
    fire.Fire(main_from_config)
