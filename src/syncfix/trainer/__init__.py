"""
Training pipeline for SyncFix.

This module contains the PyTorch Lightning training pipeline, its configuration and
the wandb sample logger used during training.

- :mod:`syncfix.trainer.trainer`: the ``TrainingPipeline`` LightningModule.
- :mod:`syncfix.trainer.training_config`: the configuration for the training pipeline.
- :mod:`syncfix.trainer.loggers`: the callback logging samples to wandb.
- :mod:`syncfix.trainer.utils`: helpers to adapt a checkpoint state dict to the model.

Example
#######

.. code-block:: python

    from pytorch_lightning import Trainer

    from syncfix.data.datasets import DataModule, DataModuleConfig
    from syncfix.trainer import TrainingConfig, TrainingPipeline

    training_config = TrainingConfig(
        learning_rate=4e-5,
        optimizer_name="AdamW",
        trainable_params=["denoiser.*"],
        log_keys=["image", "target"],
        log_samples_model_kwargs={
            "input_shape": None,
            "num_steps": [1],
            "max_samples": 1,
        },
    )

    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)

    data_module = DataModule(
        train_config=DataModuleConfig(
            shards_path_or_urls="your urls or paths",
            decoder="pil",
            per_worker_batch_size=2,
            num_workers=4,
            pair_num_views=3,
        ),
        train_filters_mappers=your_mappers_and_filters,
    )

    trainer = Trainer(accelerator="gpu", devices=1, max_steps=100000)
    trainer.fit(pipeline, data_module)
"""

from .trainer import TrainingPipeline
from .training_config import TrainingConfig

__all__ = ["TrainingPipeline", "TrainingConfig"]
