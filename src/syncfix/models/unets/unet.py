from typing import Dict, List, Optional, Union, Any

import torch
from diffusers.models import UNet2DConditionModel, UNet2DModel

from diffusers.models.attention import BasicTransformerBlock, _chunked_feed_forward
from einops import rearrange
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

    # Notice that normalization is always applied before the real computation in the following blocks.
    # 0. Self-Attention
    cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
    num_views = cross_attention_kwargs.pop("num_views", None)
    if num_views is None:
        num_views = hidden_states.shape[0]
    if hidden_states.shape[0] % num_views != 0:
        raise ValueError(
            f"hidden_states batch ({hidden_states.shape[0]}) must be divisible by num_views ({num_views})"
        )
    hidden_states = rearrange(hidden_states, "(b v) n d -> b (v n) d", v=num_views)
    batch_size = hidden_states.shape[0]
    if self.norm_type == "ada_norm":
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.norm_type == "ada_norm_zero":
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
    elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
        norm_hidden_states = self.norm1(hidden_states)
    elif self.norm_type == "ada_norm_continuous":
        norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif self.norm_type == "ada_norm_single":
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
    else:
        raise ValueError("Incorrect norm used")

    if self.pos_embed is not None:
        norm_hidden_states = self.pos_embed(norm_hidden_states)

    # 1. Prepare GLIGEN inputs
    gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )

    if self.norm_type == "ada_norm_zero":
        attn_output = gate_msa.unsqueeze(1) * attn_output
    elif self.norm_type == "ada_norm_single":
        attn_output = gate_msa * attn_output

    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    # 1.2 GLIGEN Control
    if gligen_kwargs is not None:
        hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

    hidden_states = rearrange(hidden_states, "b (v n) d -> (b v) n d", v=num_views)

    # 3. Cross-Attention
    if self.attn2 is not None:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm2(hidden_states, timestep)
        elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm2(hidden_states)
        elif self.norm_type == "ada_norm_single":
            # For PixArt norm2 isn't applied here:
            # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
            norm_hidden_states = hidden_states
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
        else:
            raise ValueError("Incorrect norm")

        if self.pos_embed is not None and self.norm_type != "ada_norm_single":
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states

    # 4. Feed-forward
    # i2vgen doesn't have this norm 🤷‍♂️
    if self.norm_type == "ada_norm_continuous":
        norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif not self.norm_type == "ada_norm_single":
        norm_hidden_states = self.norm3(hidden_states)

    if self.norm_type == "ada_norm_zero":
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    if self.norm_type == "ada_norm_single":
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

    if self._chunk_size is not None:
        # "feed_forward_chunk_size" can be used to save memory
        ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
    else:
        ff_output = self.ff(norm_hidden_states)

    if self.norm_type == "ada_norm_zero":
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    elif self.norm_type == "ada_norm_single":
        ff_output = gate_mlp * ff_output

    hidden_states = ff_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    return hidden_states

BasicTransformerBlock.forward = new_forward

class DiffusersUNet2DWrapper(UNet2DModel):
    """
    Wrapper for the UNet2DModel from diffusers

    See diffusers' UNet2DModel for more details
    """

    def __init__(self, *args, **kwargs):
        UNet2DModel.__init__(self, *args, **kwargs)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        conditioning: Dict[str, torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        """
        The forward pass of the model

        Args:

            sample (torch.Tensor): The input sample
            timesteps (Union[torch.Tensor, float, int]): The number of timesteps
        """
        if conditioning is not None:
            class_labels = conditioning["cond"].get("vector", None)
            concat = conditioning["cond"].get("concat", None)

        else:
            class_labels = None
            concat = None

        if concat is not None:
            sample = torch.cat([sample, concat], dim=1)

        return super().forward(sample, timestep, class_labels).sample

    def freeze(self):
        """
        Freeze the model
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False


class DiffusersUNet2DCondWrapper(UNet2DConditionModel):
    """
    Wrapper for the UNet2DConditionModel from diffusers

    See diffusers' Unet2DConditionModel for more details
    """

    def __init__(self, *args, **kwargs):
        UNet2DConditionModel.__init__(self, *args, **kwargs)
        # BaseModel.__init__(self, config=ModelConfig())

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        conditioning: Dict[str, torch.Tensor],
        num_views: Optional[int] = None,
        ip_adapter_cond_embedding: Optional[List[torch.Tensor]] = None,
        down_block_additional_residuals: torch.Tensor = None,
        mid_block_additional_residual: torch.Tensor = None,
        down_intrablock_additional_residuals: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        """
        The forward pass of the model

        Args:

            sample (torch.Tensor): The input sample
            timesteps (Union[torch.Tensor, float, int]): The number of timesteps
            conditioning (Dict[str, torch.Tensor]): The conditioning data
            down_block_additional_residuals (List[torch.Tensor]): Residuals for the down blocks.
                These residuals typically are used for the controlnet.
            mid_block_additional_residual (List[torch.Tensor]): Residuals for the mid blocks.
                These residuals typically are used for the controlnet.
            down_intrablock_additional_residuals (List[torch.Tensor]): Residuals for the down intrablocks.
                These residuals typically are used for the T2I adapters.middle block outputs. Defaults to False
        """

        assert isinstance(conditioning, dict), "conditionings must be a dictionary"
        # assert "crossattn" in conditioning["cond"], "crossattn must be in conditionings"

        class_labels = conditioning["cond"].get("vector", None)
        crossattn = conditioning["cond"].get("crossattn", None)
        concat = conditioning["cond"].get("concat", None)

        # concat conditioning
        if concat is not None:
            sample = torch.cat([sample, concat], dim=1)

        # down_intrablock_additional_residuals needs to be cloned, since unet will modify it
        if down_intrablock_additional_residuals is not None:
            down_intrablock_additional_residuals_clone = [
                curr_residuals.clone()
                for curr_residuals in down_intrablock_additional_residuals
            ]
        else:
            down_intrablock_additional_residuals_clone = None

        # Check diffusers.models.embeddings.py > MultiIPAdapterImageProjectionLayer > forward() for implementation
        # Exepected format : List[torch.Tensor] of shape (batch_size, num_image_embeds, embed_dim)
        # with length = number of ip_adapters loaded in the ip_adapter_wrapper
        if ip_adapter_cond_embedding is not None:
            added_cond_kwargs = {
                "image_embeds": [
                    ip_adapter_embedding.unsqueeze(1)
                    for ip_adapter_embedding in ip_adapter_cond_embedding
                ]
            }
        else:
            added_cond_kwargs = None

        cross_attention_kwargs = {}
        if num_views is not None:
            cross_attention_kwargs["num_views"] = num_views

        return (
            super()
            .forward(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=crossattn,
                class_labels=class_labels,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals_clone,
            )
            .sample
        )

    def freeze(self):
        """
        Freeze the model
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
