import logging
from typing import List, Sequence, Tuple
from PIL import Image

import PIL
import torch
from torchvision.transforms import ToPILImage, ToTensor

from syncfix.models.syncfix import SyncFixModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ASPECT_RATIOS = {
    str(512 / 2048): (512, 2048),
    str(1024 / 1024): (1024, 1024),
    str(2048 / 512): (2048, 512),
    str(896 / 1152): (896, 1152),
    str(1152 / 896): (1152, 896),
    str(512 / 1920): (512, 1920),
    str(640 / 1536): (640, 1536),
    str(768 / 1280): (768, 1280),
    str(1280 / 768): (1280, 768),
    str(1536 / 640): (1536, 640),
    str(1920 / 512): (1920, 512),
}



@torch.no_grad()
def evaluate_batch(
    model: SyncFixModel,
    source_images: Sequence[Image.Image],
    num_sampling_steps: int = 1,
    resize_hw: Tuple[int, int] = (480, 960),
    num_reference_samples: int = 0,
) -> List[Image.Image]:
    """
    Evaluate model on a batch of source images.

    Returns list of generated PIL images, each resized back to its original size.

    Notes:
      - We resize all inputs to a fixed (H,W) for batching.
      - Model is expected to accept a batch dict with key `model.source_key` (typically "source_image").
      - Model.sample is expected to return B images.
      - num_reference_samples is the number of reference images to be sliced after model prediction
    """
    if len(source_images) == 0:
        return []

    # Remember original sizes (PIL size is (W,H))
    orig_sizes = [img.size for img in source_images]

    H, W = resize_hw

    # calculate number of reference images to be sliced after model prediction
    if num_reference_samples != 0:
        ref_size = num_reference_samples
    else:
        ref_size = int(len(source_images)/2)

    tens = []
    for img in source_images:
        img_r = img.resize((W, H))
        t = ToTensor()(img_r) * 2.0 - 1.0  # [-1,1]
        tens.append(t)
    x = torch.stack(tens, dim=0).cuda().to(torch.bfloat16)  # (B,3,H,W)

    batch = {model.source_key: x}
    z_source = model.vae.encode(batch[model.source_key])

    out = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=x.shape[0],
        num_reference_samples=ref_size,
    ).clamp(-1, 1)[:-ref_size, ...] #[:ref_size, ...]

    out = (out.float().cpu() + 1.0) / 2.0
    to_pil = ToPILImage()
    outputs: List[Image.Image] = []
    for b in range(out.shape[0]):
        im = to_pil(out[b])
        ow, oh = orig_sizes[b]
        im = im.resize((ow, oh))
        outputs.append(im)
    return outputs


@torch.no_grad()
def evaluate(
    model: SyncFixModel,
    source_image: PIL.Image.Image,
    num_sampling_steps: int = 1,
):
    """
    Evaluate the model on an image coming from the source distribution and generate a new image from the target distribution.

    Args:
        model (SyncFixModel): The model to evaluate.
        source_image (PIL.Image.Image): The source image to evaluate the model on.
        num_sampling_steps (int): The number of sampling steps to use for the model.

    Returns:
        PIL.Image.Image: The generated image.
    """

    ori_h_bg, ori_w_bg = source_image.size
    # ar_bg = ori_h_bg / ori_w_bg
    # closest_ar_bg = min(ASPECT_RATIOS, key=lambda x: abs(float(x) - ar_bg))
    # source_dimensions = ASPECT_RATIOS[closest_ar_bg]
    source_image = source_image.resize((960, 480))
    img_pasted_tensor = ToTensor()(source_image).unsqueeze(0) * 2 - 1
    batch = {
        "source_image": img_pasted_tensor.cuda().to(torch.bfloat16),
    }
    z_source = model.vae.encode(batch[model.source_key])

    output_image = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    ).clamp(-1, 1)

    output_image = (output_image[0].float().cpu() + 1) / 2
    output_image = ToPILImage()(output_image)
    output_image = output_image.resize((ori_h_bg, ori_w_bg))
    return output_image
