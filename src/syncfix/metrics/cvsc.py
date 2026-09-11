"""Cross-View Semantic Consistency (CVSC).

CVSC asks whether two views of a scene agree *semantically* wherever they agree
geometrically. Keypoints are matched with RaCo/ALIKED and LightGlue, the matches are
filtered by a fundamental-matrix RANSAC, and DINOv3 patch features are sampled at the
surviving matches and compared with cosine similarity. A refinement model that
hallucinates different content into each view scores poorly here even when its
per-image PSNR or LPIPS looks healthy.

The metric needs three extra dependencies that are not part of the base install:
`lightglue`, `raco`, and a local clone of the DINOv3 repository together with its
weights. See the Evaluation section of the README.
"""

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

DINOV3_PATCH_SIZE = 16
MIN_MATCHES_FOR_RANSAC = 8


@dataclass
class CVSCModels:
    """The frozen networks CVSC needs, loaded once and reused across pairs."""

    extractor: Any
    descriptor: Any
    matcher: Any
    dinov3: Any
    device: torch.device


def load_cvsc_models(
    dinov3_repo: str,
    dinov3_weights: str,
    dinov3_model: str = "dinov3_vitl16",
    max_num_keypoints: int = 1024,
    device: str = "cuda",
) -> CVSCModels:
    """Load the keypoint matcher and the DINOv3 backbone.

    Args:
        dinov3_repo: Local clone of https://github.com/facebookresearch/dinov3.
        dinov3_weights: Path or URL of the pretrained DINOv3 weights.
        dinov3_model: Entry point in the DINOv3 hubconf, e.g. `dinov3_vitl16`.
        max_num_keypoints: Keypoint budget per image for the RaCo extractor.
        device: Device the networks are moved to.
    """
    from lightglue import ALIKED, LightGlue
    from raco import RaCo

    device = torch.device(device)

    extractor = RaCo(max_num_keypoints=max_num_keypoints).eval().to(device)
    descriptor = ALIKED().eval().to(device)
    matcher = LightGlue(features="raco-aliked").eval().to(device)
    dinov3 = (
        torch.hub.load(
            dinov3_repo, dinov3_model, source="local", weights=dinov3_weights
        )
        .eval()
        .to(device)
    )

    return CVSCModels(extractor, descriptor, matcher, dinov3, device)


def _match_keypoints(
    models: CVSCModels, image_a: Image.Image, image_b: Image.Image
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[float]]:
    """Match two images and keep only the epipolar inliers."""
    from raco.utils import rbd

    to_tensor = T.ToTensor()
    tensor_a = to_tensor(image_a).to(models.device)
    tensor_b = to_tensor(image_b).to(models.device)

    feats_a = models.extractor.extract(tensor_a)
    feats_a["descriptors"] = models.descriptor.describe(feats_a["keypoints"], tensor_a)
    feats_b = models.extractor.extract(tensor_b)
    feats_b["descriptors"] = models.descriptor.describe(feats_b["keypoints"], tensor_b)

    matches01 = models.matcher({"image0": feats_a, "image1": feats_b})
    feats_a, feats_b, matches01 = [rbd(x) for x in (feats_a, feats_b, matches01)]

    matches = matches01["matches"]
    kpts_a = feats_a["keypoints"][matches[..., 0]]
    kpts_b = feats_b["keypoints"][matches[..., 1]]

    if len(kpts_a) == 0:
        return None, None, None

    if len(kpts_a) < MIN_MATCHES_FOR_RANSAC:
        return kpts_a, kpts_b, None

    try:
        _, inliers = cv2.findFundamentalMat(
            kpts_a.cpu().numpy(), kpts_b.cpu().numpy(), cv2.USAC_MAGSAC
        )
    except cv2.error:
        return None, None, None
    if inliers is None:
        return None, None, None

    inliers = (inliers > 0).ravel()
    inlier_ratio = float(inliers.sum() / len(inliers))
    return kpts_a[inliers], kpts_b[inliers], inlier_ratio


def _dinov3_patch_features(models: CVSCModels, image: Image.Image) -> torch.Tensor:
    """Return the DINOv3 patch tokens of an image as a (1, C, H_p, W_p) map."""
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    width, height = image.size
    h_new = (height // DINOV3_PATCH_SIZE) * DINOV3_PATCH_SIZE
    w_new = (width // DINOV3_PATCH_SIZE) * DINOV3_PATCH_SIZE
    resized = image.resize((w_new, h_new), Image.BICUBIC)
    tensor = transform(resized).unsqueeze(0).to(models.device)

    with torch.no_grad():
        tokens = models.dinov3.forward_features(tensor)["x_norm_patchtokens"]

    return tokens.reshape(
        1, h_new // DINOV3_PATCH_SIZE, w_new // DINOV3_PATCH_SIZE, -1
    ).permute(0, 3, 1, 2)


def _normalized_grid(points: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Turn pixel coordinates into the [-1, 1] sampling grid `grid_sample` expects."""
    x = 2.0 * points[:, 0] / (width - 1) - 1.0
    y = 2.0 * points[:, 1] / (height - 1) - 1.0
    return torch.stack([x, y], dim=-1).unsqueeze(0).unsqueeze(0)


def cvsc_pair(
    models: CVSCModels, image_a: Image.Image, image_b: Image.Image
) -> Tuple[Optional[float], Optional[float]]:
    """Score one pair of views.

    Returns the mean cosine similarity of the DINOv3 features at the matched points and
    the RANSAC inlier ratio. Both are None when the pair cannot be matched at all, which
    is itself a sign that the two views disagree.
    """
    kpts_a, kpts_b, inlier_ratio = _match_keypoints(models, image_a, image_b)
    if kpts_a is None or len(kpts_a) == 0:
        return None, inlier_ratio

    feat_a = _dinov3_patch_features(models, image_a)
    feat_b = _dinov3_patch_features(models, image_b)

    width, height = image_a.size
    # The grid is in normalized coordinates, so features may live on a coarser patch
    # grid than the pixels the keypoints were detected in.
    sampled_a = F.grid_sample(
        feat_a,
        _normalized_grid(kpts_a, width, height),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(0).squeeze(1)
    sampled_b = F.grid_sample(
        feat_b,
        _normalized_grid(kpts_b, width, height),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(0).squeeze(1)

    similarity = F.cosine_similarity(sampled_a, sampled_b, dim=0)
    return float(similarity.mean().item()), inlier_ratio


def _safe_mean(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def compute_cvsc(
    image_paths: Sequence[str],
    models: CVSCModels,
    verbose: bool = True,
) -> dict:
    """Average CVSC over the consecutive pairs of an ordered sequence of views.

    Args:
        image_paths: Views in sequence order; pair `i` is (`i`, `i + 1`).
        models: Networks returned by `load_cvsc_models`.

    Returns:
        A dict with the mean `cvsc`, the mean `inlier_ratio`, the number of pairs that
        produced a score, and the per-pair scores.
    """
    if len(image_paths) < 2:
        raise ValueError("CVSC needs at least two views.")

    iterator = range(len(image_paths) - 1)
    if verbose:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="CVSC")

    scores: List[float] = []
    inlier_ratios: List[float] = []
    per_pair = []

    for i in iterator:
        image_a = Image.open(image_paths[i]).convert("RGB")
        image_b = Image.open(image_paths[i + 1]).convert("RGB")

        score, inlier_ratio = cvsc_pair(models, image_a, image_b)
        if score is not None:
            scores.append(score)
        if inlier_ratio is not None:
            inlier_ratios.append(inlier_ratio)

        per_pair.append(
            {
                "view_a": os.path.basename(image_paths[i]),
                "view_b": os.path.basename(image_paths[i + 1]),
                "cvsc": score,
                "inlier_ratio": inlier_ratio,
            }
        )

    return {
        "cvsc": _safe_mean(scores),
        "inlier_ratio": _safe_mean(inlier_ratios),
        "num_pairs": len(scores),
        "per_pair": per_pair,
    }
