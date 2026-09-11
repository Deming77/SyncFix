import argparse
import json
import os
from pathlib import Path

from syncfix.metrics import compute_cvsc, load_cvsc_models

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-View Semantic Consistency of a sequence of views."
    )
    parser.add_argument('--image_dir', type=str, required=True, help='Directory of views, scored in sorted filename order')
    parser.add_argument('--dinov3_repo', type=str, required=True, help='Local clone of the DINOv3 repository')
    parser.add_argument('--dinov3_weights', type=str, required=True, help='Path or URL of the DINOv3 weights')
    parser.add_argument('--dinov3_model', type=str, default='dinov3_vitl16', help='DINOv3 hubconf entry point')
    parser.add_argument('--max_keypoints', type=int, default=1024, help='Keypoint budget per image')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the metric on')
    parser.add_argument('--output_json', type=str, default=None, help='Optional path to write the full per-pair results')

    args = parser.parse_args()

    image_paths = sorted(
        str(p) for p in Path(args.image_dir).iterdir() if p.suffix.lower() in EXTS
    )
    if len(image_paths) < 2:
        raise ValueError(f"Need at least two images in {args.image_dir}")

    models = load_cvsc_models(
        dinov3_repo=args.dinov3_repo,
        dinov3_weights=args.dinov3_weights,
        dinov3_model=args.dinov3_model,
        max_num_keypoints=args.max_keypoints,
        device=args.device,
    )

    results = compute_cvsc(image_paths, models)

    print(f"views:        {len(image_paths)}")
    print(f"scored pairs: {results['num_pairs']}")
    print(f"CVSC:         {results['cvsc']:.4f}")
    if results["inlier_ratio"] is not None:
        print(f"inlier ratio: {results['inlier_ratio']:.4f}")

    if args.output_json is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.output_json}")
