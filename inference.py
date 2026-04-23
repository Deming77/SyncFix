import os
import imageio
import argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from syncfix.inference import evaluate, get_pretrained_model, evaluate_batch
from pathlib import Path
import torch
from syncfix.inference.colmap_read_write_model import read_images_binary
from syncfix.inference.utils import build_name_to_center, closest_train_for_eval

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image or directory')
    parser.add_argument('--ref_image', type=str, default=None, help='Path to the reference image or directory')
    parser.add_argument('--height', type=int, default=540, help='Height of the input image')
    parser.add_argument('--width', type=int, default=960, help='Width of the input image')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--video', action='store_true', help='If the input is a video')
    parser.add_argument("--n_per_pass", type=int, default=3, help="Number of images processed per forward pass")
    parser.add_argument("--ref_size", type=int, default=1, help="Number of reference images processed per forward pass")
    parser.add_argument('--colmap_path', type=str, default=None, help='if closest training view is desired, provide the colmap path')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    model = get_pretrained_model(args.model_path, torch_dtype=torch.bfloat16, device="cuda")

    exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".PNG", ".JPG"]
    # Load input images
    if os.path.isdir(args.input_image):
        input_images = sorted([
            str(p) for p in Path(args.input_image).iterdir() 
            if p.suffix in exts
        ])
    else:
        input_images = [args.input_image]

    # Load reference images if provided
    if args.colmap_path is not None:
        colmap_images = read_images_binary(str(colmap_path))
        name_to_center = build_name_to_center(colmap_images)
        ref_images = []
        for name in input_images:
            eval_to_ref = closest_train_for_eval(train_images, [name], name_to_center)
            ref_image_name = eval_to_ref[name]
            ref_images.append(os.path.join(args.ref_image, ref_image_name))
    else:
        if args.ref_image is not None:
            if os.path.isdir(args.ref_image):
                ref_images = [
                    str(p) for p in Path(args.ref_image).iterdir() 
                    if p.suffix in exts
                ]
            else:
                ref_images = [args.ref_image]

    ref_images = ref_images[:args.ref_size]

    # Process images
    output_images = []
    n = args.n_per_pass
    for i in tqdm(range(0, len(input_images), n), desc="Processing images"):
        images = [Image.open(img).convert('RGB') for img in input_images[i:i+n]]
        ref_image = [Image.open(img).convert('RGB') for img in ref_images[i:i+n]] if args.ref_image is not None else None
        user_defined_ref_size = 0 if ref_image is None else len(ref_image)
        images.extend(ref_image)
        outs = evaluate_batch(
            model,
            images,
            num_sampling_steps=1,
            resize_hw=(args.height, args.width),
            num_reference_samples=user_defined_ref_size
        )
        output_images.extend(outs)

    # Save outputs
    if args.video:
        # Save as video
        video_path = os.path.join(args.output_dir, "output.mp4")
        writer = imageio.get_writer(video_path, fps=30)
        for output_image in tqdm(output_images, desc="Saving video"):
            writer.append_data(np.array(output_image))
        writer.close()
    else:
        # Save as individual images
        for i, output_image in enumerate(tqdm(output_images, desc="Saving images")):
            output_image.save(os.path.join(args.output_dir, os.path.basename(input_images[i])))