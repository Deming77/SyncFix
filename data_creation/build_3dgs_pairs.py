import argparse
import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib
import re

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass
class RunSpec:
    scene_name: str
    K: int
    run_id: int
    split_json: Path
    model_dir: Path
    out_dir: Path

import os
from pathlib import Path

def prune_out_dir_to_render(out_dir: Path, render_names: List[str]) -> int:
    """
    Delete any files under out_dir (recursively) whose filename contains NONE of the
    render_names as a substring. Keeps only files that match at least one render name.

    Returns: number of deleted files
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return 0

    # fast membership check via substring scan over render_names
    def keep_file(p: Path) -> bool:
        fn = p.name
        # match if the filename contains the render item (e.g., "frame_00110" matches "frame_00110-*.png")
        return any(rn.split('.')[0] in fn for rn in render_names)

    deleted = 0
    # walk all files (not dirs)
    for p in out_dir.rglob("*"):
        if not p.is_file():
            continue
        if not keep_file(p):
            try:
                p.unlink()
                deleted += 1
            except FileNotFoundError:
                pass

    return deleted

def stable_int(s: str, mod: int = 10_000) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % mod

def list_scene_images(scene_dir: Path) -> List[str]:
    """
    Returns image filenames (basename) present in scene_dir/images.
    Assumes official 3DGS-style dataset layout.
    """
    img_dir = scene_dir / "images_4"
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing images/ in scene: {scene_dir}")

    names = []
    for p in img_dir.iterdir():
        if p.is_file() and p.suffix in IMG_EXTS:
            names.append(p.name)
    names.sort()
    if len(names) == 0:
        raise RuntimeError(f"No images found in {img_dir}")
    return names


# Helper function to extract the integer frame number from the filename
def get_frame_idx(filename):
    # This finds the first sequence of digits in the string
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

def make_disjoint_splits(
    image_names: List[str],
    K: int,
    num_runs: int,
    seed: int
) -> List[Dict[str, List[str]]]:
    N = len(image_names)
    need = K * num_runs
    if need > N:
        print(f"Not enough images")
        # raise ValueError(f"Not enough images: need K*num_runs={need}, have N={N}")
        return []

    rng = random.Random(seed)
    perm = image_names[:]
    rng.shuffle(perm)

    splits = []
    for r in range(num_runs):
        train = perm[r*K:(r+1)*K]

        train_indices = {get_frame_idx(name) for name in train}
        invalid_indices = set()
        for t_idx in train_indices:
            for offset in range(-9, 10): 
                invalid_indices.add(t_idx + offset)
        render_candidates = [n for n in image_names if get_frame_idx(n) not in invalid_indices]
        render = [n for i, n in enumerate(render_candidates) if i % 5 == 0]

        splits.append({"train": train, "render": render})

    # sanity: disjoint
    all_train = [n for s in splits for n in s["train"]]
    if len(set(all_train)) != len(all_train):
        raise RuntimeError("Train sets are not disjoint; bug in split logic.")
    return splits


def write_split(split: Dict[str, List[str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2))

def emit_job(jsonl_path: Path, job: Dict):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(job) + "\n")

def run_cmd(cmd: List[str], cwd=None):
    print(" ".join(cmd))
    if cwd is not None:
        subprocess.run(cmd, cwd=str(cwd), check=True)
    else:
        subprocess.run(cmd, check=True)


def copy_gt_images(scene_dir: Path, render_names: List[str], gt_out_dir: Path):
    gt_out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = scene_dir / "images"
    for name in render_names:
        src = img_dir / name
        if not src.exists():
            raise FileNotFoundError(f"GT image missing: {src}")
        shutil.copy2(src, gt_out_dir / name)

def shlex_quote(s: str) -> str:
    # Minimal safe quoting for bash -lc command strings
    import shlex
    return shlex.quote(s)

def chunk_list(xs, n_chunks: int):
    n = len(xs)
    # roughly equal chunks
    base = n // n_chunks
    rem  = n % n_chunks
    out = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        out.append(xs[start:start+size])
        start += size
    return out


def emit_all_jobs(args, scene_dirs, splits_root, out_root, Ks, save_its, jobs_file, overread=False):

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        if scene_name == ".cache" or scene_name == ".huggingface":
            continue
        print(f"\n=== Scene: {scene_name} ===")
        gs_scene_dir = scene_dir / "gaussian_splat"
        if gs_scene_dir.exists():
            scene_dir = gs_scene_dir
        image_names = list_scene_images(scene_dir)
        N = len(image_names)
        print(f"Found {N} images")

        for K in Ks:
            print(f"\n--- K={K}, runs={args.runs_per_K} ---")
            splits = make_disjoint_splits(
                image_names=image_names,
                K=K,
                num_runs=args.runs_per_K,
                seed=args.seed + stable_int(scene_name, 10_000) + K * 999 # hash(scene_name) % 10_000
            )
            if len(splits) != 0:
                for run_id, split in enumerate(splits):
                    run_tag = f"K_{K:02d}/run_{run_id:03d}"
                    split_json = splits_root / scene_name / f"{run_tag.replace('/', '_')}.json"

                    write_split(split, split_json)
                    model_dir = out_root / "_models" / scene_name / run_tag
                    model_dir.mkdir(parents=True, exist_ok=True)

                    logs_dir = out_root / "_logs" / scene_name / run_tag
                    logs_dir.mkdir(parents=True, exist_ok=True)

                    # Train job
                    train_done = out_root / "_done" / scene_name / run_tag / "train.done"
                    train_done.parent.mkdir(parents=True, exist_ok=True)

                    train_cmd = [
                        "python", "train.py",
                        "-s", str(scene_dir),
                        "-m", str(model_dir),
                        "--iterations", str(args.max_iter),
                        "--split_json", str(split_json),
                        "-i", "images_4",
                        "--optimizer_type", "sparse_adam",
                        "--disable_viewer",
                        "--save_iterations",
                    ] + [str(it) for it in save_its]

                    if args.extra_train_args.strip():
                        train_cmd += args.extra_train_args.strip().split()

                    emit_job(jobs_file, {
                        "job_type": "train",
                        "scene": scene_name,
                        "K": K,
                        "run_id": run_id,
                        "cmd": train_cmd,
                        "done_marker": str(train_done),
                        "deps": [],
                        "log_path": str(logs_dir / "train.log"),
                    })

                    for it in save_its:
                        out_dir = out_root / scene_name / f"K_{K:02d}" / f"run_{run_id:03d}" / f"it_{it:05d}"
                        gt_dir = out_dir / "gt"
                        meta_path = out_dir / "meta.json"

                        render_done = out_root / "_done" / scene_name / run_tag / f"render_it_{it:05d}_train.done"
                        render_done.parent.mkdir(parents=True, exist_ok=True)

                        render_cmd = [
                            "python", "render.py",
                            "-m", str(model_dir),
                            "--iteration", str(it),
                            "--split_json", str(split_json),
                            "--out_dir", str(out_dir),
                            "--skip_train",
                            "--render_only",
                        ]
                        if args.extra_render_args.strip():
                            render_cmd += args.extra_render_args.strip().split()

                        post_cmd = [
                            "python", "-c",
                            (
                                "import json, shutil; from pathlib import Path;"
                                f"scene=Path({repr(str(scene_dir))});"
                                f"out_dir=Path({repr(str(out_dir))});"
                                f"gt_dir=Path({repr(str(gt_dir))});"
                                f"gt_dir.mkdir(parents=True, exist_ok=True);"
                                f"split=json.loads(Path({repr(str(split_json))}).read_text());"
                                "img_dir=scene/'images_4';"
                                "[shutil.copy2(img_dir/n, gt_dir/n) for n in split['render']];"
                                "meta={"
                                f"'scene':{repr(scene_name)},'K':{K},'run_id':{run_id},'iteration':{it},"
                                "'train_views':split['train'],'render_views':split['render'],"
                                f"'model_dir':{repr(str(model_dir))},'split_json':{repr(str(split_json))}"
                                "};"
                                f"Path({repr(str(meta_path))}).parent.mkdir(parents=True, exist_ok=True);"
                                f"Path({repr(str(meta_path))}).write_text(json.dumps(meta, indent=2));"
                            )
                        ]

                        emit_job(jobs_file, {
                            "job_type": "render_pair",
                            "scene": scene_name,
                            "K": K,
                            "run_id": run_id,
                            "iteration": it,
                            "cmd": render_cmd,
                            "post_cmd": post_cmd,
                            "done_marker": str(render_done),
                            "deps": [str(train_done)],
                            "log_path": str(logs_dir / f"render_it_{it:05d}.log"),
                        })

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--repo_root", type=str, required=True,
    #                 help="Path to official 3DGS repository root (contains train.py, render.py).")
    ap.add_argument("--scenes_root", type=str, required=True,
                    help="Folder containing multiple scenes in official 3DGS format.")
    ap.add_argument("--out_root", type=str, required=True,
                    help="Output dataset root.")
    ap.add_argument("--splits_root", type=str, default="splits",
                    help="Where to save generated split JSON files (relative to out_root unless absolute).")
    ap.add_argument("--jobs_file", type=str, required=True,
                    help="Path to write jobs.jsonl (will be overwritten).")
    ap.add_argument("--Ks", type=str, required=True,
                    help="Comma-separated sparsity levels, e.g. '8,16,32'")
    ap.add_argument("--runs_per_K", type=int, default=5,
                    help="Number of disjoint runs per sparsity level per scene.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max_iter", type=int, default=30000,
                    help="Train iterations for the sparse model. Must be >= max(save_its).")
    ap.add_argument("--save_its", type=str, default="1000,3000",
                    help="Comma-separated iterations at which to render underfitting degradations.")
    ap.add_argument("--gpu", type=str, default="0")

    # Optional: speed knobs / quality knobs for your environment
    ap.add_argument("--extra_train_args", type=str, default="",
                    help="Extra args appended to train.py call.")
    ap.add_argument("--extra_render_args", type=str, default="",
                    help="Extra args appended to render.py call.")
    args = ap.parse_args()

    # repo_root = Path(args.repo_root).resolve()
    scenes_root = Path(args.scenes_root).resolve()
    out_root = Path(args.out_root).resolve()

    Ks = [int(x.strip()) for x in args.Ks.split(",") if x.strip()]
    save_its = [int(x.strip()) for x in args.save_its.split(",") if x.strip()]
    save_its = sorted(save_its)
    if args.max_iter < save_its[-1]:
        raise ValueError(f"--max_iter must be >= max(save_its)={save_its[-1]}")

    splits_root = Path(args.splits_root)
    if not splits_root.is_absolute():
        splits_root = out_root / splits_root
    splits_root.mkdir(parents=True, exist_ok=True)

    jobs_file = Path(args.jobs_file).resolve()
    if jobs_file.exists():
        jobs_file.unlink()  # overwrite

    # Discover scenes
    scene_dirs = [p for p in scenes_root.iterdir() if p.is_dir() and p.name != ".cache" and p.name != ".huggingface"]
    scene_dirs.sort()
    if not scene_dirs:
        raise RuntimeError(f"No scene subfolders found in {scenes_root}")

    n = len(scene_dirs)
    n_train = int(round(n * 0.9))
    n_eval = n - n_train

    shuffle deterministically, then split
    rng = random.Random(0)  # change seed if you want a different split
    rng.shuffle(scene_dirs)

    train_scene_dirs = scene_dirs[:n_train]
    eval_scene_dirs  = scene_dirs[n_train:]

    # # split train into 3 parts
    # train_parts = chunk_list(train_scene_dirs, 3)

    # # emit 3 job files
    # jobs_files = [
    #     Path(str(args.jobs_file).replace(".jsonl", f"_part{i}.jsonl")).resolve()
    #     for i in range(3)
    # ] 

    # for part_idx, (part_scenes, jf) in enumerate(zip(train_parts, jobs_files)):
    #     if jf.exists():
    #         jf.unlink()  # overwrite per-part file

    #     print(f"[part {part_idx}] scenes={len(part_scenes)} jobs_file={jf}")

    #     emit_all_jobs(
    #         args,
    #         part_scenes,
    #         splits_root,
    #         out_root / "train",  # optional: separate outputs per part
    #         Ks,
    #         save_its,
    #         jf,
    #     )

    emit_all_jobs(args, train_scene_dirs, splits_root, out_root / "train", Ks, save_its, jobs_file, overread=False)
    emit_all_jobs(args, eval_scene_dirs, splits_root, out_root / "eval", Ks, save_its, jobs_file, overread=False)

    print(f"Wrote jobs to: {jobs_file}")
    print("Next: run the wrapper scheduler to execute them on all GPUs.")


if __name__ == "__main__":
    main()
