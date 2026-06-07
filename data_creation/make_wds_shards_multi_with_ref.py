import argparse
import io
import tarfile
from pathlib import Path
from typing import List, Tuple, Dict
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from colmap_read_write_model import read_images_binary, qvec2rotmat
import json
import numpy as np
from itertools import islice
import random
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}

def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            break
        yield batch

def add_bytes_to_tar(tar: tarfile.TarFile, arcname: str, data: bytes):
    ti = tarfile.TarInfo(name=arcname)
    ti.size = len(data)
    tar.addfile(ti, io.BytesIO(data))

def shuffle_items_by_sequence(items: List[Tuple], seed: int) -> List[Tuple]:
    """
    Groups flat items by sequence ID, shuffles the sequences globally, 
    and flattens them back out so intra-sequence temporal order is maintained.
    """
    groups = defaultdict(list)
    for item in items:
        # The key is ALWAYS the last element in the tuple
        sample_key = item[-1]
        
        # Split off the view/frame index to get the sequence ID
        parts = sample_key.split("__")
        seq_id = "__".join(parts[:-1])
        
        groups[seq_id].append(item)

    # Extract the sequence chunks as a list of lists
    sequences = list(groups.values())
    # Shuffle the sequences
    rng = random.Random(seed)
    rng.shuffle(sequences)
    # Flatten back out to a 1D list for the workers
    flattened_items = [item for seq in sequences for item in seq]
    return flattened_items


def discover_triples(
    pairs_root: Path,
    gtdepth_dir: Path,
    mask_dir: Path,
    gt_subdir: str = "gt",
    degraded_subdir: str = "images",
    depth_subdir: str = "depths",
    depth_suffix: str = "-depth.npz",
) -> List[Tuple[Path, Path, Path, Path, Path, str]]:
    """Discover (degraded_img, gt_img, depth_npz, sample_key).

    Expected directory layout per sample group:
      .../<scene>/K_xx/run_xxx/it_xxxxx/
          images/<frame>.png
          gt/<frame>.png
          depths/<frame>-depth.npz

    sample_key is: f"{scene}__{K_dir}__{run_dir}__{it_tag}__{frame_stem}"
    where frame_stem is the degraded image stem (e.g., "frame_00110").
    """
    triples: List[Tuple[Path, Path, Path, Path, Path, str]] = []

    gtdepth_not_exist = ""
    mask_not_exist = ""

    max_num = 0

    # last_processed_run = None
    # target_it = "it_01000"
    # for degraded_dir in tqdm(sorted(pairs_root.rglob(degraded_subdir))):
    for degraded_dir in tqdm(pairs_root.rglob(degraded_subdir)):

        if not degraded_dir.is_dir():
            continue

        it_dir = degraded_dir.parent
        it_tag = it_dir.name

        # Parse structure: <pairs_root>/<scene>/K_xx/run_xxx/it_xxxxx
        # degraded_dir == .../it_xxxxx/images

        try:
            scene = it_dir.parent.parent.parent.name
            K_dir = it_dir.parent.parent.name
            run_dir = it_dir.parent.name
            run_id = f"{scene}/{K_dir}/{run_dir}"
        except Exception:
            # fallback (should rarely happen)
            scene = it_dir.parent.name
            K_dir = "K"
            run_dir = "run"
            run_id = str(it_dir.parent)

        gt_dir = it_dir / gt_subdir
        depth_dir = it_dir / depth_subdir
        if not gt_dir.exists() or not depth_dir.exists():
            continue
        gtdepth_scene_dir = gtdepth_dir / scene / depth_subdir

        mask_scene_dir = mask_dir / scene / "gaussian_splat" / "skyseg_output_resized"
        mask_scene_dir2 = mask_dir / scene / "skyseg_output_resized"

        if gtdepth_not_exist != "":
            print(f"{str(gtdepth_not_exist)} does not exist")
        
        if mask_not_exist != "":
            print(f"{str(mask_not_exist)} does not exist")

        for f in sorted(degraded_dir.iterdir()):
            if not f.is_file() or f.suffix not in IMG_EXTS:
                continue

            gt = gt_dir / f.name
            if not gt.exists():
                continue

            view = f.stem
            sample_key = f"{scene}__{K_dir}__{run_dir}__{it_tag}__{view}"
            triples.append((f, gt, sample_key))
            max_num += 1

            # if max_num == 1000:
            #     return triples

    return triples


def stable_split_hash(key: str, seed: int) -> int:
    b = f"{seed}|{key}".encode("utf-8")
    h = hashlib.blake2b(b, digest_size=8).digest()
    return int.from_bytes(h, "little")


def split_train_eval(items, train_frac: float, seed: int):
    threshold = int(train_frac * ((1 << 64) - 1))
    train, eval_ = [], []
    for deg, gt, depth, gt_depth, mask, key in items:
        h64 = stable_split_hash(key, seed) & ((1 << 64) - 1)
        (train if h64 <= threshold else eval_).append((deg, gt, depth, gt_depth, mask, key))
    return train, eval_


def pil_to_bytes(img: Image.Image, fmt: str, **kwargs) -> bytes:
    bio = io.BytesIO()
    img.save(bio, format=fmt, **kwargs)
    return bio.getvalue()

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
    if name in name_to_center:
        return name_to_center[name]
    b = Path(name).name
    if b in name_to_center:
        return name_to_center[b]
    raise KeyError(f"Image '{name}' not found in COLMAP model (tried exact and basename).")

def parse_splits(splits_path: Path) -> Tuple[List[str], List[str]]:
    """
    Accepts splits JSON in any of these common forms:
      - {"train":[...], "eval":[...]}
      - {"train":[...], "test":[...]}
      - {"train":[...], "render":[...]}   (treat render as eval/test)
      - {"train":[...], "val":[...]}      (treat val as eval/test)
    """
    with splits_path.open("r") as f:
        s = json.load(f)
    if not isinstance(s, Dict) or "train" not in s:
        raise ValueError(f"{splits_path} must be a JSON object containing 'train' and an eval/test list.")
    train = s["train"]
    if not isinstance(train, List):
        raise ValueError(f"{splits_path}: 'train' must be a list.")
    for k in ("eval", "test", "render", "val"):
        if k in s:
            ev = s[k]
            if not isinstance(ev, List):
                raise ValueError(f"{splits_path}: '{k}' must be a list.")
            return train, ev
    raise ValueError(f"{splits_path}: missing eval/test list (expected one of eval/test/render/val).")

def build_view_maps_from_items(items) -> Dict[str, Tuple[Path, Path]]:
    """
    Build mapping basename -> (degraded_path, gt_path) from discover_triples output.
    If multiple items share the same basename, later ones overwrite (shouldn't happen within a scene).
    """
    m: Dict[str, Tuple[Path, Path]] = {}
    for deg, gt, depth, gt_depth, mask, key in items:
        m[deg.name] = (deg, gt)
    return m

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
            # breakpoint()
            print('NO EVAL IMAGE FOUND IN COLMAP')
            continue
        Ce = resolve_center(en, name_to_center)
        d2 = np.sum((train_centers - Ce[None, :]) ** 2, axis=1)
        j = int(np.argmin(d2))
        out[en] = train_ok[j]
    return out

def find_ref_from_items(
    items,
    json_path: Path,
    colmap_path: Path,
    gtdepth_dir: Path,
    depth_suffix: str = "-depth.npz",
):
    images_dict = {}
    colmap_dict = {}

    results = []
    for i, (deg, gt, key) in enumerate(items):
        scene = deg.parent.parent.parent.parent.parent.name
        K = deg.parent.parent.parent.parent.name
        run = deg.parent.parent.parent.name

        splits = json_path / scene / f'{K}_{run}.json'
        train_names, eval_names = parse_splits(splits)
        if deg.name not in eval_names:
            print(f'{deg.name} not in {str(splits)}')
            continue
        sparse_path1 = colmap_path / scene / 'gaussian_splat' / 'sparse' / '0' / 'images.bin'
        sparse_path2 = colmap_path / scene / 'colmap' / 'sparse' / '0' / 'images.bin'
        sparse_path = ""
        if sparse_path1.exists():
            sparse_path = sparse_path1
        elif sparse_path2.exists():
            sparse_path = sparse_path2
        if sparse_path == "":
            print(f'{scene} cannot find sparse')
            continue
        if scene in colmap_dict.keys():
            name_to_center = colmap_dict[scene]
        else:
            colmap_images = read_images_binary(str(sparse_path))
            name_to_center = build_name_to_center(colmap_images)
            colmap_dict[scene] = name_to_center
        if len(colmap_dict) > 10:
            first_key = next(iter(colmap_dict))
            del colmap_dict[first_key]

        # Closest ref mapping using COLMAP (pose lookup uses basename fallback too)
        eval_to_ref = closest_train_for_eval(train_names, [deg.name], name_to_center)
        ref_path1 = colmap_path / scene / 'gaussian_splat' / 'images_4'
        ref_path2 = colmap_path / scene / 'images_4'
        ref_path = ""
        if ref_path1.exists():
            ref_path = ref_path1
        elif ref_path2.exists():
            ref_path = ref_path2
        if ref_path == "":
            print(f'{scene} cannot find ref images')
            continue
        if deg.name not in eval_to_ref.keys():
            print("image not in eval to ref!!!")
        ref_image_path = ref_path / eval_to_ref[deg.name]
        ref_gtdepth = gtdepth_dir / scene / 'depths' / f"{ref_image_path.stem}{depth_suffix}"
        results.append((deg, gt, ref_image_path, key))
        # results.append((deg, gt, ref_image_path, depth, gt_depth, ref_gtdepth, mask, key))

    return results



def write_pairs_json_from_items(
    items,
    json_path: Path,
    colmap_path: Path,
    prompt: str = "remove degradation",
):
    """
    Use discover_triples() items to decide which images are available and their (noisy, clean) paths.
    Use splits_json to pick train/eval names and COLMAP poses to select ref train view for each eval.

    Output format:
    {
      "train": { "<id>": {"image":..., "target_image":..., "ref_image":..., "prompt":...}, ... },
      "test":  { "<id>": {"image":..., "target_image":..., "ref_image":..., "prompt":...}, ... }
    }
    """
    images_dict = {}
    colmap_dict = {}
    for i, (deg, gt, depth, gt_depth, mask, key) in enumerate(items):
        scene = deg.parent.parent.parent.parent.parent.name
        K = deg.parent.parent.parent.parent.name
        run = deg.parent.parent.parent.name

        splits = json_path / scene / f'{K}_{run}.json'
        train_names, eval_names = parse_splits(splits)
        if deg.name not in eval_names:
            print(f'{deg.name} not in {str(splits)}')
            continue
        sparse_path1 = colmap_path / scene / 'gaussian_splat' / 'sparse' / '0' / 'images.bin'
        sparse_path2 = colmap_path / scene / 'colmap' / 'sparse' / '0' / 'images.bin'
        sparse_path = ""
        if sparse_path1.exists():
            sparse_path = sparse_path1
        elif sparse_path2.exists():
            sparse_path = sparse_path2
        if sparse_path == "":
            print(f'{scene} cannot find sparse')
            continue
        if scene in colmap_dict.keys():
            name_to_center = colmap_dict[scene]
        else:
            colmap_images = read_images_binary(str(sparse_path))
            name_to_center = build_name_to_center(colmap_images)
            colmap_dict[scene] = name_to_center
        if len(colmap_dict) > 10:
            first_key = next(iter(colmap_dict))
            del colmap_dict[first_key]

        # Closest ref mapping using COLMAP (pose lookup uses basename fallback too)
        eval_to_ref = closest_train_for_eval(train_names, [deg.name], name_to_center)
        ref_path1 = colmap_path / scene / 'gaussian_splat' / 'images_4'
        ref_path2 = colmap_path / scene / 'images_4'
        ref_path = ""
        if ref_path1.exists():
            ref_path = ref_path1
        elif ref_path2.exists():
            ref_path = ref_path2
        if ref_path == "":
            print(f'{scene} cannot find ref images')
            continue
        if deg.name not in eval_to_ref.keys():
            print("image not in eval to ref!!!")
        ref_image_path = ref_path / eval_to_ref[deg.name]
        
        images_dict[i] = {
            "image": str(deg.resolve()),
            "target_image": str(gt.resolve()),
            "ref_image": str(ref_image_path.resolve()),
            "prompt": prompt,
        }

    return images_dict


def prepare_sample(
    degraded_path: str,
    gt_path: str,
    ref_path: str,
    depth_path: str,
    gt_depth_path: str,
    ref_depth_path: str,
    mask_path: str,
    sample_key: str,
    raw_bytes: bool,
):
    degraded_path = Path(degraded_path)
    gt_path = Path(gt_path)
    ref_path = Path(ref_path)
    ref_depth_path = Path(ref_depth_path)
    depth_path = Path(depth_path)
    gt_depth_path = Path(gt_depth_path)
    mask_path = Path(mask_path)

    # Depth is always stored as raw bytes (npz)
    depth_bytes = depth_path.read_bytes()
    gt_depth_bytes = gt_depth_path.read_bytes()
    ref_depth_bytes = ref_depth_path.read_bytes()

    if raw_bytes:
        # trust on-disk encoding for images too
        deg_bytes = degraded_path.read_bytes()
        gt_bytes = gt_path.read_bytes()
        ref_bytes = ref_path.read_bytes()
        mask_bytes = mask_path.read_bytes()

        return sample_key, deg_bytes, gt_bytes, ref_bytes, depth_bytes, gt_depth_bytes, ref_depth_bytes, mask_bytes

    # else: decode & re-encode images to PNG to be consistent
    deg_img = Image.open(degraded_path).convert("RGB")
    gt_img = Image.open(gt_path).convert("RGB")
    deg_bytes = pil_to_bytes(deg_img, "PNG")
    gt_bytes = pil_to_bytes(gt_img, "PNG")
    return sample_key, deg_bytes, gt_bytes, depth_bytes

def prepare_sample_ref(
    degraded_path: str,
    gt_path: str,
    ref_path: str,
    sample_key: str,
    raw_bytes: bool,
):
    degraded_path = Path(degraded_path)
    gt_path = Path(gt_path)
    ref_path = Path(ref_path)

    if raw_bytes:
        # trust on-disk encoding
        deg_bytes = degraded_path.read_bytes()
        gt_bytes = gt_path.read_bytes()
        ref_bytes = ref_path.read_bytes()

        # caller must choose correct extension names
        return sample_key, deg_bytes, gt_bytes, ref_bytes

    # else: decode and possibly convert
    deg_img = Image.open(degraded_path).convert("RGB")
    gt_img = Image.open(gt_path).convert("RGB")

    deg_bytes = pil_to_bytes(deg_img, "PNG")
    gt_bytes = pil_to_bytes(gt_img, "PNG")
    return sample_key, deg_bytes, gt_bytes


def write_shards(
    items: List[Tuple[Path, Path, Path, Path, Path, str]],
    out_dir: Path,
    prefix: str,
    samples_per_shard: int,
    raw_bytes: bool,
    num_workers: int,
    stop_at: int
):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not items:
        return -1
    
    def open_shard(i: int):
        p = out_dir / f"{prefix}-{i:06d}.tar"
        print(f"Writing {p}")
        return tarfile.open(p, "w")

    shard_idx = 0
    in_shard = 0
    tar = open_shard(shard_idx)
    BATCH_SIZE = 6000

    def write_one_ref(sample_key: str, deg_bytes: bytes, gt_bytes: bytes, ref_bytes: bytes):
        # degraded image
        add_bytes_to_tar(tar, f"{sample_key}.png", deg_bytes)
        # gt image
        add_bytes_to_tar(tar, f"{sample_key}.gt.png", gt_bytes)
        add_bytes_to_tar(tar, f"{sample_key}.ref.png", ref_bytes)
        
    total_samples_needed = stop_at * samples_per_shard
    limited_items = islice(items, total_samples_needed)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        # 1. Loop through the dataset in safe batches
        for batch in batched(limited_items, BATCH_SIZE):
            futures = []
            
            # 2. Submit only the current batch to the workers
            for deg, gt, ref, key in batch:
                futures.append(
                    ex.submit(prepare_sample_ref, str(deg), str(gt), str(ref), key, raw_bytes)
                )

            # 3. Wait for and write the current batch (maintaining sequence)
            for fut in futures:
                if in_shard >= samples_per_shard:
                    tar.close()
                    shard_idx += 1
                    
                    in_shard = 0
                    tar = open_shard(shard_idx)

                sample_key, deg_bytes, gt_bytes, ref_bytes = fut.result()
                write_one_ref(sample_key, deg_bytes, gt_bytes, ref_bytes)
                in_shard += 1
 

    if tar is not None:
        tar.close()
    return shard_idx


def brace_pattern(out_dir: Path, prefix: str, last_idx: int) -> str:
    return str(out_dir / f"{prefix}-{{000000..{last_idx:06d}}}.tar")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--prefix", type=str, default="pairs")
    ap.add_argument("--samples_per_shard", type=int, default=2000)
    ap.add_argument("--train_frac", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)

    # Acceleration knobs
    ap.add_argument(
        "--raw_bytes",
        action="store_true",
        help="Fastest: do not decode/encode images; just read bytes and store. Requires correct on-disk image formats.",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Parallel preprocess workers (0=auto, 1=disable). Only useful if not --raw_bytes.",
    )

    ap.add_argument("--gt_subdir", type=str, default="gt")
    ap.add_argument("--degraded_subdir", type=str, default="images")
    ap.add_argument("--depth_subdir", type=str, default="depths")
    ap.add_argument(
        "--depth_suffix",
        type=str,
        default="-depth.npz",
        help="Depth filename suffix, e.g. '-depth.npz' for frame_00110-depth.npz",
    )
    ap.add_argument("--gtdepth_dir", type=str, default="gt_depth")
    ap.add_argument("--mask_dir", type=str, default="mask")

    ap.add_argument("--splits_json", type=str, default=None,
                    help="Optional splits file (JSON with train/eval lists). If provided with --out_pairs_json, will write the reference-pairs JSON and exit.")
    ap.add_argument("--out_pairs_json", type=str, default=None,
                    help="If set, write out the pairs JSON in the required format (train/test with ref_image).")
    ap.add_argument("--prompt", type=str, default="remove degradation")

    args = ap.parse_args()

    pairs_root = Path(args.pairs_root).resolve()
    gtdepth_dir = Path(args.gtdepth_dir).resolve()
    mask_dir = Path(args.mask_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    train_dir = out_dir / "train"
    eval_dir = out_dir / "eval"
    train_root = pairs_root / "train"
    eval_root = pairs_root / "eval"

    train_items = discover_triples(
        train_root,
        gtdepth_dir=gtdepth_dir,
        mask_dir=mask_dir,
        gt_subdir=args.gt_subdir,
        degraded_subdir=args.degraded_subdir,
        depth_subdir=args.depth_subdir,
        depth_suffix=args.depth_suffix,
    )

    eval_items = discover_triples(
        eval_root,
        gtdepth_dir=gtdepth_dir,
        mask_dir=mask_dir,
        gt_subdir=args.gt_subdir,
        degraded_subdir=args.degraded_subdir,
        depth_subdir=args.depth_subdir,
        depth_suffix=args.depth_suffix,
    )

    if not train_items:
        raise RuntimeError(f"No (image, gt, depth) triples found under {pairs_root}")

    print(f" Train: {len(train_items)} ")
    print(f"  Eval: {len(eval_items)}")

    if args.out_pairs_json is not None:
        if args.splits_json is None:
            raise ValueError("--out_pairs_json requires --splits_json")
        
        out = {}
        out['train'] = write_pairs_json_from_items(
            items=train_items,
            json_path=Path(args.splits_json).resolve(),
            colmap_path=mask_dir,
            prompt=args.prompt,
        )
        out['test'] = write_pairs_json_from_items(
            items=eval_items,
            json_path=Path(args.splits_json).resolve(),
            colmap_path=mask_dir,
            prompt=args.prompt,
        )
        out_json = Path(args.out_pairs_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote pairs JSON: {out_json} (train={len(out['train'])}, test={len(out['test'])})")

        return
    
    train_items_with_ref = find_ref_from_items(
        items=train_items,
        json_path=Path(args.splits_json).resolve(),
        colmap_path=mask_dir,
        gtdepth_dir=gtdepth_dir,
    )

    eval_items_with_ref = find_ref_from_items(
        items=eval_items,
        json_path=Path(args.splits_json).resolve(),
        colmap_path=mask_dir,
        gtdepth_dir=gtdepth_dir,
    )

    del train_items, eval_items

    print(f" Train: {len(train_items_with_ref)} | Eval: {len(eval_items_with_ref)}")

    print("Shuffling sequences globally while maintaining intra-sequence order...")
    train_items_with_ref = shuffle_items_by_sequence(train_items_with_ref, seed=args.seed)

    last_train = write_shards(
        train_items_with_ref,
        train_dir,
        args.prefix,
        args.samples_per_shard,
        raw_bytes=args.raw_bytes,
        num_workers=args.num_workers if not args.raw_bytes else 1,
        stop_at=14
    )

    last_eval = write_shards(
        eval_items_with_ref,
        eval_dir,
        args.prefix,
        args.samples_per_shard,
        raw_bytes=args.raw_bytes,
        num_workers=args.num_workers if not args.raw_bytes else 1,
        stop_at=30
    )

    print("\nYAML patterns:")
    if last_train >= 0:
        print("TRAIN:", brace_pattern(train_dir, args.prefix, last_train))
    if last_eval >= 0:
        print("EVAL :", brace_pattern(eval_dir, args.prefix, last_eval))


if __name__ == "__main__":
    main()
