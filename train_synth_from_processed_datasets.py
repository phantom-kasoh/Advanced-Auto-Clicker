
"""
train_synth_from_processed_datasets_optimized.py

Goal: drive the ANN detector training fast and crash-resilient on a single workstation
by (1) keeping the GPU fed, (2) using CPU cores for parallel scene generation, and
(3) caching sprite preprocessing in RAM (bounded LRU).

This is a drop-in replacement for your existing
train_synth_from_processed_datasets.py.
"""

from __future__ import annotations

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
from functools import lru_cache

import cv2
import numpy as np
import tensorflow as tf

# expects your existing file in same folder
from ANN_Detector import OneAnnDetector


# ----------------------------
# Hardware / performance config
# ----------------------------

def configure_tensorflow(
    *,
    intra_threads: int = 0,
    inter_threads: int = 0,
    xla: bool = False,
    mixed_precision: bool = False,
    gpu_mem_limit_mb: int = 0,
) -> None:
    """
    Best-effort performance setup:
    - GPU: enable memory growth (avoid full pre-alloc). Optional hard limit.
    - CPU: set thread pools (if provided).
    - Optional XLA and mixed precision.
    """

    # Threads: set BEFORE heavy TF work where possible.
    if intra_threads and intra_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(int(intra_threads))
    if inter_threads and inter_threads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(int(inter_threads))

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            # If TF was already initialized, this can fail; ignore.
            pass

        if gpu_mem_limit_mb and gpu_mem_limit_mb > 0:
            # Limit the first GPU (typical single-GPU workstation).
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_mem_limit_mb)],
                )
            except Exception:
                pass

    if mixed_precision:
        # GTX 730 is old and may not benefit; leave it user-controlled.
        try:
            from tensorflow.keras import mixed_precision as mp
            mp.set_global_policy("mixed_float16")
        except Exception:
            print("[WARN] mixed precision requested but could not be enabled; continuing in float32.")

    if xla:
        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            print("[WARN] XLA requested but could not be enabled.")

    # Print what TF sees (useful sanity check in PyCharm run console)
    devs = tf.config.list_logical_devices()
    gpu_logical = [d for d in devs if d.device_type == "GPU"]
    cpu_logical = [d for d in devs if d.device_type == "CPU"]
    print(f"[TF] logical devices: CPU={len(cpu_logical)} GPU={len(gpu_logical)}")
    if gpu_logical:
        print(f"[TF] GPU device: {gpu_logical[0].name}")
    else:
        print("[TF] No GPU visible to TensorFlow. Training will run on CPU.")


# ----------------------------
# Config / types
# ----------------------------

@dataclass
class Sprite:
    path: str
    cls_id: int

@dataclass
class Placed:
    cx: float
    cy: float
    w: float
    h: float
    cls_id: int


def list_images_rec(root: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dirpath, fn))
    return out


def safe_imread(path: str) -> Optional[np.ndarray]:
    return cv2.imread(path, cv2.IMREAD_COLOR)


# ----------------------------
# Mask + bbox (robust-ish)
# ----------------------------

def make_mask_and_crop(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (cropped_img, cropped_mask01)
    - Otsu split to find foreground
    - falls back to full image if it can't find a sensible foreground
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border = np.concatenate([th[0, :], th[-1, :], th[:, 0], th[:, -1]])
    border_mean = float(border.mean())

    # If border is mostly white, treat white as background; else treat black as background.
    if border_mean > 127:
        mask = (th == 0).astype(np.uint8)
    else:
        mask = (th == 255).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    ys, xs = np.where(mask > 0)
    if len(xs) < 30 or len(ys) < 30:
        full = np.ones((h, w), dtype=np.uint8)
        return img_bgr, full

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    crop_img = img_bgr[y1:y2, x1:x2]
    crop_m = mask[y1:y2, x1:x2]
    if crop_img.size == 0 or crop_m.size == 0:
        full = np.ones((h, w), dtype=np.uint8)
        return img_bgr, full

    return crop_img, crop_m


def random_affine(
    img: np.ndarray,
    mask01: np.ndarray,
    rng: random.Random,
    scale_range=(0.7, 2.4),
    rot_deg=12,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    scale = rng.uniform(*scale_range)
    ang = rng.uniform(-rot_deg, rot_deg)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - w / 2
    M[1, 2] += (nh / 2) - h / 2

    img_w = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    m_w = cv2.warpAffine(mask01 * 255, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    m_w = (m_w > 0).astype(np.uint8)
    return img_w, m_w


def jitter_color(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.astype(np.float32)

    alpha = rng.uniform(0.85, 1.20)  # contrast
    beta = rng.uniform(-20, 20)      # brightness
    out = np.clip(alpha * out + beta, 0, 255)

    hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-8, 8)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + rng.randint(-25, 25), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + rng.randint(-25, 25), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rng.random() < 0.20:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    return out


# ----------------------------
# Scene composition
# ----------------------------

def paste(canvas: np.ndarray, sprite: np.ndarray, mask01: np.ndarray, x: int, y: int) -> None:
    h, w = sprite.shape[:2]
    roi = canvas[y:y + h, x:x + w]
    m = mask01[..., None].astype(np.uint8)
    roi[:] = roi * (1 - m) + sprite * m


def encode_targets_simple(boxes: List[Placed], S: int, B: int, C: int) -> np.ndarray:
    """
    Matches your original one_ann_detector behavior:
    - assigns each gt to the cell containing its center
    - writes into box-slot 0 only
    - one-hot class in that cell
    """
    y = np.zeros((S, S, B * 5 + C), dtype=np.float32)
    for b in boxes:
        i = min(S - 1, max(0, int(b.cy * S)))
        j = min(S - 1, max(0, int(b.cx * S)))

        cell_x = b.cx * S - j
        cell_y = b.cy * S - i

        base = 0  # slot 0
        y[i, j, base + 0] = cell_x
        y[i, j, base + 1] = cell_y
        y[i, j, base + 2] = b.w
        y[i, j, base + 3] = b.h
        y[i, j, base + 4] = 1.0
        y[i, j, B * 5 + b.cls_id] = 1.0
    return y


def build_sprite_loader(max_cache_items: int) -> Callable[[str], Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Returns a path->(cropped_img, cropped_mask01) function with bounded LRU cache.

    This is where your 32 GiB RAM actually matters:
    - it avoids repeated disk reads
    - it avoids repeating Otsu + morphology for sprites every time they are sampled
    """
    max_cache_items = max(0, int(max_cache_items))

    if max_cache_items <= 0:
        # No caching
        def loader(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            img = safe_imread(path)
            if img is None:
                return None
            return make_mask_and_crop(img)
        return loader

    @lru_cache(maxsize=max_cache_items)
    def loader(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        img = safe_imread(path)
        if img is None:
            return None
        return make_mask_and_crop(img)

    return loader


def make_scene(
    rng: random.Random,
    sprites_by_group: Dict[str, List[Sprite]],
    group_probs: Dict[str, float],
    *,
    classes_count: int,
    sprite_loader: Callable[[str], Optional[Tuple[np.ndarray, np.ndarray]]],
    S=13,
    B=1,
    canvas_size=416,
    min_objs=1,
    max_objs=3,
    blank_prob=0.05,
    avoid_cell_collisions=True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, Y):
      X: (416,416,3) float32 0..1
      Y: (S,S,B*5+C)
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    # Background noise/gradient
    if rng.random() < 0.75:
        noise = rng.randint(0, 18)
        canvas[:] = noise
        if rng.random() < 0.50:
            gx = np.linspace(0, rng.randint(0, 35), canvas_size).astype(np.uint8)
            grad = np.tile(gx[None, :], (canvas_size, 1))
            canvas[..., 0] = cv2.add(canvas[..., 0], grad)
            canvas[..., 1] = cv2.add(canvas[..., 1], grad // 2)

    if rng.random() < blank_prob:
        X = canvas.astype(np.float32) / 255.0
        Y = np.zeros((S, S, B * 5 + classes_count), dtype=np.float32)
        return X, Y

    occ = np.zeros((S, S), dtype=np.uint8)
    placed: List[Placed] = []

    k = rng.randint(min_objs, max_objs)
    attempts = 0

    # Keep trying to place until we get k or we hit attempt cap
    while len(placed) < k and attempts < 120:
        attempts += 1

        g = rng.choices(list(group_probs.keys()), weights=list(group_probs.values()), k=1)[0]
        pool = sprites_by_group.get(g, [])
        if not pool:
            continue

        sp = pool[rng.randrange(len(pool))]
        loaded = sprite_loader(sp.path)
        if loaded is None:
            continue

        img, m = loaded
        img, m = random_affine(img, m, rng, scale_range=(0.7, 2.6), rot_deg=12)
        img = jitter_color(img, rng)

        sh, sw = img.shape[:2]
        if sh < 8 or sw < 8 or sh >= canvas_size or sw >= canvas_size:
            continue

        x = rng.randint(0, canvas_size - sw)
        y = rng.randint(0, canvas_size - sh)

        cx = (x + sw / 2) / canvas_size
        cy = (y + sh / 2) / canvas_size
        bw = sw / canvas_size
        bh = sh / canvas_size

        if avoid_cell_collisions:
            ci = min(S - 1, max(0, int(cy * S)))
            cj = min(S - 1, max(0, int(cx * S)))
            if occ[ci, cj] == 1:
                continue
            occ[ci, cj] = 1

        paste(canvas, img, m, x, y)
        placed.append(Placed(cx=cx, cy=cy, w=bw, h=bh, cls_id=sp.cls_id))

    X = canvas.astype(np.float32) / 255.0
    Y = encode_targets_simple(placed, S=S, B=B, C=classes_count)
    return X, Y


# ----------------------------
# Dataset indexing (your folders)
# ----------------------------

def build_class_list_and_sprites(data_root: str, text_mode: str) -> Tuple[List[str], Dict[str, List[Sprite]]]:
    """
    Expects:
      {data_root}/train/colors/<color_name>/*.png ...
      {data_root}/train/shapes/<shape_name>/*.png ...
      {data_root}/train/text/<label>/*.png ...   (EMNIST)
    """
    train_root = os.path.join(data_root, "train")
    colors_root = os.path.join(train_root, "colors")
    shapes_root = os.path.join(train_root, "shapes")
    text_root = os.path.join(train_root, "text")

    color_classes = sorted([d for d in os.listdir(colors_root) if os.path.isdir(os.path.join(colors_root, d))])
    shape_classes = sorted([d for d in os.listdir(shapes_root) if os.path.isdir(os.path.join(shapes_root, d))])

    classes: List[str] = []
    classes += [f"color:{c}" for c in color_classes]
    classes += [f"shape:{s}" for s in shape_classes]

    sprites_by_group: Dict[str, List[Sprite]] = {"colors": [], "shapes": [], "text": []}

    for c in color_classes:
        for p in list_images_rec(os.path.join(colors_root, c)):
            cls_id = classes.index(f"color:{c}")
            sprites_by_group["colors"].append(Sprite(p, cls_id))

    for s in shape_classes:
        for p in list_images_rec(os.path.join(shapes_root, s)):
            cls_id = classes.index(f"shape:{s}")
            sprites_by_group["shapes"].append(Sprite(p, cls_id))

    if text_mode == "region":
        classes.append("text")
        text_cls_id = classes.index("text")
        if os.path.isdir(text_root):
            for label_dir in sorted(os.listdir(text_root)):
                full = os.path.join(text_root, label_dir)
                if not os.path.isdir(full):
                    continue
                for p in list_images_rec(full):
                    sprites_by_group["text"].append(Sprite(p, text_cls_id))
    else:
        if not os.path.isdir(text_root):
            raise RuntimeError(f"Missing text root: {text_root}")
        label_dirs = sorted([d for d in os.listdir(text_root) if os.path.isdir(os.path.join(text_root, d))])
        for lab in label_dirs:
            classes.append(f"char:{lab}")
        for lab in label_dirs:
            cls_id = classes.index(f"char:{lab}")
            for p in list_images_rec(os.path.join(text_root, lab)):
                sprites_by_group["text"].append(Sprite(p, cls_id))

    return classes, sprites_by_group


# ----------------------------
# tf.data generator (parallel, GPU-feeding)
# ----------------------------

def make_tf_dataset_parallel(
    *,
    rng_seed: int,
    sprites_by_group: Dict[str, List[Sprite]],
    classes_count: int,
    sprite_loader: Callable[[str], Optional[Tuple[np.ndarray, np.ndarray]]],
    S=13,
    B=1,
    batch_size=8,
    min_objs=1,
    max_objs=3,
    probs=(0.33, 0.33, 0.34),
    blank_prob=0.05,
    parallel_calls: int = 8,
    prefetch: int = 0,
    private_threadpool_size: int = 0,
) -> tf.data.Dataset:
    """
    Key change vs your original:
    - Instead of a single Python generator thread, we generate examples in parallel
      using tf.numpy_function + Dataset.map(num_parallel_calls=...).

    This is usually the biggest speedup on mid-range/old GPUs: keep the GPU busy.
    """
    group_probs = {"colors": probs[0], "shapes": probs[1], "text": probs[2]}

    # Python side function invoked by tf.numpy_function
    def _py_make_one(i_np: np.ndarray):
        i = int(i_np)
        rng = random.Random(rng_seed + i)
        X, Y = make_scene(
            rng,
            sprites_by_group,
            group_probs,
            classes_count=classes_count,
            sprite_loader=sprite_loader,
            S=S,
            B=B,
            min_objs=min_objs,
            max_objs=max_objs,
            blank_prob=blank_prob,
            avoid_cell_collisions=True,
        )
        return X.astype(np.float32), Y.astype(np.float32)

    def _tf_make_one(i: tf.Tensor):
        x, y = tf.numpy_function(_py_make_one, [i], [tf.float32, tf.float32])
        x.set_shape((416, 416, 3))
        y.set_shape((S, S, B * 5 + classes_count))
        return x, y

    # Infinite stream of indices -> deterministic per-example RNG via seed+index
    ds = tf.data.Dataset.range(2**63 - 1)
    ds = ds.map(_tf_make_one, num_parallel_calls=max(1, int(parallel_calls)), deterministic=False)
    ds = ds.batch(batch_size, drop_remainder=True)

    # Dataset options: allow dedicated pool for Python scene generation.
    options = tf.data.Options()
    if private_threadpool_size and private_threadpool_size > 0:
        options.threading.private_threadpool_size = int(private_threadpool_size)
    options.experimental_deterministic = False
    ds = ds.with_options(options)

    if prefetch and prefetch > 0:
        ds = ds.prefetch(int(prefetch))
    else:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ----------------------------
# Main train
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", required=True, help="Path to your 'data' folder that contains train/ and test/")
    ap.add_argument("--text_mode", choices=["region", "byclass"], default="region")

    ap.add_argument("--weights_out", default="./models/one_ann.weights.h5")
    ap.add_argument("--classes_out", default="./classes.txt")

    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800)

    ap.add_argument("--S", type=int, default=13)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)

    # curriculum
    ap.add_argument("--min_objs", type=int, default=1)
    ap.add_argument("--max_objs", type=int, default=3)
    ap.add_argument("--blank_prob", type=float, default=0.05)

    # sampling probs: colors, shapes, text
    ap.add_argument("--p_colors", type=float, default=0.33)
    ap.add_argument("--p_shapes", type=float, default=0.33)
    ap.add_argument("--p_text", type=float, default=0.34)

    # performance knobs (use your 32 GiB RAM + CPU cores)
    ap.add_argument("--sprite_cache", type=int, default=4000, help="LRU cache size for preprocessed sprites (path->crop+mask).")
    ap.add_argument("--parallel_calls", type=int, default=0, help="How many parallel scene-generation calls (0=auto).")
    ap.add_argument("--prefetch", type=int, default=0, help="Prefetch batches (0=AUTOTUNE).")
    ap.add_argument("--tfdata_threadpool", type=int, default=0, help="Private threadpool size for tf.data (0=default).")

    # TF runtime
    ap.add_argument("--intra_threads", type=int, default=0, help="TF intra-op threads (0=default).")
    ap.add_argument("--inter_threads", type=int, default=0, help="TF inter-op threads (0=default).")
    ap.add_argument("--xla", action="store_true", help="Enable XLA JIT (best-effort).")
    ap.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision (not recommended for GTX 730).")
    ap.add_argument("--gpu_mem_limit_mb", type=int, default=0, help="Optional GPU memory limit in MiB (0=no limit).")

    args = ap.parse_args()

    # Reasonable defaults when user didn't specify.
    cpu_count = os.cpu_count() or 8
    if args.parallel_calls <= 0:
        # Scene generation is Python+OpenCV, so more threads helps until disk/cache saturates.
        args.parallel_calls = min(12, max(2, cpu_count // 2))

    configure_tensorflow(
        intra_threads=args.intra_threads,
        inter_threads=args.inter_threads,
        xla=args.xla,
        mixed_precision=args.mixed_precision,
        gpu_mem_limit_mb=args.gpu_mem_limit_mb,
    )

    os.makedirs(os.path.dirname(args.weights_out) or ".", exist_ok=True)

    classes, sprites_by_group = build_class_list_and_sprites(args.data_root, args.text_mode)
    C = len(classes)

    # write classes.txt
    with open(args.classes_out, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")
    print(f"[OK] Wrote classes file: {args.classes_out}  (C={C})")

    print(
        "[INFO] sprites: "
        f"colors={len(sprites_by_group['colors'])}, "
        f"shapes={len(sprites_by_group['shapes'])}, "
        f"text={len(sprites_by_group['text'])}"
    )

    sprite_loader = build_sprite_loader(args.sprite_cache)

    # curriculum: increase clutter over time
    q = max(1, args.epochs // 4)
    phases = [
        (q, 1, 1),
        (q, 1, 2),
        (q, 1, max(2, args.max_objs)),
        (args.epochs - 3 * q, 1, args.max_objs),
    ]

    det = OneAnnDetector(classes, S=args.S, B=args.B)
    det.compile(lr=args.lr)

    models_dir = os.path.dirname(os.path.abspath(args.weights_out)) or "."
    os.makedirs(models_dir, exist_ok=True)

    best_path = args.weights_out
    latest_epoch_path = os.path.join(models_dir, "latest_epoch.weights.h5")
    latest_step_path = os.path.join(models_dir, "latest_step.weights.h5")
    log_path = os.path.join(models_dir, "train_log.csv")

    # --- Auto-resume (newest first) ---
    for p in (latest_step_path, latest_epoch_path, best_path):
        if os.path.exists(p):
            det.model.load_weights(p)
            print(f"[RESUME] Loaded weights from: {p}")
            break

    callbacks: List[tf.keras.callbacks.Callback] = []

    # --- Crash-safe resume (includes optimizer state) if your TF has it ---
    if hasattr(tf.keras.callbacks, "BackupAndRestore"):
        callbacks.append(
            tf.keras.callbacks.BackupAndRestore(backup_dir=os.path.join(models_dir, "backup_state"))
        )

    # --- Always save latest (epoch) ---
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            latest_epoch_path,
            save_weights_only=True,
            save_best_only=False,
            save_freq="epoch",
            verbose=0,
        )
    )

    # --- Save latest periodically inside the epoch (so a crash costs minutes, not an epoch) ---
    save_every_batches = max(50, args.steps // 4)
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            latest_step_path,
            save_weights_only=True,
            save_best_only=False,
            save_freq=save_every_batches,
            verbose=0,
        )
    )

    # --- Save best ---
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            best_path,
            save_weights_only=True,
            monitor="loss",
            save_best_only=True,
            verbose=1,
        )
    )

    callbacks.append(tf.keras.callbacks.CSVLogger(log_path, append=True))
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, verbose=1))

    total_done = 0
    for (e, mn, mx) in phases:
        if e <= 0:
            continue
        total_done += e
        print(f"\n=== Phase: epochs={e}, objects=[{mn}..{mx}]  (total so far {total_done}/{args.epochs}) ===")

        train_ds = make_tf_dataset_parallel(
            rng_seed=1337 + total_done,
            sprites_by_group=sprites_by_group,
            classes_count=C,
            sprite_loader=sprite_loader,
            S=args.S,
            B=args.B,
            batch_size=args.batch,
            min_objs=mn,
            max_objs=mx,
            probs=(args.p_colors, args.p_shapes, args.p_text),
            blank_prob=args.blank_prob,
            parallel_calls=args.parallel_calls,
            prefetch=args.prefetch,
            private_threadpool_size=args.tfdata_threadpool,
        )

        try:
            det.model.fit(
                train_ds,
                epochs=e,
                steps_per_epoch=args.steps,  # critical for infinite datasets
                callbacks=callbacks,
                verbose=1,
            )
        except KeyboardInterrupt:
            print("[INTERRUPT] Saving latest weights before exiting...")
            det.save(latest_epoch_path)
            raise

    det.save(args.weights_out)
    print(f"[OK] Saved weights: {args.weights_out}")


if __name__ == "__main__":
    main()
