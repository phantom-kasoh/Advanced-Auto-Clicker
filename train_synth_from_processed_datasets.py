import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import tensorflow as tf

# expects your existing file in same folder:
# from one_ann_detector import OneAnnDetector, encode_targets
from ANN_Detector import OneAnnDetector

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
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dirpath, fn))
    return out

def safe_imread(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img

# ----------------------------
# Mask + bbox (robust-ish)
# ----------------------------
def make_mask_and_crop(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (cropped_img, cropped_mask01)
    - tries Otsu split to find foreground
    - falls back to full image if it can't find a sensible foreground
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Otsu threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # decide which side is background by looking at borders
    border = np.concatenate([th[0, :], th[-1, :], th[:, 0], th[:, -1]])
    border_mean = float(border.mean())
    # if border is mostly white, treat white as background
    if border_mean > 127:
        mask = (th == 0).astype(np.uint8)
    else:
        mask = (th == 255).astype(np.uint8)

    # clean up speckle
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    ys, xs = np.where(mask > 0)
    if len(xs) < 30 or len(ys) < 30:
        # too small -> just full sprite
        full = np.ones((h, w), dtype=np.uint8)
        return img_bgr, full

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    # sanity clamp
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    crop_img = img_bgr[y1:y2, x1:x2]
    crop_m = mask[y1:y2, x1:x2]
    if crop_img.size == 0 or crop_m.size == 0:
        full = np.ones((h, w), dtype=np.uint8)
        return img_bgr, full

    return crop_img, crop_m

def random_affine(img: np.ndarray, mask01: np.ndarray, rng: random.Random,
                  scale_range=(0.7, 2.4), rot_deg=12) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    scale = rng.uniform(*scale_range)
    ang = rng.uniform(-rot_deg, rot_deg)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)

    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - w / 2
    M[1, 2] += (nh / 2) - h / 2

    img_w = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    m_w = cv2.warpAffine(mask01 * 255, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    m_w = (m_w > 0).astype(np.uint8)
    return img_w, m_w

def jitter_color(img: np.ndarray, rng: random.Random) -> np.ndarray:
    # mild brightness/contrast + hsv jitter
    out = img.astype(np.float32)

    # brightness/contrast
    alpha = rng.uniform(0.85, 1.20)  # contrast
    beta = rng.uniform(-20, 20)      # brightness
    out = np.clip(alpha * out + beta, 0, 255)

    # hsv jitter
    hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-8, 8)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + rng.randint(-25, 25), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + rng.randint(-25, 25), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # occasional blur
    if rng.random() < 0.20:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    return out

# ----------------------------
# Scene composition
# ----------------------------
def paste(canvas: np.ndarray, sprite: np.ndarray, mask01: np.ndarray, x: int, y: int):
    h, w = sprite.shape[:2]
    roi = canvas[y:y+h, x:x+w]
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

def make_scene(rng: random.Random,
               sprites_by_group: Dict[str, List[Sprite]],
               group_probs: Dict[str, float],
               *,
               classes_count: int,
               S=13,
               B=1,
               canvas_size=416,
               min_objs=1,
               max_objs=3,
               blank_prob=0.05,
               avoid_cell_collisions=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, Y) where:
      X: (416,416,3) float32 0..1
      Y: (S,S,B*5+C)
    """
    # background (noise / gradient)
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    if rng.random() < 0.75:
        noise = rng.randint(0, 18)
        canvas[:] = noise

        # add a weak gradient sometimes
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
    while len(placed) < k and attempts < 80:
        attempts += 1

        # choose group
        g = rng.choices(list(group_probs.keys()), weights=list(group_probs.values()), k=1)[0]
        pool = sprites_by_group[g]
        if not pool:
            continue

        sp = pool[rng.randrange(len(pool))]
        img = safe_imread(sp.path)
        if img is None:
            continue

        img, m = make_mask_and_crop(img)
        img, m = random_affine(img, m, rng, scale_range=(0.7, 2.6), rot_deg=12)
        img = jitter_color(img, rng)

        sh, sw = img.shape[:2]
        if sh < 8 or sw < 8 or sh >= canvas_size or sw >= canvas_size:
            continue

        # place
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

    # class names from folder names
    color_classes = sorted([d for d in os.listdir(colors_root) if os.path.isdir(os.path.join(colors_root, d))])
    shape_classes = sorted([d for d in os.listdir(shapes_root) if os.path.isdir(os.path.join(shapes_root, d))])

    classes: List[str] = []
    classes += [f"color:{c}" for c in color_classes]
    classes += [f"shape:{s}" for s in shape_classes]

    sprites_by_group: Dict[str, List[Sprite]] = {"colors": [], "shapes": [], "text": []}

    # colors sprites
    color_id = {c: i for i, c in enumerate(color_classes)}
    for c in color_classes:
        for p in list_images_rec(os.path.join(colors_root, c)):
            cls_id = classes.index(f"color:{c}")  # stable
            sprites_by_group["colors"].append(Sprite(p, cls_id))

    # shapes sprites
    for s in shape_classes:
        for p in list_images_rec(os.path.join(shapes_root, s)):
            cls_id = classes.index(f"shape:{s}")
            sprites_by_group["shapes"].append(Sprite(p, cls_id))

    # text sprites
    if text_mode == "region":
        # one detector class for all text (recommended)
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
        # detect each EMNIST label as its own class
        if not os.path.isdir(text_root):
            raise RuntimeError(f"Missing text root: {text_root}")
        label_dirs = sorted([d for d in os.listdir(text_root) if os.path.isdir(os.path.join(text_root, d))])
        for lab in label_dirs:
            cname = f"char:{lab}"
            classes.append(cname)
        for lab in label_dirs:
            cls_id = classes.index(f"char:{lab}")
            for p in list_images_rec(os.path.join(text_root, lab)):
                sprites_by_group["text"].append(Sprite(p, cls_id))

    return classes, sprites_by_group

# ----------------------------
# tf.data generator
# ----------------------------
def make_tf_dataset(rng_seed: int,
                    sprites_by_group: Dict[str, List[Sprite]],
                    classes_count: int,
                    *,
                    S=13, B=1,
                    batch_size=8,
                    steps_per_epoch=500,
                    min_objs=1, max_objs=3,
                    probs=(0.33, 0.33, 0.34),
                    blank_prob=0.05):
    rng = random.Random(rng_seed)
    group_probs = {"colors": probs[0], "shapes": probs[1], "text": probs[2]}

    def gen():
        while True:
            X, Y = make_scene(
                rng,
                sprites_by_group,
                group_probs,
                classes_count=classes_count,
                S=S, B=B,
                min_objs=min_objs, max_objs=max_objs,
                blank_prob=blank_prob,
                avoid_cell_collisions=True
            )
            yield X, Y

    out_sig = (
        tf.TensorSpec(shape=(416, 416, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(S, S, B * 5 + classes_count), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)
    ds = ds.batch(batch_size, drop_remainder=True)
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

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.weights_out), exist_ok=True)

    classes, sprites_by_group = build_class_list_and_sprites(args.data_root, args.text_mode)
    C = len(classes)

    # write classes.txt
    with open(args.classes_out, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")
    print(f"[OK] Wrote classes file: {args.classes_out}  (C={C})")

    print(f"[INFO] sprites: colors={len(sprites_by_group['colors'])}, shapes={len(sprites_by_group['shapes'])}, text={len(sprites_by_group['text'])}")

    # build datasets
    train_ds = make_tf_dataset(
        rng_seed=1337,
        sprites_by_group=sprites_by_group,
        classes_count=C,
        S=args.S, B=args.B,
        batch_size=args.batch,
        steps_per_epoch=args.steps,
        min_objs=args.min_objs,
        max_objs=args.max_objs,
        probs=(args.p_colors, args.p_shapes, args.p_text),
        blank_prob=args.blank_prob,
    )

    # curriculum: increase clutter over time
    # (we do it manually by re-fitting in phases)
    phases = [
        (max(1, args.epochs // 4), 1, 1),
        (max(1, args.epochs // 4), 1, 2),
        (max(1, args.epochs // 4), 1, max(2, args.max_objs)),
        (args.epochs - 3 * max(1, args.epochs // 4), 1, args.max_objs),
    ]

    det = OneAnnDetector(classes, S=args.S, B=args.B)
    det.compile(lr=args.lr)

    det = OneAnnDetector(classes, S=args.S, B=args.B)
    det.compile(lr=args.lr)

    models_dir = os.path.dirname(os.path.abspath(args.weights_out)) or "."
    os.makedirs(models_dir, exist_ok=True)

    best_path = args.weights_out  # keep your CLI arg as the "best" file
    latest_epoch_path = os.path.join(models_dir, "latest_epoch.weights.h5")
    latest_step_path = os.path.join(models_dir, "latest_step.weights.h5")
    log_path = os.path.join(models_dir, "train_log.csv")

    # --- Auto-resume (newest first) ---
    for p in (latest_step_path, latest_epoch_path, best_path):
        if os.path.exists(p):
            det.model.load_weights(p)
            print(f"[RESUME] Loaded weights from: {p}")
            break

    callbacks = []

    # --- Crash-safe resume (includes optimizer state) if your TF has it ---
    if hasattr(tf.keras.callbacks, "BackupAndRestore"):
        callbacks.append(tf.keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(models_dir, "backup_state")
        ))

    # --- Always save latest (epoch) ---
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        latest_epoch_path,
        save_weights_only=True,
        save_best_only=False,
        save_freq="epoch",
        verbose=0,
    ))

    # --- Save latest periodically inside the epoch (so a crash costs minutes, not an epoch) ---
    save_every_batches = max(50, args.steps // 4)  # tweak if you want
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        latest_step_path,
        save_weights_only=True,
        save_best_only=False,
        save_freq=save_every_batches,
        verbose=0,
    ))

    # --- Save best (what you already had) ---
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        best_path,
        save_weights_only=True,
        monitor="loss",
        save_best_only=True,
        verbose=1,
    ))

    callbacks.append(tf.keras.callbacks.CSVLogger(log_path, append=True))
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())

    lr_sched = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=3, verbose=1
    )
    callbacks.append(lr_sched)

    total_done = 0
    for (e, mn, mx) in phases:
        if e <= 0:
            continue
        total_done += e
        print(f"\n=== Phase: epochs={e}, objects=[{mn}..{mx}]  (total so far {total_done}/{args.epochs}) ===")
        train_ds_phase = make_tf_dataset(
            rng_seed=1337 + total_done,
            sprites_by_group=sprites_by_group,
            classes_count=C,
            S=args.S, B=args.B,
            batch_size=args.batch,
            steps_per_epoch=args.steps,
            min_objs=mn,
            max_objs=mx,
            probs=(args.p_colors, args.p_shapes, args.p_text),
            blank_prob=args.blank_prob,
        )
        try:
            det.model.fit(
                train_ds_phase,
                epochs=e,
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
