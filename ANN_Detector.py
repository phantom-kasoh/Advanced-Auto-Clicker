# one_ann_detector.py
# A single ANN (YOLOv1-style) that detects objects *and* a TEXT class in one network.
# From-scratch Keras implementation with simple training/inference utilities.
#
# Quick start (in a venv):
#   pip install tensorflow==2.* opencv-python numpy
#   # prepare dataset in YOLO txt format (one .txt per image)
#   # class list file classes.txt (one class name per line, include 'text' if you want text as a class)
#   # train:
#   #   python -m one_ann_detector --train --images ./images --labels ./labels --classes ./classes.txt --epochs 50
#   # export weights:
#   #   python -m one_ann_detector --export ./models/one_ann.weights.h5
#   # inference usage from another script:
#   #   from one_ann_detector import OneAnnDetector
#   #   det = OneAnnDetector.load("./models/one_ann.weights.h5", classes_path="./classes.txt")
#   #   boxes = det.detect(frame_bgr)

from __future__ import annotations
import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# Utility
# -----------------------------

def read_classes(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    cls: int
    score: float

# -----------------------------
# Model (YOLOv1-lite)
# -----------------------------
class YoloV1Lite(tf.keras.Model):
    def __init__(self, S: int, B: int, C: int):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.backbone = models.Sequential([
            layers.Input(shape=(416, 416, 3)),
            layers.Conv2D(16, 3, strides=1, padding='same', activation='relu'),
            layers.MaxPooling2D(2),  # 208
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),  # 104
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),  # 52
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),  # 26
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),  # 13
            layers.Conv2D(512, 3, padding='same', activation='relu'),
            layers.Conv2D(256, 1, activation='relu'),
        ])
        # Output per cell: B*(x,y,w,h,conf) + C class probabilities
        self.pred = layers.Conv2D(B * 5 + C, 1, padding='same')

    def call(self, x, training=False):
        feat = self.backbone(x, training=training)
        out = self.pred(feat)  # (None, S, S, B*5 + C)
        return out

# -----------------------------
# Loss and training utils
# -----------------------------
class YoloV1Loss(tf.keras.losses.Loss):
    def __init__(self, S: int, B: int, C: int,
                 lambda_coord: float = 5.0, lambda_noobj: float = 0.5):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def call(self, y_true, y_pred):
        # y_true, y_pred shape: (batch, S, S, B*5 + C)
        S, B, C = self.S, self.B, self.C
        # Split predictions
        pred_boxes = y_pred[..., :5*B]  # (b,S,S,5B)
        pred_cls   = y_pred[..., 5*B:]  # (b,S,S,C)

        # true
        true_boxes = y_true[..., :5*B]
        true_cls   = y_true[..., 5*B:]

        # Reshape to (..., B, 5)
        pred_boxes = tf.reshape(pred_boxes, (-1, S, S, B, 5))
        true_boxes = tf.reshape(true_boxes, (-1, S, S, B, 5))

        # The responsible box indicator (object mask) is in true_boxes[...,4]
        obj_mask = true_boxes[..., 4:5]  # (b,S,S,B,1)
        noobj_mask = 1.0 - obj_mask

        # Coord loss (only where objects)
        coord_loss = tf.reduce_sum(obj_mask * tf.square(true_boxes[..., 0:2] - pred_boxes[..., 0:2]))
        # width/height as sqrt like YOLOv1
        coord_loss += tf.reduce_sum(
            obj_mask * tf.square(tf.sqrt(tf.maximum(true_boxes[..., 2:3], 1e-6)) -
                                   tf.sqrt(tf.maximum(pred_boxes[..., 2:3], 1e-6)))
        )
        coord_loss += tf.reduce_sum(
            obj_mask * tf.square(tf.sqrt(tf.maximum(true_boxes[..., 3:4], 1e-6)) -
                                   tf.sqrt(tf.maximum(pred_boxes[..., 3:4], 1e-6)))
        )
        coord_loss *= self.lambda_coord

        # Objectness loss
        obj_loss = tf.reduce_sum(obj_mask * tf.square(true_boxes[..., 4:5] - pred_boxes[..., 4:5]))
        noobj_loss = self.lambda_noobj * tf.reduce_sum(noobj_mask * tf.square(0.0 - pred_boxes[..., 4:5]))

        # Per-cell objectness mask (reduce over B): shape (batch, S, S, 1)
        obj_cell = tf.reduce_max(obj_mask, axis=3)  # (b,S,S,1)

        # Class loss only where there is an object in the cell
        cls_err = tf.reduce_sum(tf.square(true_cls - pred_cls), axis=-1, keepdims=True)  # (b,S,S,1)
        cls_loss = tf.reduce_sum(obj_cell * cls_err)

        total = coord_loss + obj_loss + noobj_loss + cls_loss
        return total

# -----------------------------
# Encoding / Decoding
# -----------------------------

def encode_targets(boxes: List[Tuple[float,float,float,float,int]], S: int, B: int, C: int):
    """
    boxes: list of (cx, cy, w, h, cls) normalized to [0,1] relative to image
    returns target tensor (S,S,B*5 + C)
    We assign each gt to the single cell (i,j) containing its center and to the first box slot.
    """
    target = np.zeros((S, S, B*5 + C), dtype=np.float32)
    for (cx, cy, w, h, cls_id) in boxes:
        i = min(S-1, max(0, int(cy * S)))
        j = min(S-1, max(0, int(cx * S)))
        # local coords relative to cell
        cell_x = cx * S - j
        cell_y = cy * S - i
        # fill first B slot (use slot 0)
        base = 0
        target[i, j, base+0] = cell_x
        target[i, j, base+1] = cell_y
        target[i, j, base+2] = w
        target[i, j, base+3] = h
        target[i, j, base+4] = 1.0  # objectness
        # class one-hot
        target[i, j, B*5 + cls_id] = 1.0
    return target


def decode_predictions(pred: np.ndarray, S: int, B: int, C: int, conf_thresh=0.3) -> List[Tuple[float,float,float,float,int,float]]:
    """ pred: (S,S,B*5 + C) raw; returns list of (x1,y1,x2,y2,cls,score) in normalized [0,1] """
    H = W = S
    out = []
    for i in range(S):
        for j in range(S):
            cell = pred[i, j]
            cls_scores = cell[B*5:]
            cls_id = int(np.argmax(cls_scores))
            cls_prob = float(np.max(cls_scores))
            for b in range(B):
                base = b*5
                tx, ty, tw, th, conf = cell[base:base+5]
                if conf * cls_prob < conf_thresh:
                    continue
                # convert to image coords
                cx = (j + tx) / S
                cy = (i + ty) / S
                w = max(1e-6, tw)
                h = max(1e-6, th)
                x1 = max(0.0, cx - w/2)
                y1 = max(0.0, cy - h/2)
                x2 = min(1.0, cx + w/2)
                y2 = min(1.0, cy + h/2)
                score = float(conf * cls_prob)
                out.append((x1, y1, x2, y2, cls_id, score))
    return out


def nms(dets: List[Tuple[float,float,float,float,int,float]], iou_thresh=0.45) -> List[Tuple[float,float,float,float,int,float]]:
    if not dets:
        return []
    dets = np.array(dets, dtype=np.float32)
    x1, y1, x2, y2, cls, scr = dets[:,0], dets[:,1], dets[:,2], dets[:,3], dets[:,4], dets[:,5]
    keep_all = []
    for c in np.unique(cls):
        idxs = np.where(cls == c)[0]
        s = scr[idxs]
        order = idxs[np.argsort(-s)]
        while order.size > 0:
            i = order[0]
            keep_all.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            area_i = (x2[i]-x1[i])*(y2[i]-y1[i])
            area_j = (x2[order[1:]]-x1[order[1:]])*(y2[order[1:]]-y1[order[1:]])
            iou = inter / (area_i + area_j - inter + 1e-6)
            order = order[1:][iou <= iou_thresh]
    return [tuple(map(float, dets[k])) for k in keep_all]

# -----------------------------
# Data loader (YOLO txt format)
# -----------------------------

def load_yolo_annotation(txt_path: str) -> List[Tuple[float,float,float,float,int]]:
    """ Each line: cls cx cy w h (all normalized 0..1) """
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, 'r', encoding='utf-8') as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            c, cx, cy, w, h = parts
            boxes.append((float(cx), float(cy), float(w), float(h), int(c)))
    return boxes

class YoloDataset(tf.keras.utils.Sequence):
    def __init__(self, images_dir: str, labels_dir: str, classes: List[str], S=13, B=1, batch_size=8):
        self.img_paths = []
        self.lbl_paths = []
        exts = {'.jpg','.jpeg','.png','.bmp'}
        for fn in os.listdir(images_dir):
            if os.path.splitext(fn)[1].lower() in exts:
                self.img_paths.append(os.path.join(images_dir, fn))
                base = os.path.splitext(fn)[0] + '.txt'
                self.lbl_paths.append(os.path.join(labels_dir, base))
        self.S, self.B, self.C = S, B, len(classes)
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.img_paths) / self.batch_size)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.img_paths))
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        inds = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        X = np.zeros((len(inds), 416, 416, 3), dtype=np.float32)
        Y = np.zeros((len(inds), self.S, self.S, self.B*5 + self.C), dtype=np.float32)
        for k, i in enumerate(inds):
            img = cv2.imread(self.img_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            img_resized = cv2.resize(img, (416,416))
            X[k] = img_resized/255.0
            boxes = load_yolo_annotation(self.lbl_paths[i])
            Y[k] = encode_targets(boxes, self.S, self.B, self.C)
        return X, Y

# -----------------------------
# Detector wrapper
# -----------------------------
class OneAnnDetector:
    def __init__(self, classes: List[str], S=13, B=1):
        self.classes = classes
        self.C = len(classes)
        self.S, self.B = S, B
        self.model = YoloV1Lite(S, B, self.C)
        # build model by calling once
        _ = self.model(tf.zeros((1,416,416,3)))

    @classmethod
    def load(cls, weights_path: str, classes_path: str, S=13, B=1) -> 'OneAnnDetector':
        classes = read_classes(classes_path)
        det = cls(classes, S=S, B=B)
        det.model.load_weights(weights_path)
        return det

    def save(self, weights_path: str):
        self.model.save_weights(weights_path)

    def compile(self, lr=1e-3):
        self.loss_fn = YoloV1Loss(self.S, self.B, self.C)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=self.loss_fn)

    def train(self, images_dir: str, labels_dir: str, epochs=50, batch_size=8):
        ds = YoloDataset(images_dir, labels_dir, self.classes, S=self.S, B=self.B, batch_size=batch_size)
        self.compile()
        self.model.fit(ds, epochs=epochs)

    def detect(self, frame_bgr: np.ndarray, conf_thresh=0.35, iou_thresh=0.45) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(img, (416,416)).astype(np.float32)/255.0
        pred = self.model.predict(inp[None,...], verbose=0)[0]  # (S,S,5B+C)
        dets = decode_predictions(pred, self.S, self.B, self.C, conf_thresh)
        dets = nms(dets, iou_thresh)
        out: List[Detection] = []
        for x1,y1,x2,y2,cls,score in dets:
            X1 = int(x1 * w); Y1 = int(y1 * h)
            X2 = int(x2 * w); Y2 = int(y2 * h)
            out.append(Detection(X1,Y1,X2,Y2,int(cls),float(score)))
        return out

# -----------------------------
# CLI for quick train/export
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--images', type=str, default='./images')
    ap.add_argument('--labels', type=str, default='./labels')
    ap.add_argument('--classes', type=str, default='./classes.txt')
    ap.add_argument('--weights', type=str, default='./models/one_ann.weights.h5')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--export', type=str, default=None, help='Path to save weights after load/train')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.weights), exist_ok=True)

    cls_names = read_classes(args.classes)
    det = OneAnnDetector(cls_names)

    if args.train:
        det.train(args.images, args.labels, epochs=args.epochs, batch_size=args.batch)
        det.save(args.weights)
        print(f"Saved weights to {args.weights}")

    if args.export:
        det.model.load_weights(args.weights)
        det.model.save_weights(args.export)
        print(f"Exported weights to {args.export}")

if __name__ == '__main__':
    main()
