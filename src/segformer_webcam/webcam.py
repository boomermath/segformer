"""Live semantic segmentation with NVIDIA SegFormer on a Mac webcam.

Run it with::

    uv run segformer-webcam
    uv run segformer-webcam --camera 1
    uv run segformer-webcam --list-cameras
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
WINDOW = "SegFormer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run NVIDIA SegFormer on the built-in camera or an attached webcam. "
            "Camera 0 is usually the built-in camera. USB and Continuity cameras "
            "use a higher index."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face model id. "
            f"Default: {DEFAULT_MODEL}. "
            "For street scenes try nvidia/segformer-b0-finetuned-cityscapes-1024-1024."
        ),
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open.")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Print camera indexes that return a frame, then exit.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "mps", "cpu", "cuda"),
        help="auto uses Apple Metal (MPS) when this Mac provides it.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Segmentation overlay opacity, from 0 (video only) to 1 (mask only).",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=6,
        help="Highest camera index to probe with --list-cameras or the n key.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        print("MPS is not available; using CPU.", file=sys.stderr)
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; using CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(name)


def class_palette(n: int) -> np.ndarray:
    """Stable BGR colors, one per class, spread around the hue wheel."""
    colors = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        hue = int(round((i * 137.508) % 180))
        sample = np.uint8([[[hue, 220, 255]]])
        colors[i] = cv2.cvtColor(sample, cv2.COLOR_HSV2BGR)[0, 0]
    return colors


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(camera_error(index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def camera_error(index: int) -> str:
    return (
        f"Could not open camera {index}. On macOS, allow Camera access for "
        "Terminal or Cursor in System Settings → Privacy & Security → Camera. "
        "Then retry, or pick another device with --camera or --list-cameras."
    )


def probe_cameras(max_index: int, width: int, height: int) -> list[tuple[int, int, int]]:
    found: list[tuple[int, int, int]] = []
    for index in range(max_index + 1):
        try:
            cap = open_camera(index, width, height)
        except RuntimeError:
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            frame_h, frame_w = frame.shape[:2]
            found.append((index, frame_w, frame_h))
            print(f"camera {index}: {frame_w}x{frame_h}")
        cap.release()
    if not found:
        print(
            "No camera returned a frame. Grant Camera permission, connect a webcam, "
            "and run --list-cameras again.",
            file=sys.stderr,
        )
    return found


class Segmenter:
    def __init__(self, model_id: str, device: torch.device) -> None:
        self.device = device
        print(f"Loading {model_id} on {device.type}…")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
        self.model.eval()
        self.model.to(self.device)
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.colors = class_palette(len(self.id2label))
        self.predict(np.zeros((128, 128, 3), dtype=np.uint8))

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        encoded = self.processor(images=rgb, return_tensors="pt")
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits
        labels = logits.argmax(dim=1)[0].detach().to("cpu").numpy()
        height, width = frame_bgr.shape[:2]
        return cv2.resize(
            labels.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    def predict_safe(self, frame_bgr: np.ndarray) -> np.ndarray:
        try:
            return self.predict(frame_bgr)
        except RuntimeError as exc:
            if self.device.type != "mps":
                raise
            print(f"MPS inference failed ({exc}). Switching to CPU.", file=sys.stderr)
            self.device = torch.device("cpu")
            self.model.to(self.device)
            return self.predict(frame_bgr)


def draw_hud(image: np.ndarray, lines: list[str]) -> None:
    y = 28
    for line in lines:
        if not line:
            continue
        origin = (12, y)
        cv2.putText(
            image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        y += 26


def top_classes(labels: np.ndarray, id2label: dict[int, str], limit: int = 4) -> str:
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=max(len(id2label), int(flat.max()) + 1))
    total = max(int(flat.size), 1)
    parts: list[str] = []
    for index in np.argsort(counts)[::-1]:
        fraction = counts[index] / total
        if fraction < 0.015:
            break
        parts.append(f"{id2label.get(int(index), str(int(index)))} {fraction:.0%}")
        if len(parts) >= limit:
            break
    return "   ".join(parts)


def next_camera(current: int, max_index: int, width: int, height: int) -> tuple[int, cv2.VideoCapture]:
    last_error = "No camera could be opened."
    for step in range(1, max_index + 2):
        index = (current + step) % (max_index + 1)
        try:
            return index, open_camera(index, width, height)
        except RuntimeError as exc:
            last_error = str(exc)
    raise RuntimeError(last_error)


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        probe_cameras(args.max_index, args.width, args.height)
        return

    segmenter = Segmenter(args.model, resolve_device(args.device))
    camera = args.camera
    cap = open_camera(camera, args.width, args.height)
    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    show_overlay = True
    fps = 0.0
    last = time.perf_counter()
    short_model = args.model.rsplit("/", 1)[-1]

    print("Press q to quit, n for the next camera, o to toggle the overlay, [ ] to change opacity.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print(camera_error(camera), file=sys.stderr)
                break

            now = time.perf_counter()
            elapsed = now - last
            last = now
            if elapsed > 0:
                instant = 1.0 / elapsed
                fps = instant if fps == 0 else (0.9 * fps + 0.1 * instant)

            classes = ""
            if show_overlay and alpha > 0:
                labels = segmenter.predict_safe(frame)
                color = segmenter.colors[labels]
                view = cv2.addWeighted(frame, 1.0 - alpha, color, alpha, 0.0)
                classes = top_classes(labels, segmenter.id2label)
            else:
                view = frame.copy()

            draw_hud(
                view,
                [
                    f"{short_model}    {segmenter.device.type}    {fps:4.1f} fps    cam {camera}",
                    classes if show_overlay else "overlay off",
                    "q quit     n next camera     o overlay     [ ] opacity",
                ],
            )
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("o"):
                show_overlay = not show_overlay
            elif key == ord("["):
                alpha = max(0.0, alpha - 0.1)
            elif key == ord("]"):
                alpha = min(1.0, alpha + 0.1)
            elif key == ord("n"):
                cap.release()
                try:
                    camera, cap = next_camera(camera, args.max_index, args.width, args.height)
                except RuntimeError as exc:
                    print(exc, file=sys.stderr)
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
