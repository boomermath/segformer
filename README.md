# SegFormer webcam

Live semantic segmentation on a Mac, using NVIDIA SegFormer. The built-in camera or an attached webcam is painted with a class overlay and the largest classes in the frame.

The default model is [`nvidia/segformer-b0-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) (ADE20K, 150 everyday scene classes). Inference uses Apple Metal when available, then CUDA, then CPU.

## Requirements

- macOS on Apple Silicon
- Python 3.14 (this repo pins it with `mise` and `.python-version`)
- [uv](https://docs.astral.sh/uv/)
- A built-in or attached webcam

## Setup

```bash
uv sync
```

The first run also downloads the model weights from Hugging Face. Later runs use the local cache.

## Run

```bash
uv run segformer-webcam
```

Camera 0 is usually the built-in camera. List devices, then pick one:

```bash
uv run segformer-webcam --list-cameras
uv run segformer-webcam --camera 1
```

macOS must allow Camera access for the app you launch from (Terminal, iTerm, or Cursor). If the window never opens, enable it in System Settings → Privacy & Security → Camera and run the command again.

### Keys

| Key | Action |
| --- | --- |
| `q` or Esc | Quit |
| `n` | Next camera |
| `o` | Toggle the overlay |
| `[` / `]` | Decrease or increase overlay opacity |

## Options

```text
uv run segformer-webcam --help
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--model` | `nvidia/segformer-b0-finetuned-ade-512-512` | Hugging Face model id |
| `--camera` | `0` | Camera index |
| `--list-cameras` | | Print indexes that return a frame, then exit |
| `--device` | `auto` | `auto`, `mps`, `cpu`, or `cuda` |
| `--width` / `--height` | `1280` / `720` | Requested capture size |
| `--alpha` | `0.5` | Overlay opacity, from 0 (video only) to 1 (mask only) |
| `--max-index` | `6` | Highest index to probe |

Street scenes can use the Cityscapes checkpoint:

```bash
uv run segformer-webcam --model nvidia/segformer-b0-finetuned-cityscapes-1024-1024
```

`--device cpu` forces the CPU if you want to compare it with Metal.
