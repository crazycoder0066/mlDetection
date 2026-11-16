# mlDetection

Detects whether a person is present in an image using [YOLO](https://docs.ultralytics.com/) (`yolo11n`, via `ultralytics`).

## Requirements

- [uv](https://docs.astral.sh/uv/)

uv installs the pinned Python (3.15) and the dependencies automatically. PyTorch is installed as the CPU-only build (configured in `pyproject.toml`), so no GPU is needed.

## Setup

```bash
uv sync
```

The first run downloads the model weights (`yolo11n.pt`, about 5 MB) into the project folder, so it needs internet access once.

## Usage

```bash
uv run detect_person.py path/to/image.jpg
```

Options:

| Flag | Description |
| --- | --- |
| `--conf 0.25` | Confidence threshold (default 0.25). Raise it to reduce false positives. |
| `--output out.jpg` | Save the image with boxes and confidence scores drawn. |
| `--show` | Open a window with the result (needs a display; not for headless machines). |

Example:

```bash
uv run detect_person.py photo.jpg --conf 0.4 --output result.jpg
```

### Exit codes

| Code | Meaning | Output |
| --- | --- | --- |
| 0 | Person found | `Person detected (N found)` |
| 1 | No person found | `No person detected` |
| 2 | Error (e.g. unreadable image) | `Error: ...` on stderr |

## Project layout

- `detect_person.py` – the CLI script (`detect_people()` returns `(x1, y1, x2, y2, confidence)` boxes)
- `pyproject.toml` / `uv.lock` – dependencies
- `.python-version` – pinned Python version
