# rf-detr-mcp

MCP server wrapping [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow,
Apache 2.0) for objective QA of generated character art — the "discriminator"
side of the GAN-style dataset loop.

## Tools

| Tool | Model | Purpose |
|---|---|---|
| `analyze_pose` | RFDETRKeypointPreview (COCO-17) | arm angles vs horizontal, feet/head in frame, strict T-pose verdict |
| `segment_image` | RFDETRSegMedium | instance masks (class, box, area; optional mask PNG export) |
| `detect_objects` | RFDETRMedium | plain detection boxes |

## Validation (2026-07-17)

Tested on stylized 3D-anime foxgirl renders (z-image-turbo): keypoint
confidence ≥ 0.75, and strict-T verdicts matched human review 4/4
(passes measured 4–9° from horizontal; failures 38–51°).

## Setup

```
uv venv --python 3.11
uv pip install --python .venv rfdetr "mcp[cli]"
```

Register with Claude Code:

```
claude mcp add rf-detr -- C:\Users\ernes\Desktop\rf-detr-mcp\.venv\Scripts\python C:\Users\ernes\Desktop\rf-detr-mcp\server.py
```

Runs on CPU deliberately — the GPU is assumed occupied by the diffusion
backend. First call per model downloads weights to `~/.roboflow/models/`.
