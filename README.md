# transport-rf-detr-mcp

An MCP server that runs RF-DETR keypoint, segmentation and detection models to check generated character images.

## What it is for

It gives an agent three tools over the Model Context Protocol: a pose check that reports arm angles, whether head and feet are in frame and whether the figure holds a T-pose, instance segmentation, and plain detection. It runs on the CPU so the GPU stays free for the image generator, and it downloads each model's weights on first use.

## Run it

Install `rfdetr` and `mcp` in a Python environment, then register the server with an MCP client as a stdio command:

```sh
python server.py
```

## Licence

The licence is not stated.
