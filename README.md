# EinsteinVision

Modular computer vision and 3D visualization pipeline scaffold for monocular driving video.

## Directory Structure

```text
.
├── README.md
└── src/
	└── einsteinvision/
		├── __init__.py
		├── preprocessing.py
		├── perception.py
		├── fusion.py
		└── blender_render.py
```

## Modules

- `preprocessing.py`: Camera calibration + frame undistortion interfaces.
- `perception.py`: Perception orchestrator with pluggable lane/object/depth/flow backends.
- `fusion.py`: 2D pixel + depth fusion into 3D world coordinates, with JSON export.
- `blender_render.py`: Blender (`bpy`) JSON playback and keyframing scaffold.

