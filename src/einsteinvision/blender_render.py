"""Blender-side renderer for EinsteinVision fused trajectory JSON data.

Run from Blender, for example:
    blender --python src/einsteinvision/blender_render.py -- --json data/output/fused_scene.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import bpy  # type: ignore
except ImportError:  # pragma: no cover - expected outside Blender runtime
    bpy = None


@dataclass(slots=True)
class BlenderRenderConfig:
    """Settings for mapping fused objects to Blender assets."""

    assets_dir: Path
    class_to_asset: dict[str, str]
    collection_name: str = "EinsteinVisionActors"


class BlenderSceneRenderer:
    """Loads fused frame records and keyframes object transforms in Blender."""

    def __init__(self, config: BlenderRenderConfig) -> None:
        self.config = config
        self._object_cache: dict[str, Any] = {}

    def render_from_json(self, json_path: str | Path) -> None:
        """Reads fused scene JSON and applies frame-by-frame keyframes."""

        self._require_bpy()

        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        frames = payload.get("frames", [])

        collection = self.ensure_collection(self.config.collection_name)
        for frame in frames:
            frame_index = int(frame["frame_index"])
            for obj in frame.get("objects", []):
                object_ref = self.get_or_create_actor(
                    track_id=str(obj["object_id"]),
                    class_name=str(obj["class_name"]),
                    collection=collection,
                )
                self.apply_keyframe(
                    blender_object=object_ref,
                    frame_index=frame_index,
                    position_xyz=obj["position_xyz_m"],
                    yaw_rad=obj.get("orientation_yaw_rad"),
                )

    def ensure_collection(self, name: str):
        """Ensures a target collection exists in the active Blender scene."""

        self._require_bpy()
        collection = bpy.data.collections.get(name)
        if collection is None:
            collection = bpy.data.collections.new(name)
            bpy.context.scene.collection.children.link(collection)
        return collection

    def get_or_create_actor(self, track_id: str, class_name: str, collection):
        """Returns actor object for track ID, creating it from mapped asset if needed."""

        self._require_bpy()

        if track_id in self._object_cache:
            return self._object_cache[track_id]

        actor = self.instantiate_asset(class_name=class_name, actor_name=track_id, collection=collection)
        self._object_cache[track_id] = actor
        return actor

    def instantiate_asset(self, class_name: str, actor_name: str, collection):
        """Instantiates an asset for a semantic class.

        Note:
            Asset import/append internals are intentionally left as a stub.
            You can implement this using `bpy.ops.wm.append`, `bpy.ops.import_scene.*`,
            or Blender's Asset Browser APIs.
        """

        self._require_bpy()

        asset_rel_path = self.config.class_to_asset.get(class_name)
        if asset_rel_path is None:
            bpy.ops.mesh.primitive_cube_add(size=1.0)
            actor = bpy.context.active_object
            actor.name = actor_name
            collection.objects.link(actor)
            bpy.context.scene.collection.objects.unlink(actor)
            return actor

        raise NotImplementedError(
            f"Asset loading for class '{class_name}' mapped to '{asset_rel_path}' is pending."
        )

    def apply_keyframe(
        self,
        blender_object,
        frame_index: int,
        position_xyz: list[float] | tuple[float, float, float],
        yaw_rad: float | None,
    ) -> None:
        """Applies location and yaw keyframes for one frame."""

        self._require_bpy()

        blender_object.location = (float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2]))
        if yaw_rad is not None:
            blender_object.rotation_mode = "XYZ"
            blender_object.rotation_euler[2] = float(yaw_rad)

        blender_object.keyframe_insert(data_path="location", frame=frame_index)
        blender_object.keyframe_insert(data_path="rotation_euler", frame=frame_index)

    @staticmethod
    def _require_bpy() -> None:
        """Ensures script is running inside Blender with bpy available."""

        if bpy is None:
            raise RuntimeError("This module must be executed inside Blender where 'bpy' is available.")


def _build_cli_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for Blender script execution."""

    parser = argparse.ArgumentParser(description="EinsteinVision Blender renderer")
    parser.add_argument("--json", dest="json_path", required=True, help="Path to fused scene JSON")
    parser.add_argument("--assets-dir", dest="assets_dir", default="assets", help="Directory with 3D assets")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point when executed as a Blender Python script."""

    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    config = BlenderRenderConfig(
        assets_dir=Path(args.assets_dir),
        class_to_asset={
            "vehicle": "vehicles/generic_car.blend",
            "pedestrian": "pedestrians/generic_pedestrian.blend",
            "traffic_light": "traffic/traffic_light.blend",
            "sign": "traffic/sign.blend",
        },
    )
    renderer = BlenderSceneRenderer(config=config)
    renderer.render_from_json(args.json_path)


if __name__ == "__main__":
    main()
